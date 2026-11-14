// free-model-aggregator-01: one server, two client wires, ten free-tier providers.
//
// Serves the OpenAI Chat Completions API and the Anthropic Messages API (JSON and
// SSE) by routing each request to the best free model that meets its requirements,
// across providers that speak the real OpenAI / Anthropic / Gemini wires, with
// failover when a provider refuses (402/429/5xx). The catalog it registers is the
// one `--list` prints, byte for byte.
//
// Deterministic: no wall clock, no unseeded randomness. Every choice is a function
// of the brief, the requests and the providers' answers, so a replay leaves the
// same state export.
//
// Usage (Node 22, stdlib only):
//   node main.ts            # trial mode: reads $MAKEFUN_PROMPT_FILE, probes the
//                           # providers under $MAKEFUN_ENV_URL, registers, serves,
//                           # exits 0 when GET /work says done
//   node main.ts --list     # print the catalog document and exit 0

import { createServer, type IncomingMessage, type Server, type ServerResponse } from 'node:http';
import { readFileSync } from 'node:fs';

const PROTOCOL_VERSION = 1;
const POLL_INTERVAL_MS = 100;
const PROBE_TIMEOUT_MS = 5000;
const UPSTREAM_TIMEOUT_MS = 10000;
const MAX_BODY_BYTES = 32 * 1024 * 1024;
const FIXED_CREATED = 1767225600;
const FIXED_ANTHROPIC_CREATED_AT = '2026-01-01T00:00:00Z';

const ENV_URL = (process.env.MAKEFUN_ENV_URL ?? '').replace(/\/+$/, '');
const PROMPT_FILE = process.env.MAKEFUN_PROMPT_FILE ?? '';

type Wire = 'openai' | 'anthropic' | 'gemini';
type AuthScheme = 'bearer' | 'x-api-key' | 'x-goog-api-key' | 'none';
type Json = Record<string, unknown>;

const isRecord = (v: unknown): v is Json => v !== null && typeof v === 'object' && !Array.isArray(v);
const strOf = (v: unknown): string => (typeof v === 'string' ? v : '');
const numOrNull = (v: unknown): number | null =>
  typeof v === 'number' && Number.isFinite(v) ? v : null;
const numOrZero = (v: unknown): number => (typeof v === 'number' && Number.isFinite(v) ? v : 0);

function log(message: string): void {
  process.stderr.write(`aggregator: ${message}\n`);
}

// ---------------------------------------------------------------------------
// The brief: three JSON documents found by their `schema` member.
// ---------------------------------------------------------------------------

interface ProviderEntry {
  id: string;
  name: string;
  wire: Wire;
  auth: AuthScheme;
  base: string;
  key_env: string;
  key: string;
}

interface RatingRow {
  model: string;
  provider: string;
  rating_milli: number;
  context_length: number | null;
  tools: boolean;
  vision: boolean;
  json: boolean;
  free: boolean;
}

interface Requirement {
  tools?: boolean;
  vision?: boolean;
  json?: boolean;
  min_context?: number;
}

interface WorkloadRequest {
  id: string;
  api: 'openai' | 'anthropic';
  stream: boolean;
  weight: 1 | 2;
  requires: Requirement;
}

interface Brief {
  providers: ProviderEntry[];
  ratings: RatingRow[];
  workload: WorkloadRequest[];
  error: string | null;
}

function objectEnd(text: string, start: number): number {
  let depth = 0;
  let inString = false;
  for (let i = start; i < text.length; i++) {
    const ch = text[i];
    if (inString) {
      if (ch === '\\') i += 1;
      else if (ch === '"') inString = false;
    } else if (ch === '"') inString = true;
    else if (ch === '{') depth += 1;
    else if (ch === '}') {
      depth -= 1;
      if (depth === 0) return i;
    }
  }
  return -1;
}

function documentsIn(text: string): Json[] {
  const out: Json[] = [];
  for (let start = text.indexOf('{'); start !== -1; start = text.indexOf('{', start + 1)) {
    const end = objectEnd(text, start);
    if (end === -1) continue;
    try {
      const value: unknown = JSON.parse(text.slice(start, end + 1));
      if (isRecord(value) && typeof value.schema === 'string') out.push(value);
    } catch {
      // Not JSON on its own: look on.
    }
  }
  return out;
}

function providerOf(v: unknown): ProviderEntry | null {
  if (!isRecord(v)) return null;
  const id = strOf(v.id);
  const wire = v.wire === 'openai' || v.wire === 'anthropic' || v.wire === 'gemini' ? v.wire : null;
  if (id === '' || wire === null) return null;
  const auth =
    v.auth === 'bearer' || v.auth === 'x-api-key' || v.auth === 'x-goog-api-key' || v.auth === 'none'
      ? v.auth
      : 'bearer';
  const base =
    strOf(v.path) !== ''
      ? strOf(v.path)
      : strOf(v.base_path) !== ''
        ? strOf(v.base_path)
        : strOf(v.base_url) !== ''
          ? strOf(v.base_url)
          : strOf(v.base);
  return {
    id,
    name: strOf(v.name) !== '' ? strOf(v.name) : id,
    wire,
    auth,
    base,
    key_env: strOf(v.key_env),
    key: strOf(v.key),
  };
}

function readBrief(path: string): Brief {
  const empty: Brief = { providers: [], ratings: [], workload: [], error: null };
  if (path === '') return { ...empty, error: 'MAKEFUN_PROMPT_FILE is not set' };
  let text: string;
  try {
    text = readFileSync(path, 'utf8');
  } catch (err) {
    return { ...empty, error: `cannot read ${path}: ${(err as Error).message}` };
  }
  const docs = documentsIn(text);
  const dirDoc = docs.find((d) => d.schema === 'free-model-aggregator-01/providers/v1' && Array.isArray(d.providers)) ?? null;
  if (dirDoc === null || !Array.isArray(dirDoc.providers)) {
    return { ...empty, error: 'the brief holds no provider directory document' };
  }
  const providers = dirDoc.providers.flatMap((v) => {
    const p = providerOf(v);
    return p === null ? [] : [p];
  });
  const ratingsDoc = docs.find((d) => d.schema === 'free-model-aggregator-01/ratings/v1' && Array.isArray(d.ratings)) ?? null;
  const ratings: RatingRow[] =
    ratingsDoc === null || !Array.isArray(ratingsDoc.ratings)
      ? []
      : ratingsDoc.ratings.flatMap((v) => {
          if (!isRecord(v) || strOf(v.model) === '' || strOf(v.provider) === '') return [];
          return [
            {
              model: strOf(v.model),
              provider: strOf(v.provider),
              rating_milli: numOrZero(v.rating_milli),
              context_length: numOrNull(v.context_length),
              tools: v.tools === true,
              vision: v.vision === true,
              json: v.json === true,
              free: v.free === true,
            },
          ];
        });
  const workloadDoc = docs.find((d) => d.schema === 'free-model-aggregator-01/workload/v1' && Array.isArray(d.manifest)) ?? null;
  const workload: WorkloadRequest[] =
    workloadDoc === null || !Array.isArray(workloadDoc.manifest)
      ? []
      : workloadDoc.manifest.flatMap((v) => {
          if (!isRecord(v) || strOf(v.id) === '') return [];
          const req = isRecord(v.requires) ? v.requires : {};
          return [
            {
              id: strOf(v.id),
              api: v.api === 'anthropic' ? 'anthropic' : 'openai',
              stream: v.stream === true,
              weight: v.weight === 2 ? 2 : 1,
              requires: {
                tools: req.tools === true,
                vision: req.vision === true,
                json: req.json === true,
                ...(typeof req.min_context === 'number' && Number.isFinite(req.min_context)
                  ? { min_context: req.min_context }
                  : {}),
              },
            },
          ];
        });
  return { providers, ratings, workload, error: null };
}

function ratingOf(brief: Brief, providerId: string, modelId: string): RatingRow | null {
  return brief.ratings.find((r) => r.provider === providerId && r.model === modelId) ?? null;
}

// ---------------------------------------------------------------------------
// Catalog probe: each provider's own wire, its own auth header.
// ---------------------------------------------------------------------------

interface ProbedModel {
  id: string;
  display_name: string;
  free: boolean | null;
  context_length: number | null;
  tools: boolean | null;
  vision: boolean | null;
  json: boolean | null;
}

interface Probe {
  provider: ProviderEntry;
  ok: boolean;
  status: number;
  models: ProbedModel[];
  error: string | null;
}

interface CatalogModel {
  id: string;
  free: boolean;
  context_length: number | null;
}

interface CatalogProvider {
  id: string;
  base_url: string;
  key_env: string;
  wire: Wire;
  models: CatalogModel[];
}

interface CatalogReport {
  providers: CatalogProvider[];
}

function joinBase(envUrl: string, base: string): string {
  const b = base.trim();
  if (/^https?:\/\//i.test(b)) return b.replace(/\/+$/, '');
  const prefix = envUrl.replace(/\/+$/, '');
  if (b === '') return prefix;
  return `${prefix}${b.startsWith('/') ? '' : '/'}${b.replace(/\/+$/, '')}`;
}

function authHeaders(auth: AuthScheme, key: string): Record<string, string> {
  if (key === '') return {};
  if (auth === 'bearer') return { authorization: `Bearer ${key}` };
  if (auth === 'x-api-key') return { 'x-api-key': key };
  if (auth === 'x-goog-api-key') return { 'x-goog-api-key': key };
  return {};
}

function zeroPriced(v: Json): boolean | null {
  const pricing = isRecord(v.pricing) ? v.pricing : null;
  if (pricing === null) return null;
  const values = Object.values(pricing).flatMap((x) => {
    if (typeof x === 'number') return [x];
    if (typeof x === 'string' && x.trim() !== '') return [Number(x)];
    return [];
  });
  if (values.length === 0) return null;
  return values.every((n) => Number.isFinite(n) && n === 0);
}

function modelIn(wire: Wire, v: unknown): ProbedModel | null {
  if (!isRecord(v)) return null;
  const id = wire === 'gemini' ? strOf(v.name).replace(/^models\//, '') : strOf(v.id);
  if (id === '') return null;
  return {
    id,
    display_name:
      strOf(v.display_name) !== ''
        ? strOf(v.display_name)
        : strOf(v.displayName) !== ''
          ? strOf(v.displayName)
          : id,
    free: typeof v.free === 'boolean' ? v.free : zeroPriced(v),
    context_length: numOrNull(v.context_length) ?? numOrNull(v.context_window) ?? numOrNull(v.inputTokenLimit),
    tools: typeof v.tools === 'boolean' ? v.tools : null,
    vision: typeof v.vision === 'boolean' ? v.vision : null,
    json: typeof v.json === 'boolean' ? v.json : null,
  };
}

function modelsIn(wire: Wire, body: unknown): ProbedModel[] {
  if (!isRecord(body)) return [];
  const list = wire === 'gemini' ? body.models : body.data;
  if (!Array.isArray(list)) return [];
  return list.flatMap((v) => {
    const m = modelIn(wire, v);
    return m === null ? [] : [m];
  });
}

async function probeProvider(envUrl: string, provider: ProviderEntry, timeoutMs = PROBE_TIMEOUT_MS): Promise<Probe> {
  const base = joinBase(envUrl, provider.base);
  const url = provider.wire === 'gemini' ? `${base}/v1beta/models` : `${base}/v1/models`;
  const headers: Record<string, string> = { accept: 'application/json', ...authHeaders(provider.auth, provider.key) };
  if (provider.wire === 'anthropic') headers['anthropic-version'] = '2023-06-01';
  try {
    const res = await fetch(url, { headers, signal: AbortSignal.timeout(timeoutMs) });
    if (!res.ok) return { provider, ok: false, status: res.status, models: [], error: `GET ${url} answered HTTP ${res.status}` };
    const body: unknown = await res.json();
    return { provider, ok: true, status: res.status, models: modelsIn(provider.wire, body), error: null };
  } catch (err) {
    return { provider, ok: false, status: 0, models: [], error: `GET ${url} failed: ${(err as Error).message}` };
  }
}

function modelFree(model: ProbedModel, row: RatingRow | null): boolean {
  if (row !== null) return row.free;
  return model.free ?? false;
}

interface Capabilities {
  context_length: number | null;
  tools: boolean | null;
  vision: boolean | null;
  json: boolean | null;
}

function capabilitiesOf(model: ProbedModel, row: RatingRow | null): Capabilities {
  return {
    context_length: row?.context_length ?? model.context_length ?? null,
    tools: row?.tools ?? model.tools ?? null,
    vision: row?.vision ?? model.vision ?? null,
    json: row?.json ?? model.json ?? null,
  };
}

function meetsRequirements(requires: Requirement, caps: Capabilities): boolean {
  if (requires.tools === true && caps.tools !== true) return false;
  if (requires.vision === true && caps.vision !== true) return false;
  if (requires.json === true && caps.json !== true) return false;
  if (requires.min_context !== undefined && (caps.context_length === null || caps.context_length < requires.min_context)) return false;
  return true;
}

// The catalog `--list` prints and the register body carries: the directory's own
// path as base_url (never joined to the env origin: the origin differs between a
// live pass and its replay), every model the listing named, free flag and context
// from the quality table where it has a row.
function buildCatalog(probes: readonly Probe[], brief: Brief): CatalogReport {
  const providers: CatalogProvider[] = [];
  for (const probe of probes) {
    if (!probe.ok) continue;
    const seen = new Set<string>();
    const models: CatalogModel[] = [];
    for (const model of [...probe.models].sort((a, b) => (a.id < b.id ? -1 : a.id > b.id ? 1 : 0))) {
      if (seen.has(model.id)) continue;
      seen.add(model.id);
      const row = ratingOf(brief, probe.provider.id, model.id);
      models.push({ id: model.id, free: modelFree(model, row), context_length: row?.context_length ?? model.context_length ?? null });
    }
    providers.push({ id: probe.provider.id, base_url: probe.provider.base, key_env: probe.provider.key_env, wire: probe.provider.wire, models });
  }
  return { providers };
}

// ---------------------------------------------------------------------------
// Wire translation. Every provider answer is first reduced to one normalized
// shape { text, toolCalls, usage, stopFamily }, then rendered in the wire the
// client used. The simulated providers only ever answer plain text with a
// stop-family-0 stop reason, but tool calls are carried through too, so the
// same code runs against the real services.
// Stop families: 0 stop/end_turn/STOP, 1 length/max_tokens, 2 tool use,
// 3 content filter / refusal.
// ---------------------------------------------------------------------------

interface ToolCall {
  id: string;
  name: string;
  args: string; // JSON string
}

interface Normalized {
  text: string;
  toolCalls: ToolCall[];
  usage: { input: number; output: number };
  stopFamily: number;
}

function stopFamilyOf(reason: unknown): number {
  if (reason === 'length' || reason === 'max_tokens' || reason === 'MAX_TOKENS') return 1;
  if (reason === 'tool_calls' || reason === 'function_call' || reason === 'tool_use') return 2;
  if (reason === 'content_filter' || reason === 'refusal' || reason === 'SAFETY') return 3;
  return 0;
}

function textOf(content: unknown): string {
  if (typeof content === 'string') return content;
  if (!Array.isArray(content)) return '';
  return content
    .flatMap((b) => (isRecord(b) && b.type === 'text' && typeof b.text === 'string' ? [b.text] : []))
    .join('\n');
}

function argsObject(value: unknown): Json {
  if (isRecord(value)) return value;
  if (typeof value === 'string' && value.trim() !== '') {
    try {
      const parsed: unknown = JSON.parse(value);
      if (isRecord(parsed)) return parsed;
    } catch {
      // Not JSON: an empty input.
    }
  }
  return {};
}

function splitPieces(text: string, size: number): string[] {
  if (text === '') return [];
  // A degenerate chunk size would loop forever (at += 0); fall back to
  // a single piece instead of hanging the stream renderer.
  if (!Number.isFinite(size) || size <= 0) return [text];
  const out: string[] = [];
  for (let at = 0; at < text.length; at += size) out.push(text.slice(at, at + size));
  return out;
}

// --- client OpenAI body -> upstream Anthropic body -------------------------

function openAIPartToAnthropic(block: Json): Json | null {
  if (block.type === 'text' && typeof block.text === 'string') return { type: 'text', text: block.text };
  if (block.type === 'image_url' && isRecord(block.image_url) && typeof block.image_url.url === 'string') {
    const url = block.image_url.url;
    const data = /^data:([^;,]+)?;base64,(.*)$/s.exec(url);
    if (data !== null) {
      return { type: 'image', source: { type: 'base64', media_type: data[1] ?? 'image/png', data: data[2] } };
    }
    return { type: 'image', source: { type: 'url', url } };
  }
  return null;
}

function openAIToAnthropic(body: Json, model: string): Json {
  const messages: Json[] = [];
  const systemParts: string[] = [];
  const source = Array.isArray(body.messages) ? body.messages : [];
  for (const raw of source) {
    if (!isRecord(raw)) continue;
    const role = strOf(raw.role);
    if (role === 'system') {
      const text = textOf(raw.content);
      if (text !== '') systemParts.push(text);
      continue;
    }
    if (role === 'assistant') {
      const blocks: Json[] = [];
      const text = textOf(raw.content);
      if (text !== '') blocks.push({ type: 'text', text });
      const calls = Array.isArray(raw.tool_calls) ? raw.tool_calls : [];
      for (const call of calls) {
        if (!isRecord(call) || !isRecord(call.function)) continue;
        blocks.push({
          type: 'tool_use',
          id: strOf(call.id) !== '' ? strOf(call.id) : 'toolu_unknown',
          name: strOf(call.function.name),
          input: argsObject(call.function.arguments),
        });
      }
      if (blocks.length > 0) messages.push({ role: 'assistant', content: blocks });
      continue;
    }
    if (role === 'tool') {
      messages.push({
        role: 'user',
        content: [{ type: 'tool_result', tool_use_id: strOf(raw.tool_call_id), content: textOf(raw.content) }],
      });
      continue;
    }
    // user (or anything else treated as user)
    if (typeof raw.content === 'string') {
      messages.push({ role: 'user', content: raw.content });
      continue;
    }
    if (!Array.isArray(raw.content)) continue;
    const blocks: Json[] = [];
    for (const block of raw.content) {
      if (!isRecord(block)) continue;
      if (block.type === 'tool_result') {
        blocks.push({
          type: 'tool_result',
          tool_use_id: strOf(block.tool_use_id),
          content: typeof block.content === 'string' ? block.content : textOf(block.content),
        });
        continue;
      }
      const part = openAIPartToAnthropic(block);
      if (part !== null) blocks.push(part);
    }
    if (blocks.length > 0) messages.push({ role: 'user', content: blocks });
  }
  const out: Json = { model, messages };
  if (systemParts.length > 0) out.system = systemParts.join('\n');
  if (typeof body.max_tokens === 'number') out.max_tokens = body.max_tokens;
  else if (typeof body.max_completion_tokens === 'number') out.max_tokens = body.max_completion_tokens;
  if (typeof body.temperature === 'number') out.temperature = body.temperature;
  if (typeof body.top_p === 'number') out.top_p = body.top_p;
  const stop = body.stop;
  if (typeof stop === 'string' && stop !== '') out.stop_sequences = [stop];
  else if (Array.isArray(stop) && stop.length > 0) out.stop_sequences = stop.filter((s) => typeof s === 'string');
  if (Array.isArray(body.tools)) {
    const tools = body.tools.flatMap((raw) => {
      if (!isRecord(raw) || !isRecord(raw.function) || typeof raw.function.name !== 'string') return [];
      return [
        {
          name: raw.function.name,
          description: typeof raw.function.description === 'string' ? raw.function.description : '',
          input_schema: isRecord(raw.function.parameters) ? raw.function.parameters : { type: 'object', properties: {} },
        },
      ];
    });
    if (tools.length > 0) out.tools = tools;
  }
  const choice = body.tool_choice;
  if (choice === 'auto') out.tool_choice = { type: 'auto' };
  else if (choice === 'none') out.tool_choice = { type: 'none' };
  else if (choice === 'required') out.tool_choice = { type: 'any' };
  else if (isRecord(choice) && isRecord(choice.function)) {
    out.tool_choice = { type: 'tool', name: strOf(choice.function.name) };
  }
  if (body.stream === true) out.stream = true;
  return out;
}

// --- client Anthropic body -> upstream OpenAI body --------------------------

function anthropicPartToOpenAI(block: Json): Json | null {
  if (block.type === 'text' && typeof block.text === 'string') return { type: 'text', text: block.text };
  if (block.type === 'image' && isRecord(block.source)) {
    const source = block.source;
    if (source.type === 'base64' && typeof source.data === 'string') {
      const media = typeof source.media_type === 'string' ? source.media_type : 'image/png';
      return { type: 'image_url', image_url: { url: `data:${media};base64,${source.data}` } };
    }
    if (source.type === 'url' && typeof source.url === 'string') return { type: 'image_url', image_url: { url: source.url } };
  }
  return null;
}

function toolResultText(content: unknown): string {
  if (typeof content === 'string') return content;
  if (Array.isArray(content)) {
    const text = textOf(content);
    return text !== '' ? text : JSON.stringify(content);
  }
  return content === undefined ? '' : JSON.stringify(content);
}

function anthropicToOpenAI(body: Json, model: string): Json {
  const messages: Json[] = [];
  const system = textOf(body.system);
  if (system !== '') messages.push({ role: 'system', content: system });
  const source = Array.isArray(body.messages) ? body.messages : [];
  for (const raw of source) {
    if (!isRecord(raw)) continue;
    const assistant = raw.role === 'assistant';
    const content = raw.content;
    if (typeof content === 'string') {
      messages.push({ role: assistant ? 'assistant' : 'user', content });
      continue;
    }
    if (!Array.isArray(content)) continue;
    if (assistant) {
      const toolCalls: Json[] = [];
      for (const block of content) {
        if (!isRecord(block) || block.type !== 'tool_use') continue;
        toolCalls.push({
          id: strOf(block.id) !== '' ? strOf(block.id) : '',
          type: 'function',
          function: {
            name: strOf(block.name),
            arguments: JSON.stringify(isRecord(block.input) ? block.input : {}),
          },
        });
      }
      const text = textOf(content);
      messages.push({
        role: 'assistant',
        content: text !== '' ? text : null,
        ...(toolCalls.length > 0 ? { tool_calls: toolCalls } : {}),
      });
      continue;
    }
    const parts: Json[] = [];
    for (const block of content) {
      if (!isRecord(block)) continue;
      if (block.type === 'tool_result') {
        messages.push({
          role: 'tool',
          tool_call_id: strOf(block.tool_use_id),
          content: toolResultText(block.content),
        });
        continue;
      }
      const part = anthropicPartToOpenAI(block);
      if (part !== null) parts.push(part);
    }
    if (parts.length > 0) messages.push({ role: 'user', content: parts });
  }
  const out: Json = { model, messages };
  if (typeof body.max_tokens === 'number') out.max_tokens = body.max_tokens;
  if (typeof body.temperature === 'number') out.temperature = body.temperature;
  if (typeof body.top_p === 'number') out.top_p = body.top_p;
  if (Array.isArray(body.stop_sequences) && body.stop_sequences.length > 0) out.stop = body.stop_sequences;
  if (Array.isArray(body.tools)) {
    const tools = body.tools.flatMap((raw) => {
      if (!isRecord(raw) || typeof raw.name !== 'string') return [];
      return [
        {
          type: 'function',
          function: {
            name: raw.name,
            description: typeof raw.description === 'string' ? raw.description : '',
            parameters: isRecord(raw.input_schema) ? raw.input_schema : { type: 'object', properties: {} },
          },
        },
      ];
    });
    if (tools.length > 0) out.tools = tools;
  }
  const choice = body.tool_choice;
  if (isRecord(choice)) {
    if (choice.type === 'auto') out.tool_choice = 'auto';
    else if (choice.type === 'any') out.tool_choice = 'required';
    else if (choice.type === 'tool' && typeof choice.name === 'string') {
      out.tool_choice = { type: 'function', function: { name: choice.name } };
    }
  }
  if (body.stream === true) {
    out.stream = true;
    out.stream_options = { include_usage: true };
  }
  return out;
}

// --- any client body -> upstream Gemini body --------------------------------

function geminiPartsOfOpenAIContent(content: unknown): Json[] {
  const parts: Json[] = [];
  if (typeof content === 'string') {
    if (content !== '') parts.push({ text: content });
    return parts;
  }
  if (!Array.isArray(content)) return parts;
  for (const block of content) {
    if (!isRecord(block)) continue;
    if (block.type === 'text' && typeof block.text === 'string') parts.push({ text: block.text });
    else if (block.type === 'image_url' && isRecord(block.image_url) && typeof block.image_url.url === 'string') {
      const url = block.image_url.url;
      const data = /^data:([^;,]+)?;base64,(.*)$/s.exec(url);
      if (data !== null) parts.push({ inlineData: { mimeType: data[1] ?? 'image/png', data: data[2] } });
      else parts.push({ fileData: { fileUri: url } });
    }
  }
  return parts;
}

function geminiPartsOfAnthropicContent(content: unknown): Json[] {
  const parts: Json[] = [];
  if (typeof content === 'string') {
    if (content !== '') parts.push({ text: content });
    return parts;
  }
  if (!Array.isArray(content)) return parts;
  for (const block of content) {
    if (!isRecord(block)) continue;
    if (block.type === 'text' && typeof block.text === 'string') parts.push({ text: block.text });
    else if (block.type === 'image' && isRecord(block.source)) {
      const source = block.source;
      if (source.type === 'base64' && typeof source.data === 'string') {
        parts.push({ inlineData: { mimeType: typeof source.media_type === 'string' ? source.media_type : 'image/png', data: source.data } });
      } else if (source.type === 'url' && typeof source.url === 'string') {
        parts.push({ fileData: { fileUri: source.url } });
      }
    }
  }
  return parts;
}

function geminiFunctionDeclarations(clientApi: 'openai' | 'anthropic', body: Json): Json[] | null {
  if (!Array.isArray(body.tools)) return null;
  const decls: Json[] = [];
  for (const raw of body.tools) {
    if (!isRecord(raw)) continue;
    if (clientApi === 'openai') {
      if (!isRecord(raw.function) || typeof raw.function.name !== 'string') continue;
      decls.push({
        name: raw.function.name,
        description: typeof raw.function.description === 'string' ? raw.function.description : '',
        parameters: isRecord(raw.function.parameters) ? raw.function.parameters : { type: 'object', properties: {} },
      });
    } else {
      if (typeof raw.name !== 'string') continue;
      decls.push({
        name: raw.name,
        description: typeof raw.description === 'string' ? raw.description : '',
        parameters: isRecord(raw.input_schema) ? raw.input_schema : { type: 'object', properties: {} },
      });
    }
  }
  return decls.length > 0 ? decls : null;
}

function toGemini(clientApi: 'openai' | 'anthropic', body: Json, model: string): Json {
  const contents: Json[] = [];
  const systemText = clientApi === 'openai' ? '' : textOf(body.system);
  // OpenAI system messages live in messages[]; collect them first.
  const source: Json[] = [];
  if (clientApi === 'openai') {
    for (const raw of Array.isArray(body.messages) ? body.messages : []) {
      if (isRecord(raw) && raw.role === 'system') {
        const text = textOf(raw.content);
        if (text !== '') source.push({ role: '__system__', content: text });
      } else if (isRecord(raw)) {
        source.push(raw);
      }
    }
  } else {
    for (const raw of Array.isArray(body.messages) ? body.messages : []) if (isRecord(raw)) source.push(raw);
  }
  const systemParts: Json[] = [];
  if (systemText !== '') systemParts.push({ text: systemText });
  for (const raw of source) {
    if (raw.role === '__system__') {
      const text = textOf(raw.content);
      if (text !== '') systemParts.push({ text });
      continue;
    }
    const role = strOf(raw.role);
    if (clientApi === 'openai') {
      if (role === 'assistant') {
        const parts = geminiPartsOfOpenAIContent(raw.content);
        const calls = Array.isArray(raw.tool_calls) ? raw.tool_calls : [];
        for (const call of calls) {
          if (!isRecord(call) || !isRecord(call.function)) continue;
          parts.push({ functionCall: { name: strOf(call.function.name), args: argsObject(call.function.arguments) } });
        }
        if (parts.length > 0) contents.push({ role: 'model', parts });
        continue;
      }
      if (role === 'tool') {
        contents.push({
          role: 'function',
          parts: [{ functionResponse: { name: '', response: { content: textOf(raw.content) } } }],
        });
        continue;
      }
      const parts = geminiPartsOfOpenAIContent(raw.content);
      if (parts.length > 0) contents.push({ role: 'user', parts });
    } else {
      if (role === 'assistant') {
        const parts: Json[] = [];
        if (Array.isArray(raw.content)) {
          for (const block of raw.content) {
            if (!isRecord(block)) continue;
            if (block.type === 'text' && typeof block.text === 'string') parts.push({ text: block.text });
            else if (block.type === 'tool_use') {
              parts.push({ functionCall: { name: strOf(block.name), args: isRecord(block.input) ? block.input : {} } });
            }
          }
        } else if (typeof raw.content === 'string' && raw.content !== '') {
          parts.push({ text: raw.content });
        }
        if (parts.length > 0) contents.push({ role: 'model', parts });
        continue;
      }
      // user: text/image parts stay user; tool_result blocks become function role.
      const userParts = geminiPartsOfAnthropicContent(raw.content);
      if (userParts.length > 0) contents.push({ role: 'user', parts: userParts });
      if (Array.isArray(raw.content)) {
        for (const block of raw.content) {
          if (!isRecord(block) || block.type !== 'tool_result') continue;
          contents.push({
            role: 'function',
            parts: [
              {
                functionResponse: {
                  name: '',
                  response: { content: toolResultText(block.content) },
                },
              },
            ],
          });
        }
      }
    }
  }
  const out: Json = { contents };
  if (systemParts.length > 0) out.systemInstruction = { parts: systemParts };
  const decls = geminiFunctionDeclarations(clientApi, body);
  if (decls !== null) out.tools = [{ functionDeclarations: decls }];
  const config: Json = {};
  const maxTokens =
    clientApi === 'openai'
      ? typeof body.max_tokens === 'number'
        ? body.max_tokens
        : typeof body.max_completion_tokens === 'number'
          ? body.max_completion_tokens
          : null
      : typeof body.max_tokens === 'number'
        ? body.max_tokens
        : null;
  if (maxTokens !== null) config.maxOutputTokens = maxTokens;
  if (typeof body.temperature === 'number') config.temperature = body.temperature;
  if (typeof body.top_p === 'number') config.topP = body.top_p;
  const stops: string[] =
    clientApi === 'openai'
      ? typeof body.stop === 'string'
        ? [body.stop]
        : Array.isArray(body.stop)
          ? body.stop.filter((s): s is string => typeof s === 'string')
          : []
      : Array.isArray(body.stop_sequences)
        ? body.stop_sequences.filter((s): s is string => typeof s === 'string')
        : [];
  if (stops.length > 0) config.stopSequences = stops;
  if (clientApi === 'openai' && isRecord(body.response_format)) {
    const format = body.response_format.type;
    if (format === 'json_object' || format === 'json_schema') config.responseMimeType = 'application/json';
  }
  if (Object.keys(config).length > 0) out.generationConfig = config;
  const choice = body.tool_choice;
  if (choice !== undefined) {
    let mode: string | null = null;
    if (clientApi === 'openai') {
      if (choice === 'required') mode = 'ANY';
      else if (choice === 'none') mode = 'NONE';
      else if (choice === 'auto') mode = 'AUTO';
      else if (isRecord(choice)) mode = 'ANY';
    } else if (isRecord(choice)) {
      if (choice.type === 'any' || choice.type === 'tool') mode = 'ANY';
      else if (choice.type === 'none') mode = 'NONE';
      else if (choice.type === 'auto') mode = 'AUTO';
    }
    if (mode !== null) out.toolConfig = { functionCallingConfig: { mode } };
  }
  void model;
  return out;
}

// --- upstream calls ----------------------------------------------------------

async function fetchUpstream(url: string, headers: Record<string, string>, body: Json): Promise<Response> {
  return fetch(url, {
    method: 'POST',
    headers: { 'content-type': 'application/json', ...headers },
    body: JSON.stringify(body),
    signal: AbortSignal.timeout(UPSTREAM_TIMEOUT_MS),
  });
}

// --- parsing provider answers into Normalized -------------------------------

function parseOpenAIJson(body: unknown): Normalized {
  const fallback: Normalized = { text: '', toolCalls: [], usage: { input: 0, output: 0 }, stopFamily: 0 };
  if (!isRecord(body)) return fallback;
  const choices = Array.isArray(body.choices) ? body.choices : [];
  const choice = isRecord(choices[0]) ? choices[0] : {};
  const message = isRecord(choice.message) ? choice.message : {};
  const text = typeof message.content === 'string' ? message.content : textOf(message.content);
  const toolCalls: ToolCall[] = [];
  for (const call of Array.isArray(message.tool_calls) ? message.tool_calls : []) {
    if (!isRecord(call) || !isRecord(call.function)) continue;
    toolCalls.push({
      id: strOf(call.id) !== '' ? strOf(call.id) : 'call_unknown',
      name: strOf(call.function.name),
      args: typeof call.function.arguments === 'string' ? call.function.arguments : JSON.stringify(call.function.arguments ?? {}),
    });
  }
  const usage = isRecord(body.usage) ? body.usage : {};
  const stopFamily = stopFamilyOf(choice.finish_reason);
  return {
    text,
    toolCalls,
    usage: { input: numOrZero(usage.prompt_tokens), output: numOrZero(usage.completion_tokens) },
    stopFamily: toolCalls.length > 0 && stopFamily === 0 ? 2 : stopFamily,
  };
}

function sseFrames(text: string): { event: string | null; data: string }[] {
  const frames: { event: string | null; data: string }[] = [];
  const normalized = text.replace(/\r\n/g, '\n');
  for (const frame of normalized.split('\n\n')) {
    if (frame.trim() === '') continue;
    let event: string | null = null;
    const data: string[] = [];
    for (const line of frame.split('\n')) {
      if (line.startsWith('event:')) event = line.slice('event:'.length).trim();
      else if (line.startsWith('data:')) data.push(line.slice('data:'.length).replace(/^ /, ''));
    }
    if (data.length > 0) frames.push({ event, data: data.join('\n') });
  }
  return frames;
}

function parseOpenAIStream(text: string): Normalized {
  const out: Normalized = { text: '', toolCalls: [], usage: { input: 0, output: 0 }, stopFamily: 0 };
  const byIndex = new Map<number, ToolCall>();
  for (const frame of sseFrames(text)) {
    const payload = frame.data.trim();
    if (payload === '[DONE]') continue;
    let chunk: unknown;
    try {
      chunk = JSON.parse(payload);
    } catch {
      continue;
    }
    if (!isRecord(chunk)) continue;
    if (isRecord(chunk.usage)) {
      out.usage = { input: numOrZero(chunk.usage.prompt_tokens), output: numOrZero(chunk.usage.completion_tokens) };
    }
    const choices = Array.isArray(chunk.choices) ? chunk.choices : [];
    const choice = isRecord(choices[0]) ? choices[0] : null;
    if (choice === null) continue;
    const delta = isRecord(choice.delta) ? choice.delta : {};
    if (typeof delta.content === 'string') out.text += delta.content;
    for (const call of Array.isArray(delta.tool_calls) ? delta.tool_calls : []) {
      if (!isRecord(call)) continue;
      const index = typeof call.index === 'number' ? call.index : 0;
      let entry = byIndex.get(index);
      if (entry === undefined) {
        const fn = isRecord(call.function) ? call.function : {};
        entry = {
          id: strOf(call.id) !== '' ? strOf(call.id) : `call_${index}`,
          name: strOf(fn.name),
          args: '',
        };
        byIndex.set(index, entry);
      }
      const fn = isRecord(call.function) ? call.function : {};
      if (typeof fn.name === 'string' && fn.name !== '') entry.name = fn.name;
      if (typeof call.id === 'string' && call.id !== '') entry.id = call.id;
      if (typeof fn.arguments === 'string') entry.args += fn.arguments;
    }
    if (typeof choice.finish_reason === 'string' && choice.finish_reason !== '') {
      out.stopFamily = stopFamilyOf(choice.finish_reason);
    }
  }
  out.toolCalls = [...byIndex.values()].sort((a, b) => (a.id < b.id ? -1 : a.id > b.id ? 1 : 0));
  if (out.toolCalls.length > 0 && out.stopFamily === 0) out.stopFamily = 2;
  return out;
}

function parseAnthropicJson(body: unknown): Normalized {
  const fallback: Normalized = { text: '', toolCalls: [], usage: { input: 0, output: 0 }, stopFamily: 0 };
  if (!isRecord(body)) return fallback;
  const content = Array.isArray(body.content) ? body.content : [];
  let text = '';
  const toolCalls: ToolCall[] = [];
  for (const block of content) {
    if (!isRecord(block)) continue;
    if (block.type === 'text' && typeof block.text === 'string') text += block.text;
    else if (block.type === 'tool_use') {
      toolCalls.push({
        id: strOf(block.id) !== '' ? strOf(block.id) : 'toolu_unknown',
        name: strOf(block.name),
        args: JSON.stringify(isRecord(block.input) ? block.input : {}),
      });
    }
  }
  const usage = isRecord(body.usage) ? body.usage : {};
  const stopFamily = stopFamilyOf(body.stop_reason);
  return {
    text,
    toolCalls,
    usage: { input: numOrZero(usage.input_tokens), output: numOrZero(usage.output_tokens) },
    stopFamily: toolCalls.length > 0 && stopFamily === 0 ? 2 : stopFamily,
  };
}

function parseAnthropicStream(text: string): Normalized {
  const out: Normalized = { text: '', toolCalls: [], usage: { input: 0, output: 0 }, stopFamily: 0 };
  const toolByIndex = new Map<number, ToolCall>();
  for (const frame of sseFrames(text)) {
    let data: unknown;
    try {
      data = JSON.parse(frame.data);
    } catch {
      continue;
    }
    if (!isRecord(data)) continue;
    switch (frame.event) {
      case 'message_start':
        if (isRecord(data.message) && isRecord(data.message.usage)) {
          out.usage.input = numOrZero(data.message.usage.input_tokens);
        }
        break;
      case 'content_block_start': {
        const index = numOrNull(data.index);
        if (index === null) break;
        const block = isRecord(data.content_block) ? data.content_block : {};
        if (block.type === 'tool_use') {
          toolByIndex.set(index, {
            id: strOf(block.id) !== '' ? strOf(block.id) : `toolu_${index}`,
            name: strOf(block.name),
            args: '',
          });
        }
        break;
      }
      case 'content_block_delta': {
        const index = numOrNull(data.index);
        if (index === null || !isRecord(data.delta)) break;
        if (data.delta.type === 'text_delta' && typeof data.delta.text === 'string') out.text += data.delta.text;
        else if (data.delta.type === 'input_json_delta' && typeof data.delta.partial_json === 'string') {
          const entry = toolByIndex.get(index);
          if (entry !== undefined) entry.args += data.delta.partial_json;
        }
        break;
      }
      case 'message_delta':
        if (isRecord(data.delta) && typeof data.delta.stop_reason === 'string') {
          out.stopFamily = stopFamilyOf(data.delta.stop_reason);
        }
        if (isRecord(data.usage)) out.usage.output = numOrZero(data.usage.output_tokens);
        break;
      default:
        break;
    }
  }
  out.toolCalls = [...toolByIndex.values()];
  if (out.toolCalls.length > 0 && out.stopFamily === 0) out.stopFamily = 2;
  return out;
}

function geminiTextAndCalls(body: unknown): { text: string; toolCalls: ToolCall[] } {
  let text = '';
  const toolCalls: ToolCall[] = [];
  if (!isRecord(body)) return { text, toolCalls };
  const candidates = Array.isArray(body.candidates) ? body.candidates : [];
  const candidate = isRecord(candidates[0]) ? candidates[0] : {};
  const content = isRecord(candidate.content) ? candidate.content : {};
  const parts = Array.isArray(content.parts) ? content.parts : [];
  for (const part of parts) {
    if (!isRecord(part)) continue;
    if (typeof part.text === 'string') text += part.text;
    else if (isRecord(part.functionCall)) {
      toolCalls.push({
        id: `call_${toolCalls.length}`,
        name: strOf(part.functionCall.name),
        args: JSON.stringify(isRecord(part.functionCall.args) ? part.functionCall.args : {}),
      });
    }
  }
  return { text, toolCalls };
}

function parseGeminiJson(body: unknown): Normalized {
  const { text, toolCalls } = geminiTextAndCalls(body);
  let usage = { input: 0, output: 0 };
  let stopFamily = 0;
  if (isRecord(body)) {
    if (isRecord(body.usageMetadata)) {
      usage = {
        input: numOrZero(body.usageMetadata.promptTokenCount),
        output: numOrZero(body.usageMetadata.candidatesTokenCount),
      };
    }
    const candidates = Array.isArray(body.candidates) ? body.candidates : [];
    const candidate = isRecord(candidates[0]) ? candidates[0] : {};
    stopFamily = stopFamilyOf(candidate.finishReason);
  }
  return { text, toolCalls, usage, stopFamily: toolCalls.length > 0 && stopFamily === 0 ? 2 : stopFamily };
}

function parseGeminiStream(text: string): Normalized {
  const out: Normalized = { text: '', toolCalls: [], usage: { input: 0, output: 0 }, stopFamily: 0 };
  for (const frame of sseFrames(text)) {
    let data: unknown;
    try {
      data = JSON.parse(frame.data);
    } catch {
      continue;
    }
    const { text: piece, toolCalls } = geminiTextAndCalls(data);
    out.text += piece;
    out.toolCalls.push(...toolCalls);
    if (isRecord(data)) {
      if (isRecord(data.usageMetadata)) {
        out.usage = {
          input: numOrZero(data.usageMetadata.promptTokenCount),
          output: numOrZero(data.usageMetadata.candidatesTokenCount),
        };
      }
      const candidates = Array.isArray(data.candidates) ? data.candidates : [];
      const candidate = isRecord(candidates[0]) ? candidates[0] : {};
      if (typeof candidate.finishReason === 'string' && candidate.finishReason !== '') {
        out.stopFamily = stopFamilyOf(candidate.finishReason);
      }
    }
  }
  if (out.toolCalls.length > 0 && out.stopFamily === 0) out.stopFamily = 2;
  return out;
}

// ---------------------------------------------------------------------------
// Rendering a normalized answer in the client's wire.
// ---------------------------------------------------------------------------

function sseEvent(event: string | null, data: unknown): string {
  return `${event === null ? '' : `event: ${event}\n`}data: ${JSON.stringify(data)}\n\n`;
}

function openAIFinish(family: number): string {
  if (family === 1) return 'length';
  if (family === 2) return 'tool_calls';
  if (family === 3) return 'content_filter';
  return 'stop';
}

function anthropicStop(family: number): string {
  if (family === 1) return 'max_tokens';
  if (family === 2) return 'tool_use';
  if (family === 3) return 'refusal';
  return 'end_turn';
}

function renderOpenAIJson(answer: Normalized, model: string): Json {
  const message: Json = { role: 'assistant', content: answer.text };
  if (answer.toolCalls.length > 0) {
    message.tool_calls = answer.toolCalls.map((c) => ({
      id: c.id,
      type: 'function',
      function: { name: c.name, arguments: c.args },
    }));
  }
  return {
    id: 'chatcmpl-agg',
    object: 'chat.completion',
    created: FIXED_CREATED,
    model,
    choices: [{ index: 0, message, finish_reason: openAIFinish(answer.stopFamily), logprobs: null }],
    usage: {
      prompt_tokens: answer.usage.input,
      completion_tokens: answer.usage.output,
      total_tokens: answer.usage.input + answer.usage.output,
    },
  };
}

function renderOpenAIStream(answer: Normalized, model: string): string {
  const frames: string[] = [];
  const base = { id: 'chatcmpl-agg', object: 'chat.completion.chunk', created: FIXED_CREATED, model };
  frames.push(
    sseEvent(null, { ...base, choices: [{ index: 0, delta: { role: 'assistant', content: '' }, finish_reason: null }] }),
  );
  for (const piece of splitPieces(answer.text, 32)) {
    frames.push(sseEvent(null, { ...base, choices: [{ index: 0, delta: { content: piece }, finish_reason: null }] }));
  }
  answer.toolCalls.forEach((call, i) => {
    frames.push(
      sseEvent(null, {
        ...base,
        choices: [
          {
            index: 0,
            delta: {
              tool_calls: [
                { index: i, id: call.id, type: 'function', function: { name: call.name, arguments: call.args } },
              ],
            },
            finish_reason: null,
          },
        ],
      }),
    );
  });
  frames.push(
    sseEvent(null, {
      ...base,
      choices: [{ index: 0, delta: {}, finish_reason: openAIFinish(answer.stopFamily) }],
    }),
  );
  frames.push(
    sseEvent(null, {
      ...base,
      choices: [],
      usage: {
        prompt_tokens: answer.usage.input,
        completion_tokens: answer.usage.output,
        total_tokens: answer.usage.input + answer.usage.output,
      },
    }),
  );
  frames.push('data: [DONE]\n\n');
  return frames.join('');
}

function renderAnthropicJson(answer: Normalized, model: string): Json {
  const content: Json[] = [];
  if (answer.text !== '') content.push({ type: 'text', text: answer.text });
  for (const call of answer.toolCalls) {
    content.push({ type: 'tool_use', id: call.id, name: call.name, input: argsObject(call.args) });
  }
  if (content.length === 0) content.push({ type: 'text', text: '' });
  return {
    id: 'msg_agg',
    type: 'message',
    role: 'assistant',
    model,
    content,
    stop_reason: anthropicStop(answer.stopFamily),
    stop_sequence: null,
    usage: { input_tokens: answer.usage.input, output_tokens: answer.usage.output },
  };
}

function renderAnthropicStream(answer: Normalized, model: string): string {
  const frames: string[] = [];
  frames.push(
    sseEvent('message_start', {
      type: 'message_start',
      message: {
        id: 'msg_agg',
        type: 'message',
        role: 'assistant',
        model,
        content: [],
        stop_reason: null,
        stop_sequence: null,
        usage: { input_tokens: answer.usage.input, output_tokens: 0 },
      },
    }),
  );
  let index = 0;
  frames.push(
    sseEvent('content_block_start', {
      type: 'content_block_start',
      index,
      content_block: { type: 'text', text: '' },
    }),
  );
  for (const piece of splitPieces(answer.text, 32)) {
    frames.push(
      sseEvent('content_block_delta', {
        type: 'content_block_delta',
        index,
        delta: { type: 'text_delta', text: piece },
      }),
    );
  }
  frames.push(sseEvent('content_block_stop', { type: 'content_block_stop', index }));
  index += 1;
  for (const call of answer.toolCalls) {
    frames.push(
      sseEvent('content_block_start', {
        type: 'content_block_start',
        index,
        content_block: { type: 'tool_use', id: call.id, name: call.name, input: {} },
      }),
    );
    if (call.args !== '') {
      frames.push(
        sseEvent('content_block_delta', {
          type: 'content_block_delta',
          index,
          delta: { type: 'input_json_delta', partial_json: call.args },
        }),
      );
    }
    frames.push(sseEvent('content_block_stop', { type: 'content_block_stop', index }));
    index += 1;
  }
  frames.push(
    sseEvent('message_delta', {
      type: 'message_delta',
      delta: { stop_reason: anthropicStop(answer.stopFamily), stop_sequence: null },
      usage: { output_tokens: answer.usage.output },
    }),
  );
  frames.push(sseEvent('message_stop', { type: 'message_stop' }));
  return frames.join('');
}

// ---------------------------------------------------------------------------
// Routing: best free model first, failover across providers.
// ---------------------------------------------------------------------------

interface Candidate {
  probe: Probe;
  model: ProbedModel;
  rating: number;
}

interface RouteState {
  probes: Probe[];
  brief: Brief;
  manifestIndex: number;
  strikes: Map<string, number>;
  callsUsed: Map<string, number>;
}

function candidatesFor(state: RouteState, requires: Requirement, clientApi: 'openai' | 'anthropic', wantsJson: boolean): Candidate[] {
  const out: Candidate[] = [];
  for (const probe of state.probes) {
    if (!probe.ok) continue;
    for (const model of probe.models) {
      const row = ratingOf(state.brief, probe.provider.id, model.id);
      if (!modelFree(model, row)) continue;
      if (!meetsRequirements(requires, capabilitiesOf(model, row))) continue;
      // An Anthropic-wire provider cannot carry an OpenAI JSON-mode request:
      // its echo would report json_mode absent and the request would fail.
      if (wantsJson && clientApi === 'openai' && probe.provider.wire === 'anthropic') continue;
      out.push({ probe, model, rating: row?.rating_milli ?? 0 });
    }
  }
  out.sort((a, b) => {
    if (b.rating !== a.rating) return b.rating - a.rating;
    const usedA = state.callsUsed.get(a.probe.provider.id) ?? 0;
    const usedB = state.callsUsed.get(b.probe.provider.id) ?? 0;
    if (usedA !== usedB) return usedA - usedB;
    if (a.probe.provider.id !== b.probe.provider.id) return a.probe.provider.id < b.probe.provider.id ? -1 : 1;
    return a.model.id < b.model.id ? -1 : a.model.id > b.model.id ? 1 : 0;
  });
  return out;
}

function isDead(state: RouteState, providerId: string): boolean {
  return (state.strikes.get(providerId) ?? 0) >= 2;
}

function noteFailure(state: RouteState, providerId: string, status: number | null): void {
  if (status === 402) state.strikes.set(providerId, 2);
  else state.strikes.set(providerId, (state.strikes.get(providerId) ?? 0) + 1);
}

function noteSuccess(state: RouteState, providerId: string): void {
  state.strikes.set(providerId, 0);
  state.callsUsed.set(providerId, (state.callsUsed.get(providerId) ?? 0) + 1);
}

// One upstream attempt. Returns the normalized answer on HTTP 2xx, null on a
// refusal that failover may retry, or 'fatal' for a 400/401 that spends no
// budget and must not be retried on the same provider.
async function tryCandidate(
  candidate: Candidate,
  clientApi: 'openai' | 'anthropic',
  body: Json,
  stream: boolean,
): Promise<Normalized | null | 'fatal'> {
  const { probe } = candidate;
  const base = joinBase(ENV_URL, probe.provider.base);
  const headers = authHeaders(probe.provider.auth, probe.provider.key);
  try {
    let res: Response;
    if (probe.provider.wire === 'openai') {
      // An Anthropic client body must be rewritten to Chat Completions first:
      // sent raw, the provider would see no system message, no tools and no
      // stop sequences, and its echo would fail the client's checks.
      const upstream: Json =
        clientApi === 'anthropic'
          ? anthropicToOpenAI(body, candidate.model.id)
          : { ...body, model: candidate.model.id };
      if (stream) upstream.stream_options = { include_usage: true };
      res = await fetchUpstream(`${base}/v1/chat/completions`, headers, upstream);
      if (!res.ok) return res.status === 400 || res.status === 401 ? 'fatal' : null;
      const text = await res.text();
      return stream ? parseOpenAIStream(text) : parseOpenAIJson(await safeJson(text));
    }
    if (probe.provider.wire === 'anthropic') {
      const upstream = clientApi === 'anthropic' ? { ...body, model: candidate.model.id } : openAIToAnthropic(body, candidate.model.id);
      const anthHeaders = { ...headers, 'anthropic-version': '2023-06-01' };
      res = await fetchUpstream(`${base}/v1/messages`, anthHeaders, upstream);
      if (!res.ok) return res.status === 400 || res.status === 401 ? 'fatal' : null;
      const text = await res.text();
      return stream ? parseAnthropicStream(text) : parseAnthropicJson(await safeJson(text));
    }
    // gemini
    const upstream = toGemini(clientApi, body, candidate.model.id);
    const verb = stream ? ':streamGenerateContent' : ':generateContent';
    const url = `${base}/v1beta/models/${encodeURIComponent(candidate.model.id)}${verb}${stream ? '?alt=sse' : ''}`;
    res = await fetchUpstream(url, headers, upstream);
    if (!res.ok) return res.status === 400 || res.status === 401 ? 'fatal' : null;
    const text = await res.text();
    return stream ? parseGeminiStream(text) : parseGeminiJson(await safeJson(text));
  } catch (err) {
    log(`upstream ${probe.provider.id}/${candidate.model.id} failed: ${(err as Error).message}`);
    return null;
  }
}

async function safeJson(text: string): Promise<unknown> {
  try {
    return JSON.parse(text) as unknown;
  } catch {
    return null;
  }
}

// Serve one client request: try candidates in rating order, failing over on
// every refusal. Returns the answer and the model that served it, or null.
async function serveRequest(
  state: RouteState,
  clientApi: 'openai' | 'anthropic',
  body: Json,
  stream: boolean,
  requires: Requirement,
): Promise<{ answer: Normalized; model: string } | null> {
  const wantsJson =
    clientApi === 'openai' && isRecord(body.response_format) && (body.response_format.type === 'json_object' || body.response_format.type === 'json_schema');
  const candidates = candidatesFor(state, requires, clientApi, wantsJson);
  // First pass: providers not yet counted dead. Second pass: the dead ones too,
  // rather than leaving a request unserved.
  for (const pass of [0, 1]) {
    for (const candidate of candidates) {
      const dead = isDead(state, candidate.probe.provider.id);
      if ((pass === 0 && dead) || (pass === 1 && !dead)) continue;
      const upstreamBody: Json = clientApi === 'openai' ? { ...body, stream } : { ...body, stream };
      // For an OpenAI-wire provider the body goes through untouched; the
      // conversions happen inside tryCandidate for the other wires.
      const result = await tryCandidate(candidate, clientApi, upstreamBody, stream);
      if (result === 'fatal') continue;
      if (result === null) {
        noteFailure(state, candidate.probe.provider.id, null);
        continue;
      }
      noteSuccess(state, candidate.probe.provider.id);
      return { answer: result, model: candidate.model.id };
    }
  }
  return null;
}

// ---------------------------------------------------------------------------
// HTTP server: the two client APIs and the model list.
// ---------------------------------------------------------------------------

interface ServedModel {
  id: string;
  provider: string;
  display_name: string;
}

function servedModels(state: RouteState): ServedModel[] {
  const out: ServedModel[] = [];
  for (const probe of state.probes) {
    if (!probe.ok) continue;
    for (const model of [...probe.models].sort((a, b) => (a.id < b.id ? -1 : a.id > b.id ? 1 : 0))) {
      const row = ratingOf(state.brief, probe.provider.id, model.id);
      if (!modelFree(model, row)) continue;
      out.push({ id: model.id, provider: probe.provider.id, display_name: model.display_name });
    }
  }
  return out;
}

function requirementsFor(state: RouteState, body: Json): Requirement {
  const entry = state.brief.workload[state.manifestIndex];
  state.manifestIndex += 1;
  const fromBody = requirementsFromBody(body);
  if (entry === undefined) return fromBody;
  // The manifest is authoritative; the body's own signals can only add.
  return {
    tools: entry.requires.tools === true || fromBody.tools === true,
    vision: entry.requires.vision === true || fromBody.vision === true,
    json: entry.requires.json === true || fromBody.json === true,
    ...(entry.requires.min_context !== undefined ? { min_context: entry.requires.min_context } : {}),
  };
}

function requirementsFromBody(body: Json): Requirement {
  const messages = Array.isArray(body.messages) ? body.messages : [];
  let vision = false;
  for (const message of messages) {
    if (!isRecord(message) || !Array.isArray(message.content)) continue;
    for (const block of message.content) {
      if (isRecord(block) && (block.type === 'image' || block.type === 'image_url')) vision = true;
    }
  }
  const format = isRecord(body.response_format) ? body.response_format.type : undefined;
  return {
    tools: Array.isArray(body.tools) && body.tools.length > 0,
    vision,
    json: format === 'json_object' || format === 'json_schema',
  };
}

async function readJson(req: IncomingMessage): Promise<Json | null> {
  const chunks: Buffer[] = [];
  let size = 0;
  for await (const chunk of req) {
    const buffer = Buffer.isBuffer(chunk) ? chunk : Buffer.from(String(chunk));
    size += buffer.length;
    if (size > MAX_BODY_BYTES) return null;
    chunks.push(buffer);
  }
  if (chunks.length === 0) return null;
  try {
    const value: unknown = JSON.parse(Buffer.concat(chunks).toString('utf8'));
    return isRecord(value) ? value : null;
  } catch {
    return null;
  }
}

function sendJson(res: ServerResponse, status: number, body: unknown): void {
  const text = JSON.stringify(body);
  res.writeHead(status, { 'content-type': 'application/json', 'content-length': Buffer.byteLength(text) });
  res.end(text);
}

function sendSse(res: ServerResponse, text: string): void {
  res.writeHead(200, {
    'content-type': 'text/event-stream',
    'cache-control': 'no-cache',
    connection: 'keep-alive',
    'content-length': Buffer.byteLength(text),
  });
  res.end(text);
}

function openAIError(message: string, type: string, code: string | null): Json {
  return { error: { message, type, code } };
}

function anthropicError(type: string, message: string): Json {
  return { type: 'error', error: { type, message } };
}

async function handle(state: RouteState, req: IncomingMessage, res: ServerResponse): Promise<void> {
  const url = new URL(req.url ?? '/', 'http://entry');
  if (req.method === 'GET' && url.pathname === '/v1/models') {
    const list = servedModels(state);
    if (typeof req.headers['x-api-key'] === 'string') {
      sendJson(res, 200, {
        data: list.map((m) => ({ type: 'model', id: m.id, display_name: m.display_name, created_at: FIXED_ANTHROPIC_CREATED_AT })),
        has_more: false,
      });
    } else {
      sendJson(res, 200, {
        object: 'list',
        data: list.map((m) => ({ id: m.id, object: 'model', created: FIXED_CREATED, owned_by: m.provider })),
      });
    }
    return;
  }
  if (req.method === 'POST' && url.pathname === '/v1/chat/completions') {
    const body = await readJson(req);
    if (body === null) return sendJson(res, 400, openAIError('the request body is not one JSON object', 'invalid_request_error', null));
    const stream = body.stream === true;
    const served = await serveRequest(state, 'openai', body, stream, requirementsFor(state, body));
    if (served === null) {
      return sendJson(res, 503, openAIError('no free model could serve this request', 'server_error', null));
    }
    if (stream) return sendSse(res, renderOpenAIStream(served.answer, served.model));
    return sendJson(res, 200, renderOpenAIJson(served.answer, served.model));
  }
  if (req.method === 'POST' && url.pathname === '/v1/messages') {
    const body = await readJson(req);
    if (body === null) return sendJson(res, 400, anthropicError('invalid_request_error', 'the request body is not one JSON object'));
    const stream = body.stream === true;
    const served = await serveRequest(state, 'anthropic', body, stream, requirementsFor(state, body));
    if (served === null) {
      return sendJson(res, 503, anthropicError('overloaded_error', 'no free model could serve this request'));
    }
    if (stream) return sendSse(res, renderAnthropicStream(served.answer, served.model));
    return sendJson(res, 200, renderAnthropicJson(served.answer, served.model));
  }
  sendJson(res, 404, openAIError(`no route ${req.method ?? ''} ${url.pathname}`, 'not_found_error', null));
}

// ---------------------------------------------------------------------------
// Trial lifecycle: probe, register, serve, poll /work, exit 0 when done.
// ---------------------------------------------------------------------------

async function register(port: number, catalog: CatalogReport): Promise<void> {
  let lastError: string = '';
  for (let attempt = 0; attempt < 3; attempt++) {
    try {
      const res = await fetch(`${ENV_URL}/register`, {
        method: 'POST',
        headers: { 'content-type': 'application/json' },
        body: JSON.stringify({ protocol_version: PROTOCOL_VERSION, port, catalog }),
        signal: AbortSignal.timeout(5000),
      });
      if (!res.ok) throw new Error(`POST ${ENV_URL}/register answered HTTP ${res.status}: ${(await res.text()).slice(0, 500)}`);
      const reply: unknown = await res.json();
      if (isRecord(reply) && typeof reply.requests === 'number') log(`registered: the workload holds ${reply.requests} requests`);
      return;
    } catch (err) {
      lastError = (err as Error).message;
      await new Promise((resolve) => setTimeout(resolve, 100));
    }
  }
  throw new Error(`register failed: ${lastError}`);
}

function poll(server: Server): void {
  let inFlight = false;
  const timer = setInterval(() => {
    if (inFlight) return;
    inFlight = true;
    void (async () => {
      try {
        const res = await fetch(`${ENV_URL}/work`, { signal: AbortSignal.timeout(2000) });
        if (res.status !== 200) return;
        const work: unknown = await res.json();
        if (isRecord(work) && work.done === true) {
          clearInterval(timer);
          server.close();
          process.exit(0);
        }
      } catch {
        // The env is not reachable yet: try again on the next tick.
      } finally {
        inFlight = false;
      }
    })();
  }, POLL_INTERVAL_MS);
}

async function start(): Promise<void> {
  const brief = readBrief(PROMPT_FILE);
  if (brief.error !== null) log(brief.error);
  const probes = await Promise.all(brief.providers.map((provider) => probeProvider(ENV_URL, provider)));
  for (const probe of probes) if (!probe.ok && probe.error !== null) log(probe.error);
  const catalog = buildCatalog(probes, brief);
  if (process.argv.includes('--list')) {
    const text = `${JSON.stringify(catalog)}\n`;
    await new Promise<void>((resolve) => process.stdout.write(text, () => resolve()));
    process.exit(0);
  }
  if (ENV_URL === '') throw new Error('MAKEFUN_ENV_URL is not set');
  const state: RouteState = { probes, brief, manifestIndex: 0, strikes: new Map(), callsUsed: new Map() };
  const server = createServer((req, res) => {
    void handle(state, req, res).catch((err: unknown) => {
      log((err as Error).stack ?? String(err));
      if (!res.headersSent) {
        sendJson(res, 500, openAIError('the server failed to answer', 'internal_error', null));
      }
    });
  });
  await new Promise<void>((resolve, reject) => {
    server.once('error', reject);
    server.listen(0, '0.0.0.0', () => resolve());
  });
  const address = server.address();
  if (address === null || typeof address === 'string') throw new Error('the server has no port');
  const port = address.port;
  if (port === 8080) throw new Error('the OS picked 8080, which the env owns on one layout: restart the trial');
  log(`listening on 0.0.0.0:${port} with ${catalog.providers.length} answered providers and ${servedModels(state).length} free models`);
  await register(port, catalog);
  process.on('SIGTERM', () => process.exit(0));
  process.on('SIGINT', () => process.exit(0));
  poll(server);
}

try {
  await start();
} catch (err) {
  log((err as Error).stack ?? String(err));
  process.exit(1);
}
