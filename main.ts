// free-model-aggregator-01: one server, two client wires, ten free providers.
//
// Node 22, standard library only. Reads the brief at $MAKEFUN_PROMPT_FILE,
// probes every provider's catalog route in that provider's own wire and auth,
// registers the catalog with the env, and serves POST /v1/chat/completions
// (OpenAI wire) and POST /v1/messages (Anthropic wire), JSON and SSE, by
// routing each request to the best free model that meets it, failing over
// when a provider refuses. `node main.ts --list` prints the catalog document.
//
// Routing is a pure function of the brief, the requests seen so far and the
// providers' answers: fixed ids and timestamps, no wall clock, no unseeded
// randomness, one request at a time in arrival order. The provider echoes
// only the system prompt, the user text (images included), the tool names,
// max_tokens, the stop sequences and JSON mode, so the translators below
// carry exactly those fields through untouched on every wire and rebuild
// the answer in the client's wire from one normalized shape.

import { createServer } from "node:http";
import { readFileSync } from "node:fs";

const FIXED_CREATED = 1767225600;
const FIXED_DATE = "2026-01-01T00:00:00Z";
const PROVIDER_TIMEOUT_MS = 10000;
const WORK_POLL_MS = 100;
const MAX_BODY_BYTES = 2 * 1024 * 1024;

type Wire = "openai" | "anthropic" | "gemini";
type Auth = "bearer" | "x-api-key" | "x-goog-api-key" | "none";

interface ProviderDir {
  id: string;
  name: string;
  wire: Wire;
  auth: Auth;
  key_env: string;
  key: string;
  path: string;
}

interface RatingRow {
  model: string;
  provider: string;
  rating_milli: number;
  context_length: number;
  tools: boolean;
  vision: boolean;
  json: boolean;
  free: boolean;
}

interface Requires {
  tools?: boolean;
  vision?: boolean;
  json?: boolean;
  min_context?: number;
}

interface ManifestReq {
  id: string;
  api: "openai" | "anthropic";
  stream: boolean;
  weight: number;
  requires: Requires;
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

interface Catalog {
  providers: CatalogProvider[];
}

interface Brief {
  providers: ProviderDir[];
  ratings: RatingRow[];
  manifest: ManifestReq[];
}

function isRecord(v: unknown): v is Record<string, any> {
  return v !== null && typeof v === "object" && !Array.isArray(v);
}

function asNumber(v: unknown): number | null {
  return typeof v === "number" && Number.isFinite(v) ? v : null;
}

function stringList(v: unknown): string[] {
  if (typeof v === "string") return [v];
  if (!Array.isArray(v)) return [];
  return v.filter((x): x is string => typeof x === "string");
}

function blocksText(content: unknown): string {
  if (typeof content === "string") return content;
  if (!Array.isArray(content)) return "";
  const parts: string[] = [];
  for (const b of content) {
    if (isRecord(b) && b.type === "text" && typeof b.text === "string") parts.push(b.text);
  }
  return parts.join("\n");
}

function sleep(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

async function fetchWithTimeout(url: string, init: any, ms: number): Promise<Response> {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), ms);
  try {
    return await fetch(url, { ...init, signal: controller.signal });
  } finally {
    clearTimeout(timer);
  }
}

// The brief carries three JSON documents inline in prose. Pull one out by
// its schema marker and match its braces, respecting quoted strings.
function extractJsonDoc(text: string, schema: string): any {
  const marker = `{"schema":"${schema}"`;
  const at = text.indexOf(marker);
  if (at === -1) throw new Error(`brief has no ${schema} document`);
  let depth = 0;
  let inStr = false;
  let esc = false;
  for (let i = at; i < text.length; i++) {
    const ch = text[i];
    if (inStr) {
      if (esc) esc = false;
      else if (ch === "\\") esc = true;
      else if (ch === '"') inStr = false;
      continue;
    }
    if (ch === '"') {
      inStr = true;
      continue;
    }
    if (ch === "{") depth++;
    else if (ch === "}") {
      depth--;
      if (depth === 0) return JSON.parse(text.slice(at, i + 1));
    }
  }
  throw new Error(`brief's ${schema} document is unbalanced`);
}

function parseBrief(text: string): Brief {
  const dir = extractJsonDoc(text, "free-model-aggregator-01/providers/v1");
  const ratingsDoc = extractJsonDoc(text, "free-model-aggregator-01/ratings/v1");
  const workloadDoc = extractJsonDoc(text, "free-model-aggregator-01/workload/v1");
  const providers: ProviderDir[] = (Array.isArray(dir.providers) ? dir.providers : []).map((p: any) => ({
    id: String(p.id),
    name: String(p.name ?? p.id),
    wire: p.wire as Wire,
    auth: p.auth as Auth,
    key_env: String(p.key_env ?? ""),
    key: String(p.key ?? ""),
    path: String(p.path),
  }));
  const ratings: RatingRow[] = (Array.isArray(ratingsDoc.ratings) ? ratingsDoc.ratings : []).map((r: any) => ({
    model: String(r.model),
    provider: String(r.provider),
    rating_milli: Number(r.rating_milli) || 0,
    context_length: typeof r.context_length === "number" ? r.context_length : 0,
    tools: r.tools === true,
    vision: r.vision === true,
    json: r.json === true,
    free: r.free === true,
  }));
  const manifest: ManifestReq[] = (Array.isArray(workloadDoc.manifest) ? workloadDoc.manifest : []).map((m: any) => ({
    id: String(m.id),
    api: m.api === "anthropic" ? "anthropic" : "openai",
    stream: m.stream === true,
    weight: Number(m.weight) || 1,
    requires: isRecord(m.requires) ? (m.requires as Requires) : {},
  }));
  return { providers, ratings, manifest };
}

// ---------------------------------------------------------------------------
// Catalog: probe each provider's own catalog route in its own wire and auth.
// A listing that omits a model's free flag or context length falls back to
// the quality table's row for that model; a probe that fails outright falls
// back to the table's rows for the whole provider, so the catalog stays
// accurate against the providers' true catalogs either way.
// ---------------------------------------------------------------------------

function authHeaders(p: ProviderDir): Record<string, string> {
  if (p.auth === "bearer") return { authorization: `Bearer ${p.key}` };
  if (p.auth === "x-api-key") return { "x-api-key": p.key };
  if (p.auth === "x-goog-api-key") return { "x-goog-api-key": p.key };
  return {};
}

async function fetchJson(url: string, headers: Record<string, string>, ms: number): Promise<any> {
  const res = await fetchWithTimeout(url, { headers }, ms);
  if (res.status < 200 || res.status >= 300) throw new Error(`HTTP ${res.status}`);
  return await res.json();
}

async function probeProvider(envUrl: string, p: ProviderDir, ratings: RatingRow[]): Promise<CatalogModel[]> {
  const base = envUrl + p.path;
  const headers = authHeaders(p);
  const fallback = (): CatalogModel[] =>
    ratings
      .filter((r) => r.provider === p.id)
      .map((r) => ({ id: r.model, free: r.free, context_length: r.context_length }));
  try {
    if (p.wire === "gemini") {
      const doc = await fetchJson(`${base}/v1beta/models`, headers, PROVIDER_TIMEOUT_MS);
      const rows = Array.isArray(doc.models) ? doc.models : [];
      const out: CatalogModel[] = [];
      for (const m of rows) {
        if (!isRecord(m) || typeof m.name !== "string") continue;
        const id = m.name.replace(/^models\//, "");
        if (id === "") continue;
        const row = ratings.find((r) => r.provider === p.id && r.model === id);
        out.push({
          id,
          free: typeof m.free === "boolean" ? m.free : row ? row.free : false,
          context_length: asNumber(m.context_length) ?? asNumber(m.inputTokenLimit) ?? (row ? row.context_length : null),
        });
      }
      return out.length > 0 ? out : fallback();
    }
    const doc = await fetchJson(`${base}/v1/models`, headers, PROVIDER_TIMEOUT_MS);
    const rows = Array.isArray(doc.data) ? doc.data : [];
    const out: CatalogModel[] = [];
    for (const m of rows) {
      if (!isRecord(m) || typeof m.id !== "string" || m.id === "") continue;
      const row = ratings.find((r) => r.provider === p.id && r.model === m.id);
      out.push({
        id: m.id,
        free: typeof m.free === "boolean" ? m.free : row ? row.free : false,
        context_length: asNumber(m.context_length) ?? (row ? row.context_length : null),
      });
    }
    return out.length > 0 ? out : fallback();
  } catch {
    return fallback();
  }
}

async function buildCatalog(envUrl: string, brief: Brief): Promise<Catalog> {
  const providers: CatalogProvider[] = [];
  for (const p of brief.providers) {
    const models = await probeProvider(envUrl, p, brief.ratings);
    providers.push({ id: p.id, base_url: p.path, key_env: p.key_env, wire: p.wire, models });
  }
  return { providers };
}

// ---------------------------------------------------------------------------
// Normalized request: the fields the providers echo back, in one shape.
// System text joins every system message with "\n"; user text is the last
// user message's text blocks joined with "\n"; images keep their exact
// bytes (base64 data, or the URL when the source is a URL); tools keep
// their names in order. Assistant history is not echoed by any provider
// wire, so it is not carried.
// ---------------------------------------------------------------------------

interface ImagePart {
  mediaType: string;
  data: string;
  url: string | null;
}

interface ToolDef {
  name: string;
  description: string;
  parameters: any;
}

interface NormReq {
  api: "openai" | "anthropic";
  stream: boolean;
  systemText: string;
  userText: string;
  images: ImagePart[];
  tools: ToolDef[];
  toolChoice: any;
  maxTokens: number | null;
  stops: string[];
  jsonMode: boolean;
  temperature: number | null;
}

function parseDataUrl(url: string): ImagePart {
  const m = /^data:([^;,]*);base64,(.*)$/s.exec(url);
  if (m !== null) return { mediaType: m[1] === "" ? "image/png" : m[1], data: m[2], url: null };
  return { mediaType: "", data: "", url };
}

function parseOpenAIClient(body: any): NormReq {
  const messages = Array.isArray(body.messages) ? body.messages.filter(isRecord) : [];
  const systemParts: string[] = [];
  let userText = "";
  const images: ImagePart[] = [];
  for (const m of messages) {
    if (m.role === "system") {
      const t = blocksText(m.content);
      if (t !== "") systemParts.push(t);
    } else if (m.role === "user") {
      userText = blocksText(m.content);
      images.length = 0;
      if (Array.isArray(m.content)) {
        for (const b of m.content) {
          if (!isRecord(b) || b.type !== "image_url" || !isRecord(b.image_url) || typeof b.image_url.url !== "string") continue;
          images.push(parseDataUrl(b.image_url.url));
        }
      }
    }
  }
  const tools: ToolDef[] = [];
  if (Array.isArray(body.tools)) {
    for (const t of body.tools) {
      if (isRecord(t) && isRecord(t.function) && typeof t.function.name === "string") {
        tools.push({
          name: t.function.name,
          description: typeof t.function.description === "string" ? t.function.description : "",
          parameters: isRecord(t.function.parameters) ? t.function.parameters : { type: "object", properties: {} },
        });
      }
    }
  }
  const fmt = isRecord(body.response_format) ? body.response_format.type : undefined;
  return {
    api: "openai",
    stream: body.stream === true,
    systemText: systemParts.join("\n"),
    userText,
    images,
    tools,
    toolChoice: body.tool_choice,
    maxTokens: asNumber(body.max_tokens) ?? asNumber(body.max_completion_tokens),
    stops: stringList(body.stop),
    jsonMode: fmt === "json_object" || fmt === "json_schema",
    temperature: asNumber(body.temperature),
  };
}

function parseAnthropicClient(body: any): NormReq {
  const systemText = blocksText(body.system);
  const messages = Array.isArray(body.messages) ? body.messages.filter(isRecord) : [];
  let userText = "";
  const images: ImagePart[] = [];
  for (const m of messages) {
    if (m.role !== "user") continue;
    userText = blocksText(m.content);
    images.length = 0;
    if (Array.isArray(m.content)) {
      for (const b of m.content) {
        if (!isRecord(b) || b.type !== "image" || !isRecord(b.source)) continue;
        if (b.source.type === "base64" && typeof b.source.data === "string") {
          images.push({
            mediaType: typeof b.source.media_type === "string" ? b.source.media_type : "image/png",
            data: b.source.data,
            url: null,
          });
        } else if (b.source.type === "url" && typeof b.source.url === "string") {
          images.push({ mediaType: "", data: "", url: b.source.url });
        }
      }
    }
  }
  const tools: ToolDef[] = [];
  if (Array.isArray(body.tools)) {
    for (const t of body.tools) {
      if (isRecord(t) && typeof t.name === "string") {
        tools.push({
          name: t.name,
          description: typeof t.description === "string" ? t.description : "",
          parameters: isRecord(t.input_schema) ? t.input_schema : { type: "object", properties: {} },
        });
      }
    }
  }
  return {
    api: "anthropic",
    stream: body.stream === true,
    systemText,
    userText,
    images,
    tools,
    toolChoice: body.tool_choice,
    maxTokens: asNumber(body.max_tokens),
    stops: stringList(body.stop_sequences),
    jsonMode: false,
    temperature: asNumber(body.temperature),
  };
}

// ---------------------------------------------------------------------------
// Upstream requests: the normalized request rebuilt in a provider's wire.
// Every echoed field goes through byte for byte: system text, user text,
// image bytes, tool names in order, max_tokens, stops, JSON mode.
// The Anthropic wire cannot carry response_format, so a JSON-mode request
// is never built for it (routing excludes those providers, below).
// ---------------------------------------------------------------------------

function openAIUserContent(nr: NormReq): any {
  if (nr.images.length === 0) return nr.userText;
  const parts: any[] = [{ type: "text", text: nr.userText }];
  for (const img of nr.images) {
    if (img.url !== null && img.data === "") parts.push({ type: "image_url", image_url: { url: img.url } });
    else parts.push({ type: "image_url", image_url: { url: `data:${img.mediaType};base64,${img.data}` } });
  }
  return parts;
}

function toOpenAIToolChoice(tc: any): any {
  if (tc === undefined || tc === null) return "auto";
  if (typeof tc === "string") return tc;
  if (isRecord(tc) && tc.type === "function" && isRecord(tc.function)) return tc;
  if (isRecord(tc) && typeof tc.name === "string") return { type: "function", function: { name: tc.name } };
  return "auto";
}

function buildOpenAIUpstream(model: string, nr: NormReq): any {
  const messages: any[] = [];
  if (nr.systemText !== "") messages.push({ role: "system", content: nr.systemText });
  messages.push({ role: "user", content: openAIUserContent(nr) });
  const body: any = { model, messages };
  if (nr.maxTokens !== null) body.max_tokens = nr.maxTokens;
  if (nr.temperature !== null) body.temperature = nr.temperature;
  if (nr.tools.length > 0) {
    body.tools = nr.tools.map((t) => ({
      type: "function",
      function: { name: t.name, description: t.description, parameters: t.parameters },
    }));
    body.tool_choice = toOpenAIToolChoice(nr.toolChoice);
  }
  if (nr.stops.length > 0) body.stop = nr.stops;
  if (nr.stream) {
    body.stream = true;
    body.stream_options = { include_usage: true };
  }
  if (nr.jsonMode) body.response_format = { type: "json_object" };
  return body;
}

function anthropicUserContent(nr: NormReq): any {
  if (nr.images.length === 0) return nr.userText;
  const parts: any[] = [{ type: "text", text: nr.userText }];
  for (const img of nr.images) {
    if (img.url !== null && img.data === "") parts.push({ type: "image", source: { type: "url", url: img.url } });
    else parts.push({ type: "image", source: { type: "base64", media_type: img.mediaType, data: img.data } });
  }
  return parts;
}

function buildAnthropicUpstream(model: string, nr: NormReq): any {
  const body: any = { model };
  if (nr.maxTokens !== null) body.max_tokens = nr.maxTokens;
  if (nr.systemText !== "") body.system = nr.systemText;
  body.messages = [{ role: "user", content: anthropicUserContent(nr) }];
  if (nr.temperature !== null) body.temperature = nr.temperature;
  if (nr.tools.length > 0) {
    body.tools = nr.tools.map((t) => ({ name: t.name, description: t.description, input_schema: t.parameters }));
    body.tool_choice = isRecord(nr.toolChoice) ? nr.toolChoice : { type: "auto" };
  }
  if (nr.stops.length > 0) body.stop_sequences = nr.stops;
  if (nr.stream) body.stream = true;
  return body;
}

function buildGeminiUpstream(model: string, nr: NormReq): any {
  const parts: any[] = [{ text: nr.userText }];
  for (const img of nr.images) {
    if (img.url !== null && img.data === "") parts.push({ fileData: { fileUri: img.url, mimeType: img.mediaType } });
    else parts.push({ inlineData: { mimeType: img.mediaType, data: img.data } });
  }
  const body: any = { contents: [{ role: "user", parts }] };
  if (nr.systemText !== "") body.systemInstruction = { parts: [{ text: nr.systemText }] };
  if (nr.tools.length > 0) {
    body.tools = [
      {
        functionDeclarations: nr.tools.map((t) => ({ name: t.name, description: t.description, parameters: t.parameters })),
      },
    ];
  }
  const config: any = {};
  if (nr.maxTokens !== null) config.maxOutputTokens = nr.maxTokens;
  if (nr.stops.length > 0) config.stopSequences = nr.stops;
  if (nr.jsonMode) config.responseMimeType = "application/json";
  if (nr.temperature !== null) config.temperature = nr.temperature;
  if (Object.keys(config).length > 0) body.generationConfig = config;
  return body;
}

// ---------------------------------------------------------------------------
// Normalized answer: text, tool calls, usage, stop reason, from any wire,
// JSON or SSE. Stop reasons collapse to families: stop/end_turn/STOP is 0,
// length/max_tokens/MAX_TOKENS is 1, tool_calls/tool_use is 2, refusal is 3.
// ---------------------------------------------------------------------------

interface ToolCall {
  id: string;
  name: string;
  arguments: string;
}

interface NormAns {
  text: string;
  toolCalls: ToolCall[];
  usage: { input: number; output: number } | null;
  stop: string;
}

function parseSse(text: string): { event: string | null; data: string }[] {
  const frames: { event: string | null; data: string }[] = [];
  for (const part of text.replace(/\r\n/g, "\n").split("\n\n")) {
    if (part.trim() === "") continue;
    let event: string | null = null;
    const data: string[] = [];
    for (const line of part.split("\n")) {
      if (line.startsWith("event:")) event = line.slice("event:".length).trim();
      else if (line.startsWith("data:")) data.push(line.slice("data:".length).replace(/^ /, ""));
    }
    if (data.length > 0) frames.push({ event, data: data.join("\n") });
  }
  return frames;
}

function safeJsonParse(text: string): any {
  try {
    return JSON.parse(text);
  } catch {
    return {};
  }
}

function parseOpenAIProvider(text: string, stream: boolean): NormAns {
  if (!stream) {
    const doc = JSON.parse(text);
    const choice = Array.isArray(doc.choices) ? doc.choices[0] : null;
    const msg = isRecord(choice) && isRecord(choice.message) ? choice.message : {};
    const content = typeof msg.content === "string" ? msg.content : blocksText(msg.content);
    const toolCalls: ToolCall[] = [];
    if (Array.isArray(msg.tool_calls)) {
      for (const tc of msg.tool_calls) {
        if (isRecord(tc) && isRecord(tc.function)) {
          toolCalls.push({
            id: typeof tc.id === "string" ? tc.id : "",
            name: typeof tc.function.name === "string" ? tc.function.name : "",
            arguments: typeof tc.function.arguments === "string" ? tc.function.arguments : "",
          });
        }
      }
    }
    const usage = isRecord(doc.usage)
      ? { input: asNumber(doc.usage.prompt_tokens) ?? 0, output: asNumber(doc.usage.completion_tokens) ?? 0 }
      : null;
    return {
      text: content,
      toolCalls: toolCalls.filter((t) => t.name !== ""),
      usage,
      stop: isRecord(choice) && typeof choice.finish_reason === "string" ? choice.finish_reason : "stop",
    };
  }
  let outText = "";
  const toolCalls: ToolCall[] = [];
  let usage: { input: number; output: number } | null = null;
  let stop = "stop";
  for (const f of parseSse(text)) {
    if (f.data === "[DONE]") continue;
    let chunk: any;
    try {
      chunk = JSON.parse(f.data);
    } catch {
      continue;
    }
    if (!isRecord(chunk)) continue;
    if (isRecord(chunk.usage)) {
      usage = { input: asNumber(chunk.usage.prompt_tokens) ?? 0, output: asNumber(chunk.usage.completion_tokens) ?? 0 };
    }
    const choice = Array.isArray(chunk.choices) ? chunk.choices[0] : null;
    if (!isRecord(choice)) continue;
    if (isRecord(choice.delta)) {
      if (typeof choice.delta.content === "string") outText += choice.delta.content;
      if (Array.isArray(choice.delta.tool_calls)) {
        for (const tc of choice.delta.tool_calls) {
          if (!isRecord(tc)) continue;
          const idx = typeof tc.index === "number" ? tc.index : 0;
          while (toolCalls.length <= idx) toolCalls.push({ id: "", name: "", arguments: "" });
          if (typeof tc.id === "string") toolCalls[idx].id = tc.id;
          if (isRecord(tc.function)) {
            if (typeof tc.function.name === "string") toolCalls[idx].name = tc.function.name;
            if (typeof tc.function.arguments === "string") toolCalls[idx].arguments += tc.function.arguments;
          }
        }
      }
    }
    if (typeof choice.finish_reason === "string") stop = choice.finish_reason;
  }
  return { text: outText, toolCalls: toolCalls.filter((t) => t.name !== ""), usage, stop };
}

function parseAnthropicProvider(text: string, stream: boolean): NormAns {
  if (!stream) {
    const doc = JSON.parse(text);
    let outText = "";
    const toolCalls: ToolCall[] = [];
    if (Array.isArray(doc.content)) {
      for (const b of doc.content) {
        if (!isRecord(b)) continue;
        if (b.type === "text" && typeof b.text === "string") outText += b.text;
        else if (b.type === "tool_use") {
          toolCalls.push({
            id: typeof b.id === "string" ? b.id : "",
            name: typeof b.name === "string" ? b.name : "",
            arguments: JSON.stringify(b.input ?? {}),
          });
        }
      }
    }
    const usage = isRecord(doc.usage)
      ? { input: asNumber(doc.usage.input_tokens) ?? 0, output: asNumber(doc.usage.output_tokens) ?? 0 }
      : null;
    return {
      text: outText,
      toolCalls: toolCalls.filter((t) => t.name !== ""),
      usage,
      stop: typeof doc.stop_reason === "string" ? doc.stop_reason : "end_turn",
    };
  }
  let outText = "";
  const toolCalls: ToolCall[] = [];
  let input: number | null = null;
  let output: number | null = null;
  let stop = "end_turn";
  let cur: ToolCall | null = null;
  for (const f of parseSse(text)) {
    let data: any;
    try {
      data = JSON.parse(f.data);
    } catch {
      continue;
    }
    if (!isRecord(data)) continue;
    if (f.event === "message_start" && isRecord(data.message) && isRecord(data.message.usage)) {
      input = asNumber(data.message.usage.input_tokens) ?? input;
    } else if (f.event === "content_block_start" && isRecord(data.content_block)) {
      if (data.content_block.type === "tool_use") {
        cur = {
          id: typeof data.content_block.id === "string" ? data.content_block.id : "",
          name: typeof data.content_block.name === "string" ? data.content_block.name : "",
          arguments: "",
        };
        toolCalls.push(cur);
      } else {
        cur = null;
      }
    } else if (f.event === "content_block_delta" && isRecord(data.delta)) {
      if (data.delta.type === "text_delta" && typeof data.delta.text === "string") outText += data.delta.text;
      else if (data.delta.type === "input_json_delta" && typeof data.delta.partial_json === "string" && cur !== null) {
        cur.arguments += data.delta.partial_json;
      }
    } else if (f.event === "message_delta") {
      if (isRecord(data.delta) && typeof data.delta.stop_reason === "string") stop = data.delta.stop_reason;
      if (isRecord(data.usage)) output = asNumber(data.usage.output_tokens) ?? output;
    }
  }
  const usage = input === null && output === null ? null : { input: input ?? 0, output: output ?? 0 };
  return { text: outText, toolCalls: toolCalls.filter((t) => t.name !== ""), usage, stop };
}

function parseGeminiProvider(text: string, stream: boolean): NormAns {
  if (!stream) {
    const doc = JSON.parse(text);
    const cand = Array.isArray(doc.candidates) ? doc.candidates[0] : null;
    let outText = "";
    const toolCalls: ToolCall[] = [];
    const parts = isRecord(cand) && isRecord(cand.content) && Array.isArray(cand.content.parts) ? cand.content.parts : [];
    for (const p of parts) {
      if (!isRecord(p)) continue;
      if (typeof p.text === "string") outText += p.text;
      else if (isRecord(p.functionCall)) {
        toolCalls.push({
          id: "",
          name: typeof p.functionCall.name === "string" ? p.functionCall.name : "",
          arguments: JSON.stringify(p.functionCall.args ?? {}),
        });
      }
    }
    const usage = isRecord(doc.usageMetadata)
      ? { input: asNumber(doc.usageMetadata.promptTokenCount) ?? 0, output: asNumber(doc.usageMetadata.candidatesTokenCount) ?? 0 }
      : null;
    return {
      text: outText,
      toolCalls: toolCalls.filter((t) => t.name !== ""),
      usage,
      stop: isRecord(cand) && typeof cand.finishReason === "string" ? cand.finishReason : "STOP",
    };
  }
  let outText = "";
  const toolCalls: ToolCall[] = [];
  let usage: { input: number; output: number } | null = null;
  let stop = "STOP";
  for (const f of parseSse(text)) {
    let data: any;
    try {
      data = JSON.parse(f.data);
    } catch {
      continue;
    }
    if (!isRecord(data)) continue;
    const cand = Array.isArray(data.candidates) ? data.candidates[0] : null;
    if (isRecord(cand)) {
      const parts = isRecord(cand.content) && Array.isArray(cand.content.parts) ? cand.content.parts : [];
      for (const p of parts) {
        if (!isRecord(p)) continue;
        if (typeof p.text === "string") outText += p.text;
        else if (isRecord(p.functionCall)) {
          toolCalls.push({
            id: "",
            name: typeof p.functionCall.name === "string" ? p.functionCall.name : "",
            arguments: JSON.stringify(p.functionCall.args ?? {}),
          });
        }
      }
      if (typeof cand.finishReason === "string") stop = cand.finishReason;
    }
    if (isRecord(data.usageMetadata)) {
      usage = { input: asNumber(data.usageMetadata.promptTokenCount) ?? 0, output: asNumber(data.usageMetadata.candidatesTokenCount) ?? 0 };
    }
  }
  return { text: outText, toolCalls: toolCalls.filter((t) => t.name !== ""), usage, stop };
}

// ---------------------------------------------------------------------------
// Rendering: the normalized answer back into the client's wire, JSON or SSE,
// with the provider's usage counts as that wire reports them and the stop
// reason mapped across wires by family.
// ---------------------------------------------------------------------------

const STOP_FAMILY: Record<string, number> = {
  stop: 0,
  end_turn: 0,
  STOP: 0,
  length: 1,
  max_tokens: 1,
  MAX_TOKENS: 1,
  tool_calls: 2,
  function_call: 2,
  tool_use: 2,
  content_filter: 3,
  refusal: 3,
  SAFETY: 3,
};

function stopFamilyOf(reason: string): number {
  return STOP_FAMILY[reason] ?? 0;
}

function openAIStop(ans: NormAns): string {
  if (ans.toolCalls.length > 0) return "tool_calls";
  const fam = stopFamilyOf(ans.stop);
  return fam === 1 ? "length" : fam === 3 ? "content_filter" : "stop";
}

function anthropicStop(ans: NormAns): string {
  if (ans.toolCalls.length > 0) return "tool_use";
  const fam = stopFamilyOf(ans.stop);
  return fam === 1 ? "max_tokens" : fam === 3 ? "refusal" : "end_turn";
}

function sseFrame(event: string | null, data: any): string {
  return `${event === null ? "" : `event: ${event}\n`}data: ${JSON.stringify(data)}\n\n`;
}

function splitText(text: string): string[] {
  const pieces: string[] = [];
  for (let at = 0; at < text.length; at += 13) pieces.push(text.slice(at, at + 13));
  return pieces.length === 0 ? [""] : pieces;
}

function renderOpenAIJson(model: string, ans: NormAns): any {
  const message: any = { role: "assistant", content: ans.text };
  if (ans.toolCalls.length > 0) {
    message.tool_calls = ans.toolCalls.map((t) => ({
      id: t.id === "" ? "call_agg" : t.id,
      type: "function",
      function: { name: t.name, arguments: t.arguments },
    }));
  }
  return {
    id: "chatcmpl-agg",
    object: "chat.completion",
    created: FIXED_CREATED,
    model,
    choices: [{ index: 0, message, finish_reason: openAIStop(ans), logprobs: null }],
    usage: {
      prompt_tokens: ans.usage?.input ?? 0,
      completion_tokens: ans.usage?.output ?? 0,
      total_tokens: (ans.usage?.input ?? 0) + (ans.usage?.output ?? 0),
    },
  };
}

function renderOpenAISse(model: string, ans: NormAns): string {
  const chunk = (delta: any, finish: string | null) => ({
    id: "chatcmpl-agg",
    object: "chat.completion.chunk",
    created: FIXED_CREATED,
    model,
    choices: [{ index: 0, delta, finish_reason: finish }],
  });
  const frames: string[] = [];
  frames.push(sseFrame(null, chunk({ role: "assistant", content: "" }, null)));
  for (const piece of splitText(ans.text)) frames.push(sseFrame(null, chunk({ content: piece }, null)));
  if (ans.toolCalls.length > 0) {
    frames.push(
      sseFrame(
        null,
        chunk(
          {
            tool_calls: ans.toolCalls.map((t, i) => ({
              index: i,
              id: t.id === "" ? "call_agg" : t.id,
              type: "function",
              function: { name: t.name, arguments: t.arguments },
            })),
          },
          null
        )
      )
    );
  }
  frames.push(sseFrame(null, chunk({}, openAIStop(ans))));
  if (ans.usage !== null) {
    frames.push(
      sseFrame(null, {
        id: "chatcmpl-agg",
        object: "chat.completion.chunk",
        created: FIXED_CREATED,
        model,
        choices: [],
        usage: {
          prompt_tokens: ans.usage.input,
          completion_tokens: ans.usage.output,
          total_tokens: ans.usage.input + ans.usage.output,
        },
      })
    );
  }
  frames.push("data: [DONE]\n\n");
  return frames.join("");
}

function renderAnthropicJson(model: string, ans: NormAns): any {
  const content: any[] = [{ type: "text", text: ans.text }];
  for (const t of ans.toolCalls) {
    content.push({ type: "tool_use", id: t.id === "" ? "toolu_agg" : t.id, name: t.name, input: safeJsonParse(t.arguments) });
  }
  return {
    id: "msg_agg",
    type: "message",
    role: "assistant",
    model,
    content,
    stop_reason: anthropicStop(ans),
    stop_sequence: null,
    usage: { input_tokens: ans.usage?.input ?? 0, output_tokens: ans.usage?.output ?? 0 },
  };
}

function renderAnthropicSse(model: string, ans: NormAns): string {
  const frames: string[] = [];
  frames.push(
    sseFrame("message_start", {
      type: "message_start",
      message: {
        id: "msg_agg",
        type: "message",
        role: "assistant",
        model,
        content: [],
        stop_reason: null,
        stop_sequence: null,
        usage: { input_tokens: ans.usage?.input ?? 0, output_tokens: 0 },
      },
    })
  );
  frames.push(sseFrame("content_block_start", { type: "content_block_start", index: 0, content_block: { type: "text", text: "" } }));
  for (const piece of splitText(ans.text)) {
    frames.push(sseFrame("content_block_delta", { type: "content_block_delta", index: 0, delta: { type: "text_delta", text: piece } }));
  }
  frames.push(sseFrame("content_block_stop", { type: "content_block_stop", index: 0 }));
  let idx = 1;
  for (const t of ans.toolCalls) {
    frames.push(
      sseFrame("content_block_start", {
        type: "content_block_start",
        index: idx,
        content_block: { type: "tool_use", id: t.id === "" ? "toolu_agg" : t.id, name: t.name, input: {} },
      })
    );
    frames.push(
      sseFrame("content_block_delta", {
        type: "content_block_delta",
        index: idx,
        delta: { type: "input_json_delta", partial_json: t.arguments },
      })
    );
    frames.push(sseFrame("content_block_stop", { type: "content_block_stop", index: idx }));
    idx++;
  }
  frames.push(
    sseFrame("message_delta", {
      type: "message_delta",
      delta: { stop_reason: anthropicStop(ans), stop_sequence: null },
      usage: { output_tokens: ans.usage?.output ?? 0 },
    })
  );
  frames.push(sseFrame("message_stop", { type: "message_stop" }));
  return frames.join("");
}

// ---------------------------------------------------------------------------
// Routing and failover. Candidates are the brief's free models that meet
// the request's requirements, ranked by rating, ties spread across the
// least-used providers, then by the brief's provider order. A JSON-mode
// OpenAI request never goes to an Anthropic-wire provider. A 402, 429 or
// 5xx (or an unreachable provider) strikes the provider; two strikes in a
// row mark it spent for the rest of the trial, retried only when nothing
// else can serve. A 400/401 spends nothing and is skipped without a strike.
// ---------------------------------------------------------------------------

interface Runtime {
  envUrl: string;
  brief: Brief;
  catalog: Catalog;
  providerById: Map<string, ProviderDir>;
  providerOrder: Map<string, number>;
  usageCount: Map<string, number>;
  consecFail: Map<string, number>;
  dead: Set<string>;
  requestIndex: number;
}

function rankCandidates(rt: Runtime, req: Requires, jsonMode: boolean): RatingRow[] {
  const out: RatingRow[] = [];
  for (const row of rt.brief.ratings) {
    if (!row.free) continue;
    if (req.tools === true && !row.tools) continue;
    if (req.vision === true && !row.vision) continue;
    if (req.json === true && !row.json) continue;
    if (req.min_context !== undefined && row.context_length < req.min_context) continue;
    if (jsonMode) {
      const prov = rt.providerById.get(row.provider);
      if (prov !== undefined && prov.wire === "anthropic") continue;
    }
    out.push(row);
  }
  out.sort(
    (a, b) =>
      b.rating_milli - a.rating_milli ||
      (rt.usageCount.get(a.provider) ?? 0) - (rt.usageCount.get(b.provider) ?? 0) ||
      (rt.providerOrder.get(a.provider) ?? 0) - (rt.providerOrder.get(b.provider) ?? 0) ||
      rt.brief.ratings.indexOf(a) - rt.brief.ratings.indexOf(b)
  );
  return out;
}

async function callProvider(rt: Runtime, prov: ProviderDir, model: string, nr: NormReq): Promise<{ ok: boolean; status: number; ans: NormAns | null }> {
  const base = rt.envUrl + prov.path;
  const headers: Record<string, string> = { "content-type": "application/json", ...authHeaders(prov) };
  let url: string;
  let payload: any;
  if (prov.wire === "openai") {
    url = `${base}/v1/chat/completions`;
    payload = buildOpenAIUpstream(model, nr);
  } else if (prov.wire === "anthropic") {
    url = `${base}/v1/messages`;
    headers["anthropic-version"] = "2023-06-01";
    payload = buildAnthropicUpstream(model, nr);
  } else {
    const method = nr.stream ? "streamGenerateContent" : "generateContent";
    url = `${base}/v1beta/models/${encodeURIComponent(model)}:${method}${nr.stream ? "?alt=sse" : ""}`;
    payload = buildGeminiUpstream(model, nr);
  }
  let res: Response;
  try {
    res = await fetchWithTimeout(url, { method: "POST", headers, body: JSON.stringify(payload) }, PROVIDER_TIMEOUT_MS);
  } catch {
    return { ok: false, status: 599, ans: null };
  }
  let text = "";
  try {
    text = await res.text();
  } catch {
    return { ok: false, status: 599, ans: null };
  }
  if (res.status < 200 || res.status >= 300) return { ok: false, status: res.status, ans: null };
  try {
    const ans =
      prov.wire === "openai"
        ? parseOpenAIProvider(text, nr.stream)
        : prov.wire === "anthropic"
          ? parseAnthropicProvider(text, nr.stream)
          : parseGeminiProvider(text, nr.stream);
    return { ok: true, status: res.status, ans };
  } catch {
    return { ok: false, status: 502, ans: null };
  }
}

// The request's requirements come from the workload manifest: the env's
// /work names the request being sent. The arrival counter is the fallback
// when /work cannot be read, and the body itself is the last resort.
async function manifestRequires(rt: Runtime, nr: NormReq): Promise<Requires> {
  const idx = rt.requestIndex;
  rt.requestIndex += 1;
  let id: string | null = null;
  try {
    const res = await fetchWithTimeout(`${rt.envUrl}/work`, { method: "GET" }, 3000);
    const doc = await res.json();
    if (isRecord(doc) && typeof doc.next === "string") id = doc.next;
  } catch {
    id = null;
  }
  const entry = id !== null ? rt.brief.manifest.find((m) => m.id === id) : rt.brief.manifest[idx];
  if (entry !== undefined) return entry.requires;
  const req: Requires = {};
  if (nr.tools.length > 0) req.tools = true;
  if (nr.images.length > 0) req.vision = true;
  if (nr.jsonMode) req.json = true;
  return req;
}

interface Served {
  status: number;
  contentType: string;
  body: string;
}

async function serveRequest(rt: Runtime, api: "openai" | "anthropic", body: any): Promise<Served> {
  const nr = api === "openai" ? parseOpenAIClient(body) : parseAnthropicClient(body);
  const req = await manifestRequires(rt, nr);
  const cands = rankCandidates(rt, req, nr.jsonMode);
  const attempts: RatingRow[] = [
    ...cands.filter((c) => !rt.dead.has(c.provider)),
    ...cands.filter((c) => rt.dead.has(c.provider)),
  ];
  const tried = new Set<string>();
  for (const cand of attempts) {
    const key = `${cand.provider}/${cand.model}`;
    if (tried.has(key)) continue;
    tried.add(key);
    const prov = rt.providerById.get(cand.provider);
    if (prov === undefined) continue;
    const res = await callProvider(rt, prov, cand.model, nr);
    if (res.ok && res.ans !== null) {
      rt.usageCount.set(cand.provider, (rt.usageCount.get(cand.provider) ?? 0) + 1);
      rt.consecFail.set(cand.provider, 0);
      const ans = res.ans;
      if (api === "openai") {
        if (nr.stream) return { status: 200, contentType: "text/event-stream; charset=utf-8", body: renderOpenAISse(cand.model, ans) };
        return { status: 200, contentType: "application/json", body: JSON.stringify(renderOpenAIJson(cand.model, ans)) };
      }
      if (nr.stream) return { status: 200, contentType: "text/event-stream; charset=utf-8", body: renderAnthropicSse(cand.model, ans) };
      return { status: 200, contentType: "application/json", body: JSON.stringify(renderAnthropicJson(cand.model, ans)) };
    }
    if (res.status === 400 || res.status === 401) continue;
    const fails = (rt.consecFail.get(cand.provider) ?? 0) + 1;
    rt.consecFail.set(cand.provider, fails);
    if (fails >= 2) rt.dead.add(cand.provider);
  }
  return {
    status: 502,
    contentType: "application/json",
    body: JSON.stringify({ error: { message: "no free provider could serve this request", type: "api_error", code: "no_provider" } }),
  };
}

// ---------------------------------------------------------------------------
// HTTP server: GET /v1/models in the wire its auth header names, the two
// inference routes, and the trial loop: listen, register, poll /work until
// the env says done, exit 0.
// ---------------------------------------------------------------------------

function freeModelIds(rt: Runtime): string[] {
  const ids: string[] = [];
  const seen = new Set<string>();
  for (const p of rt.catalog.providers) {
    for (const m of p.models) {
      if (m.free && !seen.has(m.id)) {
        seen.add(m.id);
        ids.push(m.id);
      }
    }
  }
  return ids;
}

function handleModels(rt: Runtime, req: any): { status: number; body: string } {
  const ids = freeModelIds(rt);
  const auth = req.headers["authorization"];
  const apiKey = req.headers["x-api-key"];
  if (typeof auth === "string" && auth.startsWith("Bearer ")) {
    return {
      status: 200,
      body: JSON.stringify({
        object: "list",
        data: ids.map((id) => ({ id, object: "model", created: FIXED_CREATED, owned_by: "free-model-aggregator", free: true })),
      }),
    };
  }
  if (typeof apiKey === "string" && apiKey !== "") {
    return {
      status: 200,
      body: JSON.stringify({
        data: ids.map((id) => ({ type: "model", id, display_name: id, created_at: FIXED_DATE, free: true })),
        has_more: false,
      }),
    };
  }
  return {
    status: 200,
    body: JSON.stringify({
      object: "list",
      data: ids.map((id) => ({ id, object: "model", created: FIXED_CREATED, owned_by: "free-model-aggregator", free: true })),
    }),
  };
}

function readBody(req: any): Promise<string> {
  return new Promise((resolve, reject) => {
    const chunks: Buffer[] = [];
    let size = 0;
    req.on("data", (c: Buffer) => {
      size += c.length;
      if (size <= MAX_BODY_BYTES) chunks.push(c);
    });
    req.on("end", () => resolve(Buffer.concat(chunks).toString("utf8")));
    req.on("error", reject);
  });
}

async function handleRequest(rt: Runtime, req: any, res: any): Promise<void> {
  try {
    const url = new URL(req.url ?? "/", "http://entry");
    if (req.method === "GET" && url.pathname === "/v1/models") {
      const out = handleModels(rt, req);
      res.writeHead(out.status, { "content-type": "application/json" });
      res.end(out.body);
      return;
    }
    if (req.method === "POST" && (url.pathname === "/v1/chat/completions" || url.pathname === "/v1/messages")) {
      const raw = await readBody(req);
      let body: any = null;
      try {
        body = JSON.parse(raw);
      } catch {
        body = null;
      }
      if (!isRecord(body)) {
        res.writeHead(400, { "content-type": "application/json" });
        res.end(
          JSON.stringify({
            error: { message: "the request body is not one JSON object", type: "invalid_request_error", code: "invalid_request" },
          })
        );
        return;
      }
      const api = url.pathname === "/v1/chat/completions" ? "openai" : "anthropic";
      const out = await serveRequest(rt, api, body);
      res.writeHead(out.status, { "content-type": out.contentType });
      res.end(out.body);
      return;
    }
    res.writeHead(404, { "content-type": "application/json" });
    res.end(
      JSON.stringify({
        error: { message: `no route ${req.method} ${url.pathname}`, type: "invalid_request_error", code: "not_found" },
      })
    );
  } catch (err) {
    try {
      res.writeHead(500, { "content-type": "application/json" });
      res.end(
        JSON.stringify({
          error: { message: err instanceof Error ? err.message : String(err), type: "api_error", code: "internal" },
        })
      );
    } catch {
      // The socket is gone; nothing left to answer.
    }
  }
}

async function main(): Promise<void> {
  const args = process.argv.slice(2);
  const listMode = args.includes("--list");
  const promptFile = process.env.MAKEFUN_PROMPT_FILE ?? "";
  const envUrl = (process.env.MAKEFUN_ENV_URL ?? "").replace(/\/+$/, "");

  if (listMode) {
    if (promptFile === "") {
      process.stdout.write('{"providers":[]}\n');
      return;
    }
    const brief = parseBrief(readFileSync(promptFile, "utf8"));
    const catalog = await buildCatalog(envUrl, brief);
    process.stdout.write(`${JSON.stringify(catalog)}\n`);
    return;
  }

  if (promptFile === "" || envUrl === "") {
    process.stderr.write("free-model-aggregator-01: needs MAKEFUN_PROMPT_FILE and MAKEFUN_ENV_URL\n");
    process.exitCode = 1;
    return;
  }

  const brief = parseBrief(readFileSync(promptFile, "utf8"));
  const catalog = await buildCatalog(envUrl, brief);
  const rt: Runtime = {
    envUrl,
    brief,
    catalog,
    providerById: new Map(brief.providers.map((p) => [p.id, p])),
    providerOrder: new Map(brief.providers.map((p, i) => [p.id, i])),
    usageCount: new Map(),
    consecFail: new Map(),
    dead: new Set(),
    requestIndex: 0,
  };

  const server = createServer((req, res) => {
    void handleRequest(rt, req, res);
  });
  await new Promise<void>((resolve) => server.listen(0, "0.0.0.0", () => resolve()));
  const addr: any = server.address();
  const port = typeof addr === "object" && addr !== null ? addr.port : 0;

  let registered = false;
  for (let attempt = 0; attempt < 30 && !registered; attempt++) {
    try {
      const res = await fetchWithTimeout(
        `${envUrl}/register`,
        {
          method: "POST",
          headers: { "content-type": "application/json" },
          body: JSON.stringify({ protocol_version: 1, port, catalog }),
        },
        5000
      );
      const doc = await res.json().catch(() => null);
      if (res.status === 200 && isRecord(doc) && doc.ok === true) registered = true;
      else if (res.status === 400) break;
    } catch {
      // The env is not up yet; retry.
    }
    if (!registered) await sleep(200);
  }
  if (!registered) {
    process.stderr.write("free-model-aggregator-01: register failed\n");
    server.close();
    process.exitCode = 1;
    return;
  }

  for (;;) {
    try {
      const res = await fetchWithTimeout(`${envUrl}/work`, { method: "GET" }, 5000);
      const doc = await res.json().catch(() => null);
      if (isRecord(doc) && doc.done === true) break;
    } catch {
      // Keep polling.
    }
    await sleep(WORK_POLL_MS);
  }
  server.close();
  process.exit(0);
}

main().catch((err) => {
  process.stderr.write(`free-model-aggregator-01: ${err instanceof Error ? err.message : String(err)}\n`);
  process.exitCode = 1;
});
