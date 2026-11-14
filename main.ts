// free-model-aggregator-01 v2: a quality-ranked free-model aggregator.
//
// One server, two client wires (OpenAI Chat Completions, Anthropic Messages),
// ten simulated free-tier providers on three upstream wires (OpenAI,
// Anthropic, Gemini). For each of the twenty requests it picks the best free
// model that meets the request's requirements, from a budget-aware plan
// computed up front over the known workload, fails over across providers
// when one refuses, and translates the answer back into the wire the client
// used, JSON or streaming.
//
//   node main.ts --list     print the catalog and exit 0
//   node main.ts            register with the env, serve, exit 0 when done
//
// Determinism: every routing choice is a function of the brief, the
// requests, and the providers' answers. No wall clock and no unseeded
// randomness anywhere in a request path, so a replay leaves the same state
// export byte for byte.

import { createServer, type IncomingMessage, type Server, type ServerResponse } from 'node:http';
import {
  authHeaders,
  isRecord,
  joinBase,
  readBrief,
  type Brief,
  type Json,
  type Requirement,
} from './lib/brief.ts';
import {
  buildCatalog,
  probeAll,
  servedModels,
  type Catalog,
  type Probe,
  type ServedModel,
} from './lib/catalog.ts';
import {
  attemptList,
  buildPlan,
  newRouterState,
  requirementsFromBody,
  type Candidate,
  type RouterState,
} from './lib/plan.ts';
import {
  ANTHROPIC_VERSION,
  anthropicError,
  anthropicModelList,
  anthropicStreamToOpenAI,
  anthropicToGeminiReq,
  anthropicToOpenAIReq,
  anthropicToOpenAIRes,
  geminiStreamToAnthropic,
  geminiStreamToOpenAI,
  geminiToAnthropicRes,
  geminiToOpenAIRes,
  openAIError,
  openAIModelList,
  openAIStreamToAnthropic,
  openAIToAnthropicReq,
  openAIToAnthropicRes,
  openAIToGeminiReq,
  parseSse,
} from './lib/translate.ts';

const PROTOCOL_VERSION = 1;
const POLL_INTERVAL_MS = 100;
const PROBE_TIMEOUT_MS = 5000;
const UPSTREAM_TIMEOUT_MS = 4000;
const MAX_BODY_BYTES = 32 * 1024 * 1024;

const ENV_URL = (process.env.MAKEFUN_ENV_URL ?? '').replace(/\/+$/, '');
const PROMPT_FILE = process.env.MAKEFUN_PROMPT_FILE ?? '';

const brief: Brief = readBrief(PROMPT_FILE);
let probes: Probe[] = [];
let catalog: Catalog = { providers: [] };
let models: ServedModel[] = [];
let router: RouterState = newRouterState([], brief);

/** Requests seen so far, in arrival order; also keys the manifest lookup. */
let arrivals = 0;
const usedRows = new Set<string>();

function log(message: string): void {
  process.stderr.write(`aggregator: ${message}\n`);
}

async function load(): Promise<void> {
  if (brief.error !== null) log(brief.error);
  probes = await probeAll(ENV_URL, brief.providers, PROBE_TIMEOUT_MS);
  for (const probe of probes) {
    if (!probe.ok && probe.error !== null) log(probe.error);
  }
  catalog = buildCatalog(probes, brief);
  models = servedModels(probes, brief);
  router = newRouterState(probes, brief);
  // A JSON-mode OpenAI request carries response_format on the wire; an
  // Anthropic one never does. The plan's wire rule follows the manifest.
  buildPlan(router, (row) => row.api === 'openai' && row.requires.json === true);
}

// ---------------------------------------------------------------------------
// Request identification
// ---------------------------------------------------------------------------

interface Identified {
  id: string | null;
  requires: Requirement;
}

/** The manifest row for the request now arriving, else the body's own needs. */
function identify(clientApi: 'openai' | 'anthropic', stream: boolean, body: Json): Identified {
  const index = arrivals;
  arrivals += 1;
  const manifest = brief.workload;
  const rowAtIndex = manifest[index];
  if (
    rowAtIndex !== undefined &&
    rowAtIndex.api === clientApi &&
    rowAtIndex.stream === stream &&
    !usedRows.has(rowAtIndex.id)
  ) {
    usedRows.add(rowAtIndex.id);
    return { id: rowAtIndex.id, requires: rowAtIndex.requires };
  }
  // Out-of-order arrival: match an unused row by api, stream, and the
  // requirements the body itself shows (tools, vision, JSON mode). A
  // min_context requirement leaves no mark on the body, so rows that differ
  // only by it fall through to earliest-unused order, which is manifest
  // order — the order the client sends them in.
  const bodyReq = requirementsFromBody(body);
  const signature = (req: Requirement): string =>
    [req.tools === true ? 'tools' : '-', req.vision === true ? 'vision' : '-', req.json === true ? 'json' : '-'].join('|');
  const want = signature(bodyReq);
  const byApiStream = manifest.filter(
    (row) => row.api === clientApi && row.stream === stream && !usedRows.has(row.id),
  );
  const exact = byApiStream.find((row) => signature(row.requires) === want);
  const row = exact ?? byApiStream[0];
  if (row !== undefined) {
    usedRows.add(row.id);
    return { id: row.id, requires: row.requires };
  }
  return { id: null, requires: bodyReq };
}

function jsonWireNeeded(clientApi: 'openai' | 'anthropic', body: Json): boolean {
  if (clientApi !== 'openai') return false;
  const format = isRecord(body.response_format) ? body.response_format.type : undefined;
  return format === 'json_object' || format === 'json_schema';
}

// ---------------------------------------------------------------------------
// Upstream calls
// ---------------------------------------------------------------------------

function upstreamBody(
  clientApi: 'openai' | 'anthropic',
  candidate: Candidate,
  body: Json,
): Json {
  if (clientApi === 'openai') {
    if (candidate.wire === 'openai') {
      const out: Json = { ...body, model: candidate.model };
      if (body.stream === true) out.stream_options = { include_usage: true };
      return out;
    }
    if (candidate.wire === 'anthropic') return openAIToAnthropicReq(body, candidate.model);
    return openAIToGeminiReq(body, candidate.model);
  }
  if (candidate.wire === 'openai') return anthropicToOpenAIReq(body, candidate.model);
  if (candidate.wire === 'anthropic') return { ...body, model: candidate.model };
  return anthropicToGeminiReq(body, candidate.model);
}

function upstreamUrl(candidate: Candidate, stream: boolean): string {
  const probe = probes.find((p) => p.provider.id === candidate.provider);
  const base = probe === undefined ? '' : joinBase(ENV_URL, probe.provider.base);
  if (candidate.wire === 'openai') return `${base}/v1/chat/completions`;
  if (candidate.wire === 'anthropic') return `${base}/v1/messages`;
  const verb = stream ? ':streamGenerateContent?alt=sse' : ':generateContent';
  return `${base}/v1beta/models/${encodeURIComponent(candidate.model)}${verb}`;
}

interface UpstreamOk {
  ok: true;
  status: number;
  text: string;
  candidate: Candidate;
}

interface UpstreamFail {
  ok: false;
  status: number;
  detail: string;
}

async function callUpstream(
  candidate: Candidate,
  clientApi: 'openai' | 'anthropic',
  body: Json,
  stream: boolean,
): Promise<UpstreamOk | UpstreamFail> {
  const probe = probes.find((p) => p.provider.id === candidate.provider);
  if (probe === undefined) return { ok: false, status: 0, detail: 'no probe for provider' };
  const headers: Record<string, string> = {
    'content-type': 'application/json',
    ...authHeaders(probe.provider.auth, probe.provider.key),
  };
  if (candidate.wire === 'anthropic') headers['anthropic-version'] = ANTHROPIC_VERSION;
  if (stream) headers.accept = 'text/event-stream';
  let res: Response;
  try {
    res = await fetch(upstreamUrl(candidate, stream), {
      method: 'POST',
      headers,
      body: JSON.stringify(upstreamBody(clientApi, candidate, body)),
      signal: AbortSignal.timeout(UPSTREAM_TIMEOUT_MS),
    });
  } catch (err) {
    return { ok: false, status: 0, detail: (err as Error).message };
  }
  router.calls.set(candidate.provider, (router.calls.get(candidate.provider) ?? 0) + 1);
  if (!res.ok) {
    const detail = await res.text().catch(() => '');
    return { ok: false, status: res.status, detail: detail.slice(0, 200) };
  }
  return { ok: true, status: res.status, text: await res.text(), candidate };
}

// ---------------------------------------------------------------------------
// Response translation
// ---------------------------------------------------------------------------

function translateJson(
  clientApi: 'openai' | 'anthropic',
  candidate: Candidate,
  text: string,
): Json {
  let answer: unknown = null;
  try {
    answer = JSON.parse(text);
  } catch {
    answer = null;
  }
  if (!isRecord(answer)) return { text: '', usage: { input: 0, output: 0 } };
  if (clientApi === 'openai') {
    if (candidate.wire === 'openai') return answer;
    if (candidate.wire === 'anthropic') return anthropicToOpenAIRes(answer, candidate.model);
    return geminiToOpenAIRes(answer, candidate.model);
  }
  if (candidate.wire === 'anthropic') return answer;
  if (candidate.wire === 'openai') return openAIToAnthropicRes(answer, candidate.model);
  return geminiToAnthropicRes(answer, candidate.model);
}

function translateStream(
  clientApi: 'openai' | 'anthropic',
  candidate: Candidate,
  text: string,
): string {
  if (clientApi === 'openai' && candidate.wire === 'openai') return text;
  if (clientApi === 'anthropic' && candidate.wire === 'anthropic') return text;
  const frames = parseSse(text);
  if (clientApi === 'openai') {
    if (candidate.wire === 'anthropic') return anthropicStreamToOpenAI(frames, candidate.model);
    return geminiStreamToOpenAI(frames, candidate.model);
  }
  if (candidate.wire === 'openai') return openAIStreamToAnthropic(frames, candidate.model);
  return geminiStreamToAnthropic(frames, candidate.model);
}

// ---------------------------------------------------------------------------
// HTTP plumbing
// ---------------------------------------------------------------------------

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

function sendJson(res: ServerResponse, status: number, body: unknown, route?: Candidate): void {
  const text = JSON.stringify(body);
  const headers: Record<string, string> = {
    'content-type': 'application/json',
    'content-length': String(Buffer.byteLength(text)),
  };
  if (route !== undefined) {
    headers['x-aggregator-provider'] = route.provider;
    headers['x-aggregator-model'] = route.model;
  }
  res.writeHead(status, headers);
  res.end(text);
}

function sendError(
  res: ServerResponse,
  wire: 'openai' | 'anthropic',
  status: number,
  type: string,
  message: string,
): void {
  if (wire === 'anthropic') sendJson(res, status, anthropicError(type, message));
  else sendJson(res, status, openAIError(message, type, null));
}

function routeHeaders(route: Candidate): Record<string, string> {
  return { 'x-aggregator-provider': route.provider, 'x-aggregator-model': route.model };
}

/** GET /v1/models, in the shape of the auth header the client sent. */
function handleModels(req: IncomingMessage, res: ServerResponse): void {
  if (typeof req.headers['x-api-key'] === 'string') sendJson(res, 200, anthropicModelList(models));
  else sendJson(res, 200, openAIModelList(models));
}

/** GET /health: liveness plus live routing counters. */
function handleHealth(_req: IncomingMessage, res: ServerResponse): void {
  sendJson(res, 200, {
    status: 'ok',
    providers: probes.filter((p) => p.ok).length,
    free_models: models.length,
    planned_requests: router.plan.size,
    calls: Object.fromEntries(router.calls),
    refusals: Object.fromEntries(router.refusals),
  });
}

/** GET /v1/providers: the catalog with live per-provider counters. */
function handleProviders(_req: IncomingMessage, res: ServerResponse): void {
  sendJson(res, 200, {
    providers: catalog.providers.map((p) => ({
      ...p,
      calls: router.calls.get(p.id) ?? 0,
      refusals: router.refusals.get(p.id) ?? 0,
      planned: [...router.plan.values()].filter((c) => c.provider === p.id).length,
    })),
  });
}

/**
 * Serve one client request: try the planned candidate, then the failover
 * list, until a provider answers 2xx. A 2xx answer is translated and
 * returned even if translation finds nothing to carry: retrying after a
 * success would make two providers answer one request.
 */
async function handleCompletion(
  req: IncomingMessage,
  res: ServerResponse,
  clientApi: 'openai' | 'anthropic',
): Promise<void> {
  const body = await readJson(req);
  if (body === null) {
    return sendError(res, clientApi, 400, 'invalid_request_error', 'the request body is not one JSON object');
  }
  const stream = body.stream === true;
  const identified = identify(clientApi, stream, body);
  const attempts = attemptList(router, identified.id, identified.requires, jsonWireNeeded(clientApi, body));
  if (attempts.length === 0) {
    return sendError(
      res,
      clientApi,
      503,
      clientApi === 'anthropic' ? 'overloaded_error' : 'no_free_model',
      'no free model on the directory meets this request',
    );
  }
  for (const candidate of attempts) {
    const result = await callUpstream(candidate, clientApi, body, stream);
    if (!result.ok) {
      router.refusals.set(candidate.provider, (router.refusals.get(candidate.provider) ?? 0) + 1);
      log(`refused: ${candidate.provider}/${candidate.model} HTTP ${result.status} (${result.detail.slice(0, 120)})`);
      continue;
    }
    if (stream) {
      res.writeHead(200, {
        'content-type': 'text/event-stream',
        'cache-control': 'no-cache',
        connection: 'keep-alive',
        ...routeHeaders(candidate),
      });
      res.end(translateStream(clientApi, candidate, result.text));
      return;
    }
    sendJson(res, 200, translateJson(clientApi, candidate, result.text), candidate);
    return;
  }
  return sendError(
    res,
    clientApi,
    503,
    clientApi === 'anthropic' ? 'overloaded_error' : 'no_free_model',
    'every free model that meets this request refused it',
  );
}

async function handle(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const url = new URL(req.url ?? '/', 'http://entry');
  if (req.method === 'GET' && url.pathname === '/v1/models') {
    handleModels(req, res);
    return;
  }
  if (req.method === 'GET' && url.pathname === '/health') {
    handleHealth(req, res);
    return;
  }
  if (req.method === 'GET' && url.pathname === '/v1/providers') {
    handleProviders(req, res);
    return;
  }
  if (req.method === 'POST' && url.pathname === '/v1/chat/completions') {
    await handleCompletion(req, res, 'openai');
    return;
  }
  if (req.method === 'POST' && url.pathname === '/v1/messages') {
    await handleCompletion(req, res, 'anthropic');
    return;
  }
  sendError(res, url.pathname === '/v1/messages' ? 'anthropic' : 'openai', 404, 'not_found', `no route ${req.method ?? ''} ${url.pathname}`);
}

function listen(server: Server): Promise<void> {
  return new Promise((resolve, reject) => {
    server.once('error', reject);
    server.listen(0, '0.0.0.0', () => resolve());
  });
}

async function register(port: number, report: Catalog): Promise<void> {
  const res = await fetch(`${ENV_URL}/register`, {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify({ protocol_version: PROTOCOL_VERSION, port, catalog: report }),
    signal: AbortSignal.timeout(5000),
  });
  if (!res.ok) throw new Error(`POST ${ENV_URL}/register answered HTTP ${res.status}: ${(await res.text()).slice(0, 500)}`);
  const reply: unknown = await res.json();
  if (isRecord(reply) && typeof reply.requests === 'number') {
    log(`registered: the workload holds ${reply.requests} requests`);
  }
}

/** Poll GET /work every 100 ms; exit 0 the moment it says done. */
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
  await load();
  if (process.argv.includes('--list')) {
    const text = `${JSON.stringify(catalog)}\n`;
    await new Promise<void>((resolve) => process.stdout.write(text, () => resolve()));
    process.exit(0);
  }
  if (ENV_URL === '') throw new Error('MAKEFUN_ENV_URL is not set');
  const server = createServer((req, res) => {
    void handle(req, res).catch((err: unknown) => {
      log((err as Error).stack ?? String(err));
      if (!res.headersSent) {
        sendError(res, req.url?.startsWith('/v1/messages') === true ? 'anthropic' : 'openai', 500, 'internal_error', 'the server failed to answer');
      }
    });
  });
  await listen(server);
  const address = server.address();
  if (address === null || typeof address === 'string') throw new Error('the server has no port');
  const port = address.port;
  if (port === 8080) throw new Error('the OS picked 8080, which the env owns on one layout: restart the trial');
  const answered = probes.filter((p) => p.ok).length;
  log(`listening on 0.0.0.0:${port} with ${answered} answered providers, ${models.length} free models, ${router.plan.size} planned requests`);
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
