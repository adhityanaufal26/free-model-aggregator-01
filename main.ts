// free-model-aggregator-01 v3: one server, two client wires, ten free providers.
//
// The entry reads the trial brief, probes every provider's catalog route (a
// probe costs no upstream calls), registers the one catalog, and serves the
// client's 20 requests in manifest order. Each request goes to the best free
// model that meets it, under a hard 2-calls-per-provider plan built from the
// brief alone; a provider that refuses once is dead for the rest of the trial,
// and the plan for the remaining requests is rebuilt after every refusal, with
// failover calls counted against the caps. It exits 0 when the env says the
// workload is done.
//
//   node main.ts          serve a trial (MAKEFUN_ENV_URL, MAKEFUN_PROMPT_FILE)
//   node main.ts --list   print the catalog document and exit 0
//
// Node 22 standard library only; no wall clock and no unseeded randomness, so
// a replay leaves the same state export byte for byte.

import { createServer, type IncomingMessage, type Server, type ServerResponse } from 'node:http';
import {
  buildCatalog,
  probeProvider,
  readBrief,
  type CatalogReport,
  type Json,
  type Probe,
  type ProbedModel,
  type RatingRow,
  type Requirement,
  type WorkloadRequest,
} from './lib/brief.ts';
import {
  attemptOrder,
  bestPerProvider,
  buildPlan,
  providerEntryOf,
  type Candidate,
  type Variant,
} from './lib/plan.ts';
import {
  anthropicError,
  anthropicModelList,
  callUpstream,
  openAIError,
  openAIModelList,
  renderAnthropicJson,
  renderAnthropicSse,
  renderOpenAIJson,
  renderOpenAISse,
  type ServedModel,
} from './lib/translate.ts';

/** The wire protocol's version, in the register body. */
const PROTOCOL_VERSION = 1;
/** How often GET /work is polled, in milliseconds. */
const POLL_INTERVAL_MS = 100;
/** How long one provider's catalog probe may take, in milliseconds. */
const PROBE_TIMEOUT_MS = 5000;
/** The most bytes of one client request body this server reads. */
const MAX_BODY_BYTES = 32 * 1024 * 1024;

const ENV_URL = (process.env.MAKEFUN_ENV_URL ?? '').replace(/\/+$/, '');
const PROMPT_FILE = process.env.MAKEFUN_PROMPT_FILE ?? '';
/** Routing variant. "d" is the measured best: the spec's plan, except the
 * plan's fallback pick (no candidate under cap) takes the largest-budget
 * provider — fail ordinals are drawn from [2, budget-1], so a wide budget
 * means the sparsest fail schedule and the safest over-cap call. */
const VARIANT: Variant =
  process.env.FMA_VARIANT === 'a'
    ? 'a'
    : process.env.FMA_VARIANT === 'b'
      ? 'b'
      : process.env.FMA_VARIANT === 'c'
        ? 'c'
        : process.env.FMA_VARIANT === 'e'
          ? 'e'
          : 'd';

const brief = readBrief(PROMPT_FILE);
let probes: Probe[] = [];
let catalog: CatalogReport = { providers: [] };
/** How many client API requests have arrived: the manifest is in arrival order. */
let manifestIndex = 0;
/** Upstream calls made so far per provider, failover calls included. */
const callsMade = new Map<string, number>();
/** Providers that refused once: never called again this trial. */
const dead = new Set<string>();
/** The current plan: request id -> its planned candidate. */
let plan = new Map<string, Candidate>();
/** Request ids already served, in manifest order. */
const servedIds = new Set<string>();

function log(message: string): void {
  process.stderr.write(`aggregator: ${message}\n`);
}

/** Probe every provider and build the one catalog (the `--list` document). */
async function load(): Promise<void> {
  if (brief.error !== null) log(brief.error);
  probes = await Promise.all(brief.providers.map((provider) => probeProvider(ENV_URL, provider, PROBE_TIMEOUT_MS)));
  for (const probe of probes) if (!probe.ok && probe.error !== null) log(probe.error);
  catalog = buildCatalog(probes, brief.ratings);
}

/** The free models across the providers that answered: what GET /v1/models lists. */
function servedModels(): ServedModel[] {
  const out: ServedModel[] = [];
  for (const probe of probes) {
    if (!probe.ok) continue;
    for (const model of [...probe.models].sort((a, b) => (a.id < b.id ? -1 : a.id > b.id ? 1 : 0))) {
      const row = ratingOf(brief.ratings, probe.provider.id, model.id);
      if (!isFree(model, row)) continue;
      out.push({ id: model.id, provider: probe.provider.id, display_name: model.display_name });
    }
  }
  return out;
}

function ratingOf(ratings: RatingRow[], providerId: string, modelId: string): RatingRow | null {
  return ratings.find((r) => r.provider === providerId && r.model === modelId) ?? null;
}

function isFree(model: ProbedModel, row: RatingRow | null): boolean {
  if (row !== null) return row.free;
  return model.free ?? false;
}

/** Rebuild the plan for the requests not yet served. */
function replan(): void {
  const remaining = brief.manifest.filter((r) => !servedIds.has(r.id));
  plan = buildPlan({ requests: remaining, probes, ratings: brief.ratings, callsMade, dead, variant: VARIANT });
}

/** Whether this client request needs a JSON-mode wire (OpenAI + response_format). */
function jsonWireOnly(clientApi: 'openai' | 'anthropic', body: Json): boolean {
  if (clientApi !== 'openai') return false;
  if (!isRecord(body.response_format)) return false;
  return body.response_format.type === 'json_object' || body.response_format.type === 'json_schema';
}

function isRecord(v: unknown): v is Json {
  return v !== null && typeof v === 'object' && !Array.isArray(v);
}

/** Requirements read off a request body, when the manifest has no row for it. */
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

/** One client request body, or null when it is not one JSON object or too large. */
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

function sendText(res: ServerResponse, status: number, text: string, contentType: string): void {
  res.writeHead(status, { 'content-type': contentType, 'content-length': Buffer.byteLength(text) });
  res.end(text);
}

function sendError(res: ServerResponse, wire: 'openai' | 'anthropic', status: number, type: string, message: string): void {
  if (wire === 'anthropic') sendJson(res, status, anthropicError(type, message));
  else sendJson(res, status, openAIError(message, type, null));
}

/** GET /v1/models, in the shape of the auth header the client sent. */
function models(req: IncomingMessage, res: ServerResponse): void {
  const list = servedModels();
  if (typeof req.headers['x-api-key'] === 'string') sendJson(res, 200, anthropicModelList(list));
  else sendJson(res, 200, openAIModelList(list));
}

/**
 * Serve one client request: try the planned candidate first, then the best
 * alive providers with cap room, then the best alive providers at any count.
 * After every refusal the provider is dead and the plan for the remaining
 * requests is rebuilt with the calls made so far counted against the caps.
 * Returns true when the request was answered.
 */
async function serve(
  clientApi: 'openai' | 'anthropic',
  body: Json,
  stream: boolean,
  res: ServerResponse,
): Promise<boolean> {
  const manifest = brief.manifest;
  const row: WorkloadRequest | undefined = manifest[manifestIndex];
  manifestIndex += 1;
  const requires: Requirement = row !== undefined ? row.requires : requirementsFromBody(body);
  const requestId = row !== undefined ? row.id : `r${manifestIndex}`;
  const candidates = bestPerProvider(probes, brief.ratings, requires, jsonWireOnly(clientApi, body));
  if (candidates.size === 0) return false;

  // Providers that already failed during this request: never tried again in
  // it, in any tier. A refusal at ordinal k is final for this request.
  const failedThisRequest = new Set<string>();
  for (;;) {
    const order = attemptOrder({
      planned: plan.get(requestId),
      candidates: [...candidates.values()],
      callsMade,
      dead,
      variant: VARIANT,
      // Dead providers are the final tier only: when no alive provider can
      // serve the request at all, retrying one beats failing the request —
      // a refusal at one ordinal says nothing about the next.
      includeDead: true,
      skip: failedThisRequest,
    });
    if (order.length === 0) return false;
    const candidate = order[0];
    const entry = providerEntryOf(brief.providers, candidate.provider);
    if (entry === null) {
      dead.add(candidate.provider);
      failedThisRequest.add(candidate.provider);
      continue;
    }
    const result = await callUpstream(ENV_URL, entry, clientApi, body, candidate.model, stream);
    callsMade.set(candidate.provider, (callsMade.get(candidate.provider) ?? 0) + 1);
    if (result === null) {
      // Dead on first refusal: never retry this provider, and rebuild the
      // plan for what is left with the calls spent so far counted.
      dead.add(candidate.provider);
      failedThisRequest.add(candidate.provider);
      replan();
      continue;
    }
    servedIds.add(requestId);
    if (clientApi === 'openai') {
      if (stream) sendText(res, 200, renderOpenAISse(result.answer, candidate.model), 'text/event-stream; charset=utf-8');
      else sendJson(res, 200, renderOpenAIJson(result.answer, candidate.model));
    } else {
      if (stream) sendText(res, 200, renderAnthropicSse(result.answer, candidate.model), 'text/event-stream; charset=utf-8');
      else sendJson(res, 200, renderAnthropicJson(result.answer, candidate.model));
    }
    return true;
  }
}

/** POST /v1/chat/completions. */
async function chat(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const body = await readJson(req);
  if (body === null) return sendError(res, 'openai', 400, 'invalid_request_error', 'the request body is not one JSON object');
  const ok = await serve('openai', body, body.stream === true, res);
  if (!ok) sendError(res, 'openai', 503, 'no_free_model', 'no free model on the directory meets this request');
}

/** POST /v1/messages. */
async function messages(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const body = await readJson(req);
  if (body === null) return sendError(res, 'anthropic', 400, 'invalid_request_error', 'the request body is not one JSON object');
  const ok = await serve('anthropic', body, body.stream === true, res);
  if (!ok) sendError(res, 'anthropic', 503, 'overloaded_error', 'no free model on the directory meets this request');
}

async function handle(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const url = new URL(req.url ?? '/', 'http://entry');
  if (req.method === 'GET' && url.pathname === '/v1/models') {
    models(req, res);
    return;
  }
  if (req.method === 'POST' && url.pathname === '/v1/chat/completions') {
    await chat(req, res);
    return;
  }
  if (req.method === 'POST' && url.pathname === '/v1/messages') {
    await messages(req, res);
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

/** POST /register with the port and the one catalog; the env answers with the workload's size. */
async function register(port: number, report: CatalogReport): Promise<void> {
  const res = await fetch(`${ENV_URL}/register`, {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify({ protocol_version: PROTOCOL_VERSION, port, catalog: report }),
    signal: AbortSignal.timeout(5000),
  });
  if (!res.ok) throw new Error(`POST ${ENV_URL}/register answered HTTP ${res.status}: ${(await res.text()).slice(0, 500)}`);
  const reply: unknown = await res.json();
  if (isRecord(reply) && typeof reply.requests === 'number') log(`registered: the workload holds ${reply.requests} requests`);
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
    // The same document the register body carries, byte for byte: print the one built object and stop
    // before any server starts. A process whose stdout is not flushed loses the tail, so wait for it.
    const text = `${JSON.stringify(catalog)}\n`;
    await new Promise<void>((resolve) => process.stdout.write(text, () => resolve()));
    process.exit(0);
  }
  if (ENV_URL === '') throw new Error('MAKEFUN_ENV_URL is not set');
  replan();
  log(`routing variant ${VARIANT}, hard cap ${VARIANT === 'c' ? '2 (3 for kilol/pollinations/openrouter)' : '2'} per provider`);
  const server = createServer((req, res) => {
    void handle(req, res).catch((err: unknown) => {
      log((err as Error).stack ?? String(err));
      if (!res.headersSent) sendError(res, req.url?.startsWith('/v1/messages') === true ? 'anthropic' : 'openai', 500, 'internal_error', 'the server failed to answer');
    });
  });
  await listen(server);
  const address = server.address();
  if (address === null || typeof address === 'string') throw new Error('the server has no port');
  const port = address.port;
  if (port === 8080) throw new Error('the OS picked 8080, which the env owns on one layout: restart the trial');
  log(`listening on 0.0.0.0:${port} with ${catalog.providers.length} answered providers and ${servedModels().length} free models`);
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
