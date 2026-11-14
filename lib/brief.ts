// Brief parsing: the trial brief (MAKEFUN_PROMPT_FILE) is prose wrapped around
// three JSON documents, found by their `schema` member. This file reads those
// three documents and nothing else: the provider directory, the sealed quality
// table, and the workload manifest.
//
// The three documents:
//   free-model-aggregator-01/providers/v1  - ten providers: id, wire, auth,
//     key env var, key, base path.
//   free-model-aggregator-01/ratings/v1    - every model's rating_milli,
//     context length, tools/vision/json capability, free flag, plus each
//     request's best rating.
//   free-model-aggregator-01/workload/v1   - the twenty requests in order:
//     id, api, stream, weight, requirements.

import { readFileSync } from 'node:fs';

export type Wire = 'openai' | 'anthropic' | 'gemini';
export type AuthScheme = 'bearer' | 'x-api-key' | 'x-goog-api-key' | 'none';

export const PROVIDERS_SCHEMA = 'free-model-aggregator-01/providers/v1';
export const RATINGS_SCHEMA = 'free-model-aggregator-01/ratings/v1';
export const WORKLOAD_SCHEMA = 'free-model-aggregator-01/workload/v1';

export type Json = Record<string, unknown>;

export function isRecord(v: unknown): v is Json {
  return v !== null && typeof v === 'object' && !Array.isArray(v);
}

export interface ProviderEntry {
  id: string;
  name: string;
  wire: Wire;
  auth: AuthScheme;
  /** Base path (e.g. "/p/groq") or absolute URL, as the directory spells it. */
  base: string;
  keyEnv: string;
  key: string;
}

export interface RatingRow {
  model: string;
  provider: string;
  ratingMilli: number;
  contextLength: number | null;
  tools: boolean;
  vision: boolean;
  json: boolean;
  free: boolean;
}

export interface Requirement {
  tools?: boolean;
  vision?: boolean;
  json?: boolean;
  minContext?: number;
}

export interface WorkloadRow {
  id: string;
  api: 'openai' | 'anthropic';
  stream: boolean;
  weight: number;
  requires: Requirement;
}

export interface Brief {
  providers: ProviderEntry[];
  ratings: RatingRow[];
  best: Map<string, number>;
  workload: WorkloadRow[];
  /** Non-null when the brief could not be read at all. */
  error: string | null;
}

function asString(v: unknown): string {
  return typeof v === 'string' ? v : '';
}

function asNumberOrNull(v: unknown): number | null {
  return typeof v === 'number' && Number.isFinite(v) ? v : null;
}

function asWire(v: unknown): Wire | null {
  return v === 'openai' || v === 'anthropic' || v === 'gemini' ? v : null;
}

function asAuth(v: unknown): AuthScheme {
  if (v === 'bearer' || v === 'x-api-key' || v === 'x-goog-api-key' || v === 'none') return v;
  return 'bearer';
}

/** Index of the `}` closing the object that opens at `start`, or -1. */
function matchingBrace(text: string, start: number): number {
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

/** Every top-level JSON object in `text` that carries a string `schema`. */
function schemaDocuments(text: string): Json[] {
  const out: Json[] = [];
  for (let at = text.indexOf('{'); at !== -1; at = text.indexOf('{', at + 1)) {
    const end = matchingBrace(text, at);
    if (end === -1) continue;
    try {
      const value: unknown = JSON.parse(text.slice(at, end + 1));
      if (isRecord(value) && typeof value.schema === 'string') out.push(value);
    } catch {
      // Not a document on its own; keep scanning.
    }
  }
  return out;
}

function parseProvider(v: unknown): ProviderEntry | null {
  if (!isRecord(v)) return null;
  const id = asString(v.id);
  const wire = asWire(v.wire);
  if (id === '' || wire === null) return null;
  const base =
    asString(v.path) !== ''
      ? asString(v.path)
      : asString(v.base_path) !== ''
        ? asString(v.base_path)
        : asString(v.base_url) !== ''
          ? asString(v.base_url)
          : asString(v.base);
  return {
    id,
    name: asString(v.name) !== '' ? asString(v.name) : id,
    wire,
    auth: asAuth(v.auth),
    base,
    keyEnv: asString(v.key_env),
    key: asString(v.key),
  };
}

function parseRating(v: unknown): RatingRow | null {
  if (!isRecord(v)) return null;
  const model = asString(v.model);
  const provider = asString(v.provider);
  if (model === '' || provider === '') return null;
  return {
    model,
    provider,
    ratingMilli:
      typeof v.rating_milli === 'number' && Number.isFinite(v.rating_milli) ? v.rating_milli : 0,
    contextLength: asNumberOrNull(v.context_length),
    tools: v.tools === true,
    vision: v.vision === true,
    json: v.json === true,
    free: v.free === true,
  };
}

function parseWorkloadRow(v: unknown): WorkloadRow | null {
  if (!isRecord(v)) return null;
  const id = asString(v.id);
  if (id === '') return null;
  const req = isRecord(v.requires) ? v.requires : {};
  const out: Requirement = {};
  if (req.tools === true) out.tools = true;
  if (req.vision === true) out.vision = true;
  if (req.json === true) out.json = true;
  if (typeof req.min_context === 'number' && Number.isFinite(req.min_context)) {
    out.minContext = req.min_context;
  }
  return {
    id,
    api: v.api === 'anthropic' ? 'anthropic' : 'openai',
    stream: v.stream === true,
    weight: v.weight === 2 ? 2 : 1,
    requires: out,
  };
}

export function emptyBrief(error: string): Brief {
  return { providers: [], ratings: [], best: new Map(), workload: [], error };
}

/** Read the brief at `path`. A missing path yields an empty brief with `error` set. */
export function readBrief(path: string): Brief {
  if (path === '') return emptyBrief('MAKEFUN_PROMPT_FILE is not set');
  let text: string;
  try {
    text = readFileSync(path, 'utf8');
  } catch (err) {
    return emptyBrief(`cannot read ${path}: ${(err as Error).message}`);
  }
  const docs = schemaDocuments(text);
  const dirDoc = docs.find((d) => d.schema === PROVIDERS_SCHEMA && Array.isArray(d.providers));
  if (dirDoc === undefined || !Array.isArray(dirDoc.providers)) {
    return emptyBrief(`the brief at ${path} holds no ${PROVIDERS_SCHEMA} document`);
  }
  const providers = dirDoc.providers.flatMap((v) => {
    const p = parseProvider(v);
    return p === null ? [] : [p];
  });
  const ratingsDoc = docs.find((d) => d.schema === RATINGS_SCHEMA && Array.isArray(d.ratings));
  const ratings: RatingRow[] = [];
  const best = new Map<string, number>();
  if (ratingsDoc !== undefined && Array.isArray(ratingsDoc.ratings)) {
    for (const v of ratingsDoc.ratings) {
      const r = parseRating(v);
      if (r !== null) ratings.push(r);
    }
    if (Array.isArray(ratingsDoc.best)) {
      for (const v of ratingsDoc.best) {
        if (isRecord(v) && asString(v.id) !== '' && typeof v.rating_milli === 'number') {
          best.set(asString(v.id), v.rating_milli);
        }
      }
    }
  }
  const workloadDoc = docs.find((d) => d.schema === WORKLOAD_SCHEMA && Array.isArray(d.manifest));
  const workload: WorkloadRow[] = [];
  if (workloadDoc !== undefined && Array.isArray(workloadDoc.manifest)) {
    for (const v of workloadDoc.manifest) {
      const w = parseWorkloadRow(v);
      if (w !== null) workload.push(w);
    }
  }
  return { providers, ratings, best, workload, error: null };
}

/** The quality-table row for one provider's model, when the table has one. */
export function ratingRowOf(brief: Brief, providerId: string, modelId: string): RatingRow | null {
  return brief.ratings.find((r) => r.provider === providerId && r.model === modelId) ?? null;
}

/** Join a provider base (path or absolute URL) onto the env origin. */
export function joinBase(envUrl: string, base: string): string {
  const b = base.trim();
  if (/^https?:\/\//i.test(b)) return b.replace(/\/+$/, '');
  const prefix = envUrl.replace(/\/+$/, '');
  if (b === '') return prefix;
  return `${prefix}${b.startsWith('/') ? '' : '/'}${b.replace(/\/+$/, '')}`;
}

/** The headers that carry one provider's key, per its own auth scheme. */
export function authHeaders(auth: AuthScheme, key: string): Record<string, string> {
  if (key === '') return {};
  if (auth === 'bearer') return { authorization: `Bearer ${key}` };
  if (auth === 'x-api-key') return { 'x-api-key': key };
  if (auth === 'x-goog-api-key') return { 'x-goog-api-key': key };
  return {};
}
