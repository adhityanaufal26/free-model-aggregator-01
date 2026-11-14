// Catalog probing and the catalog document.
//
// At startup the server probes every provider's catalog route in that
// provider's own wire, in parallel, and builds the one catalog that `--list`
// prints and that the register body carries, byte for byte.
//
//   openai     GET <base>/v1/models          Authorization: Bearer <key>
//   anthropic  GET <base>/v1/models          x-api-key: <key>,
//                                            anthropic-version: 2023-06-01
//   gemini     GET <base>/v1beta/models      x-goog-api-key: <key>
//
// A probe that fails is retried once, then its provider is left out of the
// catalog and gets no traffic. The catalog's `free` flag and context length
// come from the brief's quality table where it has a row (the sealed truth),
// and from the listing itself where it does not. Paid models are listed too,
// with free:false: the catalog is scored against the providers' true
// catalogs, free and paid alike.

import { authHeaders, joinBase, ratingRowOf, type Brief, type ProviderEntry, type Wire } from './brief.ts';

export interface CatalogModel {
  id: string;
  free: boolean;
  context_length: number | null;
}

export interface CatalogProvider {
  id: string;
  base_url: string;
  key_env: string;
  wire: Wire;
  models: CatalogModel[];
}

export interface Catalog {
  providers: CatalogProvider[];
}

export interface ListedModel {
  id: string;
  displayName: string;
  free: boolean | null;
  contextLength: number | null;
}

export interface Probe {
  provider: ProviderEntry;
  ok: boolean;
  status: number;
  models: ListedModel[];
  error: string | null;
  /** Probe round-trip in whole milliseconds, dev diagnostics only. */
  probeMs: number;
}

export const EMPTY_CATALOG: Catalog = { providers: [] };

function asString(v: unknown): string {
  return typeof v === 'string' ? v : '';
}

function asNumberOrNull(v: unknown): number | null {
  return typeof v === 'number' && Number.isFinite(v) ? v : null;
}

function asBoolOrNull(v: unknown): boolean | null {
  return typeof v === 'boolean' ? v : null;
}

/** A price object whose every price is zero means the listing is free. */
function zeroPriced(v: Record<string, unknown>): boolean | null {
  const pricing = v.pricing;
  if (pricing === null || typeof pricing !== 'object' || Array.isArray(pricing)) return null;
  const values: number[] = [];
  for (const x of Object.values(pricing as Record<string, unknown>)) {
    if (typeof x === 'number') values.push(x);
    else if (typeof x === 'string' && x.trim() !== '') values.push(Number(x));
  }
  if (values.length === 0) return null;
  return values.every((n) => Number.isFinite(n) && n === 0);
}

function modelFromListing(wire: Wire, v: unknown): ListedModel | null {
  if (v === null || typeof v !== 'object' || Array.isArray(v)) return null;
  const rec = v as Record<string, unknown>;
  const id =
    wire === 'gemini' ? asString(rec.name).replace(/^models\//, '') : asString(rec.id);
  if (id === '') return null;
  return {
    id,
    displayName:
      asString(rec.display_name) !== ''
        ? asString(rec.display_name)
        : asString(rec.displayName) !== ''
          ? asString(rec.displayName)
          : id,
    free: asBoolOrNull(rec.free) ?? zeroPriced(rec),
    contextLength:
      asNumberOrNull(rec.context_length) ??
      asNumberOrNull(rec.context_window) ??
      asNumberOrNull(rec.inputTokenLimit),
  };
}

function modelsFromBody(wire: Wire, body: unknown): ListedModel[] {
  if (body === null || typeof body !== 'object' || Array.isArray(body)) return [];
  const rec = body as Record<string, unknown>;
  const list = wire === 'gemini' ? rec.models : rec.data;
  if (!Array.isArray(list)) return [];
  const out: ListedModel[] = [];
  for (const v of list) {
    const m = modelFromListing(wire, v);
    if (m !== null) out.push(m);
  }
  return out;
}

async function probeOnce(
  envUrl: string,
  provider: ProviderEntry,
  timeoutMs: number,
): Promise<Probe> {
  const base = joinBase(envUrl, provider.base);
  const url = provider.wire === 'gemini' ? `${base}/v1beta/models` : `${base}/v1/models`;
  const headers: Record<string, string> = {
    accept: 'application/json',
    ...authHeaders(provider.auth, provider.key),
  };
  if (provider.wire === 'anthropic') headers['anthropic-version'] = '2023-06-01';
  const started = Date.now();
  try {
    const res = await fetch(url, { headers, signal: AbortSignal.timeout(timeoutMs) });
    const ms = Date.now() - started;
    if (!res.ok) {
      return { provider, ok: false, status: res.status, models: [], error: `GET ${url} answered HTTP ${res.status}`, probeMs: ms };
    }
    const body: unknown = await res.json();
    return { provider, ok: true, status: res.status, models: modelsFromBody(provider.wire, body), error: null, probeMs: ms };
  } catch (err) {
    return { provider, ok: false, status: 0, models: [], error: `GET ${url} failed: ${(err as Error).message}`, probeMs: Date.now() - started };
  }
}

/** Probe one provider, with a single retry on failure. */
export async function probeProvider(
  envUrl: string,
  provider: ProviderEntry,
  timeoutMs = 5000,
): Promise<Probe> {
  const first = await probeOnce(envUrl, provider, timeoutMs);
  if (first.ok) return first;
  return probeOnce(envUrl, provider, timeoutMs);
}

/** Probe every provider in parallel. */
export async function probeAll(
  envUrl: string,
  providers: readonly ProviderEntry[],
  timeoutMs = 5000,
): Promise<Probe[]> {
  return Promise.all(providers.map((p) => probeProvider(envUrl, p, timeoutMs)));
}

const byId = (a: { id: string }, b: { id: string }): number =>
  a.id < b.id ? -1 : a.id > b.id ? 1 : 0;

/**
 * The catalog: every provider that answered, every model it listed, models
 * sorted by id. `base_url` is the directory's own path ("/p/<id>"), never the
 * path joined to the env origin: the origin differs between a live pass and
 * its replay, and the catalog is part of the state export a replay must
 * reproduce byte for byte.
 */
export function buildCatalog(probes: readonly Probe[], brief: Brief): Catalog {
  const providers: CatalogProvider[] = [];
  for (const probe of probes) {
    if (!probe.ok) continue;
    const seen = new Set<string>();
    const models: CatalogModel[] = [];
    for (const model of [...probe.models].sort(byId)) {
      if (seen.has(model.id)) continue;
      seen.add(model.id);
      const row = ratingRowOf(brief, probe.provider.id, model.id);
      models.push({
        id: model.id,
        free: row !== null ? row.free : (model.free ?? false),
        context_length: row?.contextLength ?? model.contextLength ?? null,
      });
    }
    providers.push({
      id: probe.provider.id,
      base_url: probe.provider.base,
      key_env: probe.provider.keyEnv,
      wire: probe.provider.wire,
      models,
    });
  }
  return { providers };
}

/** One free model the entry serves, for GET /v1/models. */
export interface ServedModel {
  id: string;
  provider: string;
  displayName: string;
}

/**
 * The free models across the providers that answered, in the brief's provider
 * order, deduplicated by model id (one model id can be free on several
 * providers; the first provider in brief order owns the listing).
 */
export function servedModels(probes: readonly Probe[], brief: Brief): ServedModel[] {
  const out: ServedModel[] = [];
  const seen = new Set<string>();
  for (const probe of probes) {
    if (!probe.ok) continue;
    for (const model of [...probe.models].sort(byId)) {
      if (seen.has(model.id)) continue;
      const row = ratingRowOf(brief, probe.provider.id, model.id);
      const free = row !== null ? row.free : (model.free ?? false);
      if (!free) continue;
      seen.add(model.id);
      out.push({ id: model.id, provider: probe.provider.id, displayName: model.displayName });
    }
  }
  return out;
}
