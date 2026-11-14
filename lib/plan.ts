// Routing: quality-ranked candidates, a budget-aware upfront plan, and
// adaptive failover.
//
// The brief's quality table rates every free model. The naive best-first
// router sends nearly all twenty requests to the single best model, whose
// per-trial budget (4-16 calls, drawn per trial and never published) then
// runs dry and takes the tail of the workload down with it. This router
// instead:
//
//   1. Plans the whole known workload up front. The manifest lists all
//      twenty requests, so before serving anything it greedily assigns each
//      request its best eligible model, weight-first, subject to a soft
//      per-provider cap. The caps are the published minimum budgets of the
//      kit's provider table (calls_min): groq 9, cerebras 5, gemini 6,
//      zai 4, openrouter 10, sambanova 4, pollinations 10, cloudflare 7,
//      ovh 5, kilol 12. A cap is a planning prior, not a limit: when every
//      provider is at its cap the best model still wins.
//   2. Fails over on any refused call (402, 429, 5xx, timeout, unreachable)
//      to the next eligible model by rating, with providers that already
//      refused pushed down 45 milli per refusal, so a provider that is out
//      of budget stops being tried first. Refused providers are retried
//      only after every untried provider has been tried, and at most five
//      provider calls are spent on one request.
//   3. Falls back to live ranking (rating - 4 milli per call already made
//      to the provider - 45 per refusal) when the brief carries no workload.
//
// Eligibility is the model's capabilities against the request's
// requirements, from the quality table. One wire rule on top: an OpenAI
// request in JSON mode (response_format json_object/json_schema) may only
// go to an openai- or gemini-wire provider, because the Anthropic wire has
// no way to carry JSON mode and its echo would report it absent. (An
// Anthropic client never sends JSON mode on its wire, so its json
// requirements are a capability check only.)
//
// Every choice is a pure function of the brief, the requests, and the
// providers' answers so far: no clock, no unseeded randomness, so a replay
// leaves the same state export.

import { ratingRowOf, type Brief, type Requirement, type WorkloadRow } from './brief.ts';
import type { Probe } from './catalog.ts';

export interface Candidate {
  provider: string;
  model: string;
  ratingMilli: number;
  wire: 'openai' | 'anthropic' | 'gemini';
}

export interface Caps {
  contextLength: number | null;
  tools: boolean | null;
  vision: boolean | null;
  json: boolean | null;
}

/** Published minimum per-trial budgets (the kit provider table's calls_min). */
export const SOFT_CAP: Record<string, number> = {
  groq: 9,
  cerebras: 5,
  gemini: 6,
  zai: 4,
  openrouter: 10,
  sambanova: 4,
  pollinations: 10,
  cloudflare: 7,
  ovh: 5,
  kilol: 12,
};

/** Penalty in milli-points per refusal from a provider, for failover order. */
export const REFUSAL_PENALTY_MILLI = 45;
/** Penalty in milli-points per call already made to a provider, live ranking. */
export const LOAD_PENALTY_MILLI = 4;
/** Most provider calls spent on one client request. */
export const MAX_ATTEMPTS = 5;

export function meetsRequirements(req: Requirement, caps: Caps): boolean {
  if (req.tools === true && caps.tools !== true) return false;
  if (req.vision === true && caps.vision !== true) return false;
  if (req.json === true && caps.json !== true) return false;
  if (req.minContext !== undefined && (caps.contextLength === null || caps.contextLength < req.minContext)) {
    return false;
  }
  return true;
}

/** All free models that can serve `req`, best rating first. */
export function eligibleCandidates(
  probes: readonly Probe[],
  brief: Brief,
  req: Requirement,
  jsonWire: boolean,
): Candidate[] {
  const out: Candidate[] = [];
  for (const probe of probes) {
    if (!probe.ok) continue;
    if (jsonWire && probe.provider.wire === 'anthropic') continue;
    for (const model of probe.models) {
      const row = ratingRowOf(brief, probe.provider.id, model.id);
      const free = row !== null ? row.free : (model.free ?? false);
      if (!free) continue;
      const caps: Caps = {
        contextLength: row?.contextLength ?? model.contextLength ?? null,
        tools: row?.tools ?? null,
        vision: row?.vision ?? null,
        json: row?.json ?? null,
      };
      if (!meetsRequirements(req, caps)) continue;
      out.push({
        provider: probe.provider.id,
        model: model.id,
        ratingMilli: row?.ratingMilli ?? 0,
        wire: probe.provider.wire,
      });
    }
  }
  out.sort((a, b) => {
    if (b.ratingMilli !== a.ratingMilli) return b.ratingMilli - a.ratingMilli;
    if (a.provider !== b.provider) return a.provider < b.provider ? -1 : 1;
    return a.model < b.model ? -1 : a.model > b.model ? 1 : 0;
  });
  return out;
}

export interface RouterState {
  probes: Probe[];
  brief: Brief;
  /** The upfront assignment: request id -> candidate. */
  plan: Map<string, Candidate>;
  /** Provider calls made so far (successes and refusals alike). */
  calls: Map<string, number>;
  /** Provider refusals so far. */
  refusals: Map<string, number>;
  /** Arrival counter for requests with no manifest row. */
  arrivals: number;
}

export function newRouterState(probes: Probe[], brief: Brief): RouterState {
  return { probes, brief, plan: new Map(), calls: new Map(), refusals: new Map(), arrivals: 0 };
}

function capOf(providerId: string): number {
  return SOFT_CAP[providerId] ?? 4;
}

/**
 * The upfront plan: weight-first, best-model-first, one provider per request
 * until its soft cap is reached. `jsonWire` marks the requests whose client
 * asked for JSON mode on the OpenAI wire.
 */
export function buildPlan(
  state: RouterState,
  jsonWireById: (row: WorkloadRow) => boolean,
): void {
  const ordered = [...state.brief.workload].sort((a, b) => {
    if (b.weight !== a.weight) return b.weight - a.weight;
    return state.brief.workload.indexOf(a) - state.brief.workload.indexOf(b);
  });
  const planned = new Map<string, number>();
  for (const row of ordered) {
    const candidates = eligibleCandidates(state.probes, state.brief, row.requires, jsonWireById(row));
    if (candidates.length === 0) continue;
    // Best rating tier first; inside a tier, the provider with the fewest
    // planned calls wins, so two providers sharing one rating (zai and
    // cloudflare both serve glm-4.7-flash at 1470) split the load instead
    // of one of them absorbing the whole tier and running dry.
    let pick: Candidate | undefined;
    const tiers = new Map<number, Candidate[]>();
    for (const c of candidates) {
      const list = tiers.get(c.ratingMilli) ?? [];
      list.push(c);
      tiers.set(c.ratingMilli, list);
    }
    for (const rating of [...tiers.keys()].sort((a, b) => b - a)) {
      const tier = tiers.get(rating) ?? [];
      const under = tier.filter((c) => (planned.get(c.provider) ?? 0) < capOf(c.provider));
      if (under.length === 0) continue;
      under.sort((a, b) => {
        const d = (planned.get(a.provider) ?? 0) - (planned.get(b.provider) ?? 0);
        if (d !== 0) return d;
        if (a.provider !== b.provider) return a.provider < b.provider ? -1 : 1;
        return a.model < b.model ? -1 : a.model > b.model ? 1 : 0;
      });
      pick = under[0];
      break;
    }
    if (pick === undefined) pick = candidates[0];
    state.plan.set(row.id, pick);
    planned.set(pick.provider, (planned.get(pick.provider) ?? 0) + 1);
  }
}

/** Live ranking used when no plan covers a request: rating, load, refusals. */
export function liveRank(
  state: RouterState,
  req: Requirement,
  jsonWire: boolean,
): Candidate[] {
  const candidates = eligibleCandidates(state.probes, state.brief, req, jsonWire);
  const score = (c: Candidate): number =>
    c.ratingMilli -
    LOAD_PENALTY_MILLI * (state.calls.get(c.provider) ?? 0) -
    REFUSAL_PENALTY_MILLI * (state.refusals.get(c.provider) ?? 0);
  candidates.sort((a, b) => {
    const d = score(b) - score(a);
    if (d !== 0) return d;
    if (a.provider !== b.provider) return a.provider < b.provider ? -1 : 1;
    return a.model < b.model ? -1 : a.model > b.model ? 1 : 0;
  });
  return candidates;
}

/**
 * The ordered attempt list for one request: the planned candidate first,
 * then every other eligible model by refusal-penalised rating, then (only
 * if attempts remain) the refused providers again. Already-tried
 * (provider, model) pairs never repeat.
 */
export function attemptList(
  state: RouterState,
  requestId: string | null,
  req: Requirement,
  jsonWire: boolean,
): Candidate[] {
  const planned = requestId !== null ? state.plan.get(requestId) : undefined;
  const ranked = eligibleCandidates(state.probes, state.brief, req, jsonWire);
  const penalised = (c: Candidate): number =>
    c.ratingMilli - REFUSAL_PENALTY_MILLI * (state.refusals.get(c.provider) ?? 0);
  ranked.sort((a, b) => {
    const d = penalised(b) - penalised(a);
    if (d !== 0) return d;
    // Same penalised rating: the provider called least so far goes first,
    // so a tier shared by two providers splits instead of draining one.
    const calls = (state.calls.get(a.provider) ?? 0) - (state.calls.get(b.provider) ?? 0);
    if (calls !== 0) return calls;
    if (a.provider !== b.provider) return a.provider < b.provider ? -1 : 1;
    return a.model < b.model ? -1 : a.model > b.model ? 1 : 0;
  });
  const out: Candidate[] = [];
  const seen = new Set<string>();
  const push = (c: Candidate): void => {
    const key = `${c.provider}/${c.model}`;
    if (seen.has(key)) return;
    seen.add(key);
    out.push(c);
  };
  // The plan's pick goes first, but only when it is still eligible for this
  // request (a JSON-mode body, for instance, rules out an anthropic-wire
  // pick the manifest could not see coming).
  if (planned !== undefined && ranked.some((c) => c.provider === planned.provider && c.model === planned.model)) {
    push(planned);
  }
  const refusedProviders = new Set(
    [...state.refusals.entries()].filter(([, n]) => n > 0).map(([p]) => p),
  );
  for (const c of ranked) if (!refusedProviders.has(c.provider)) push(c);
  for (const c of ranked) push(c);
  return out.slice(0, MAX_ATTEMPTS);
}

/** A request body's own requirements, when the manifest has no row for it. */
export function requirementsFromBody(body: Record<string, unknown>): Requirement {
  const out: Requirement = {};
  const messages = Array.isArray(body.messages) ? body.messages : [];
  let vision = false;
  for (const message of messages) {
    if (message === null || typeof message !== 'object' || Array.isArray(message)) continue;
    const content = (message as Record<string, unknown>).content;
    if (!Array.isArray(content)) continue;
    for (const block of content) {
      if (block !== null && typeof block === 'object' && !Array.isArray(block)) {
        const t = (block as Record<string, unknown>).type;
        if (t === 'image' || t === 'image_url') vision = true;
      }
    }
  }
  if (Array.isArray(body.tools) && body.tools.length > 0) out.tools = true;
  if (vision) out.vision = true;
  const format = body.response_format;
  if (format !== null && typeof format === 'object' && !Array.isArray(format)) {
    const t = (format as Record<string, unknown>).type;
    if (t === 'json_object' || t === 'json_schema') out.json = true;
  }
  return out;
}

/** Signature of a request for matching it to a manifest row. */
export function requestSignature(
  api: 'openai' | 'anthropic',
  stream: boolean,
  req: Requirement,
): string {
  return [
    api,
    stream ? 'stream' : 'json',
    req.tools === true ? 'tools' : '-',
    req.vision === true ? 'vision' : '-',
    req.json === true ? 'json' : '-',
    req.minContext !== undefined ? `ctx${req.minContext}` : '-',
  ].join('|');
}
