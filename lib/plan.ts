// lib/plan.ts — the routing plan.
//
// The game is minimizing provider calls at a fixed score of 99: quality is
// capped (the best model for 17 of 20 requests sits on one provider whose
// budget cannot cover them all), so the whole contest is won on the call count.
// The plan is built from the brief alone, at startup, and rebuilt for the
// remaining requests after every refusal:
//
//   1. Upfront plan: weight-first (weight 2, then 1; manifest order inside a
//      weight), best eligible rating first, HARD CAP 2 calls per provider
//      (10 providers x 2 = 20, exactly the workload). Never plan a provider's
//      3rd call. Tie-break inside a rating tier: larger calls_min first (the
//      kit's public provider table: kilol 12, pollinations 10, openrouter 10,
//      groq 9, cloudflare 7, gemini 6, cerebras 5, ovh 5, zai 4, sambanova 4),
//      then fewest planned calls, then provider id.
//   2. Serve in manifest order. Attempt order per request: planned candidate
//      first (if alive), then best-rating alive providers with remaining cap,
//      then best-rating alive providers at any count.
//   3. DEAD-ON-FIRST-REFUSAL: any 402/429/5xx/timeout marks that provider DEAD
//      for the rest of the trial — never retried, not even as a last resort.
//   4. REPLAN AFTER EVERY REFUSAL: the plan for the remaining requests is
//      recomputed with dead providers removed and remaining caps = cap minus
//      calls made so far. Failover calls consume cap too.
//
// Nothing here knows a trial's failure schedule: every choice derives from the
// brief (quality table, workload manifest, provider list) and from the
// providers' answers so far (calls made, refusals seen).

import {
  modelFree,
  ratingOfModel,
  type Probe,
  type ProviderEntry,
  type RatingRow,
  type Requirement,
  type WorkloadRequest,
} from './brief.ts';

/** One serving candidate: a provider and its best eligible free model. */
export interface Candidate {
  provider: string;
  model: string;
  rating: number;
}

/**
 * The kit's public provider table's calls_min, used only as a static tie-break
 * preference (larger first): a provider that can serve more calls is a safer
 * home for a planned 2nd call than one that can serve few. The brief carries
 * no calls_min, so this ordering is a constant, not per-trial data.
 */
export const CALLS_MIN_PREFERENCE: Record<string, number> = {
  kilol: 12,
  pollinations: 10,
  openrouter: 10,
  groq: 9,
  cloudflare: 7,
  gemini: 6,
  cerebras: 5,
  ovh: 5,
  zai: 4,
  sambanova: 4,
};

/** Routing variant: "a" is the spec; "b" and "c" are measured alternatives. */
export type Variant = 'a' | 'b' | 'c' | 'd' | 'e';

/** The per-provider call cap under a variant. */
export function capFor(providerId: string, variant: Variant): number {
  if (variant === 'c' && (providerId === 'kilol' || providerId === 'pollinations' || providerId === 'openrouter')) return 3;
  return 2;
}

/** Whether a model's capabilities meet one request's requirements. */
export function meets(
  requires: Requirement,
  caps: { context_length: number | null; tools: boolean | null; vision: boolean | null; json: boolean | null },
): boolean {
  if (requires.tools === true && caps.tools !== true) return false;
  if (requires.vision === true && caps.vision !== true) return false;
  if (requires.json === true && caps.json !== true) return false;
  if (requires.min_context !== undefined && (caps.context_length === null || caps.context_length < requires.min_context)) return false;
  return true;
}

/**
 * The best eligible free model per provider for one request: the highest-rated
 * free model meeting the requirements. JSON-mode OpenAI requests only ever go
 * to openai/gemini-wire providers (the Anthropic wire has no JSON mode, so its
 * echo would fail the request's JSON check); Anthropic-API requests may use
 * any wire, since their JSON requirement is a capability, not a wire mode.
 */
export function bestPerProvider(
  probes: readonly Probe[],
  ratings: RatingRow[],
  requires: Requirement,
  jsonWireOnly: boolean,
): Map<string, Candidate> {
  const out = new Map<string, Candidate>();
  for (const probe of probes) {
    if (!probe.ok) continue;
    if (jsonWireOnly && probe.provider.wire === 'anthropic') continue;
    for (const model of probe.models) {
      const row = ratingOfModel(ratings, probe.provider.id, model.id);
      if (!modelFree(model, row)) continue;
      const caps = {
        context_length: row?.context_length ?? model.context_length ?? null,
        tools: row?.tools ?? model.tools ?? null,
        vision: row?.vision ?? model.vision ?? null,
        json: row?.json ?? model.json ?? null,
      };
      if (!meets(requires, caps)) continue;
      const rating = row?.rating_milli ?? 0;
      const cur = out.get(probe.provider.id);
      if (cur === undefined || rating > cur.rating || (rating === cur.rating && model.id < cur.model)) {
        out.set(probe.provider.id, { provider: probe.provider.id, model: model.id, rating });
      }
    }
  }
  return out;
}

/**
 * Sort candidates best-first: rating desc, then the variant's tie-break, then
 * provider id. Variant "a": larger calls_min, then fewer planned calls.
 * Variant "b": fewer planned calls, ignoring calls_min.
 */
export function sortCandidates(
  candidates: Candidate[],
  planned: ReadonlyMap<string, number>,
  variant: Variant,
): Candidate[] {
  const byId = (a: Candidate, b: Candidate): number => (a.provider < b.provider ? -1 : a.provider > b.provider ? 1 : 0);
  return [...candidates].sort((a, b) => {
    if (a.rating !== b.rating) return b.rating - a.rating;
    if (variant === 'b') {
      const pa = planned.get(a.provider) ?? 0;
      const pb = planned.get(b.provider) ?? 0;
      if (pa !== pb) return pa - pb;
      return byId(a, b);
    }
    const ca = CALLS_MIN_PREFERENCE[a.provider] ?? 0;
    const cb = CALLS_MIN_PREFERENCE[b.provider] ?? 0;
    if (ca !== cb) return cb - ca;
    const pa = planned.get(a.provider) ?? 0;
    const pb = planned.get(b.provider) ?? 0;
    if (pa !== pb) return pa - pb;
    return byId(a, b);
  });
}

export interface PlanInput {
  /** Requests still unserved, in manifest order. */
  requests: WorkloadRequest[];
  probes: readonly Probe[];
  ratings: RatingRow[];
  /** Calls made so far per provider (failover calls included). */
  callsMade: ReadonlyMap<string, number>;
  dead: ReadonlySet<string>;
  variant: Variant;
}

/**
 * The plan for the remaining requests: weight-first, best rating first, hard
 * cap per provider. A request whose every candidate is at cap still gets a
 * first choice (its best candidate): the attempt order, not the plan, decides
 * whether a 3rd call is ever spent, and only as a last resort.
 */
export function buildPlan(input: PlanInput): Map<string, Candidate> {
  const { requests, probes, ratings, callsMade, dead, variant } = input;
  const ordered = [...requests].sort((a, b) => {
    if (a.weight !== b.weight) return b.weight - a.weight;
    return requests.indexOf(a) - requests.indexOf(b);
  });
  const plan = new Map<string, Candidate>();
  const planned = new Map<string, number>();
  for (const request of ordered) {
    const jsonWireOnly = request.api === 'openai' && request.requires.json === true;
    const candidates = sortCandidates(
      [...bestPerProvider(probes, ratings, request.requires, jsonWireOnly).values()].filter((c) => !dead.has(c.provider)),
      planned,
      variant,
    );
    if (candidates.length === 0) continue;
    let pick: Candidate | undefined;
    for (const candidate of candidates) {
      const remaining = capFor(candidate.provider, variant) - (callsMade.get(candidate.provider) ?? 0) - (planned.get(candidate.provider) ?? 0);
      if (remaining > 0) {
        pick = candidate;
        break;
      }
    }
    if (pick === undefined) {
      if (variant === 'd' || variant === 'e') {
        // No candidate has cap room left: the plan must still name a first
        // choice, and a 3rd call is a gamble. Take the largest-budget
        // provider: fail ordinals are drawn from [2, budget-1], so a wide
        // budget means a sparse fail schedule and the safest over-cap call.
        pick = [...candidates].sort((a, b) => {
          const ca = CALLS_MIN_PREFERENCE[a.provider] ?? 0;
          const cb = CALLS_MIN_PREFERENCE[b.provider] ?? 0;
          if (ca !== cb) return cb - ca;
          const na = (callsMade.get(a.provider) ?? 0) + (planned.get(a.provider) ?? 0);
          const nb = (callsMade.get(b.provider) ?? 0) + (planned.get(b.provider) ?? 0);
          if (na !== nb) return na - nb;
          if (a.rating !== b.rating) return b.rating - a.rating;
          return a.provider < b.provider ? -1 : a.provider > b.provider ? 1 : 0;
        })[0];
      } else {
        pick = candidates[0];
      }
    }
    plan.set(request.id, pick);
    planned.set(pick.provider, (planned.get(pick.provider) ?? 0) + 1);
  }
  return plan;
}

export interface AttemptInput {
  /** This request's planned candidate, if the plan holds one and it is alive. */
  planned: Candidate | undefined;
  candidates: Candidate[];
  callsMade: ReadonlyMap<string, number>;
  dead: ReadonlySet<string>;
  variant: Variant;
  /**
   * When true, dead providers are appended as a final tier, best rating
   * first: a provider that refused once is not certainly spent — its next
   * ordinal may well answer — so when no alive provider can serve the
   * request at all, retrying a dead one beats failing the request outright.
   * Dead providers are never tried while any alive candidate remains.
   */
  includeDead?: boolean;
  /**
   * Providers that already failed during this request: never tried again in
   * it, in any tier. A refusal at ordinal k is final for this request.
   */
  skip?: ReadonlySet<string>;
}

/**
 * The attempt order for one request: planned candidate first (if alive), then
 * best-rating alive providers with remaining cap, then best-rating alive
 * providers at any count. Dead providers never appear. Within a tier, fewer
 * calls made sorts first (spread the load), then provider id.
 */
export function attemptOrder(input: AttemptInput): Candidate[] {
  const { planned, candidates, callsMade, dead, variant } = input;
  const skip = input.skip ?? new Set<string>();
  const alive = candidates.filter((c) => !dead.has(c.provider) && !skip.has(c.provider));
  const byRatingCallsId = (a: Candidate, b: Candidate): number => {
    if (a.rating !== b.rating) return b.rating - a.rating;
    const ca = callsMade.get(a.provider) ?? 0;
    const cb = callsMade.get(b.provider) ?? 0;
    if (ca !== cb) return ca - cb;
    return a.provider < b.provider ? -1 : a.provider > b.provider ? 1 : 0;
  };
  /**
   * Variant "e" orders the any-count tier the way the plan's fallback pick is
   * ordered: an over-cap call is a gamble, and the largest-budget provider is
   * the safest gamble. Every other variant keeps the rating order there.
   */
  const byBudgetCallsRatingId = (a: Candidate, b: Candidate): number => {
    const ca = CALLS_MIN_PREFERENCE[a.provider] ?? 0;
    const cb = CALLS_MIN_PREFERENCE[b.provider] ?? 0;
    if (ca !== cb) return cb - ca;
    const na = callsMade.get(a.provider) ?? 0;
    const nb = callsMade.get(b.provider) ?? 0;
    if (na !== nb) return na - nb;
    if (a.rating !== b.rating) return b.rating - a.rating;
    return a.provider < b.provider ? -1 : a.provider > b.provider ? 1 : 0;
  };
  const order: Candidate[] = [];
  const seen = new Set<string>();
  const push = (c: Candidate): void => {
    const key = `${c.provider}/${c.model}`;
    if (seen.has(key)) return;
    seen.add(key);
    order.push(c);
  };
  if (planned !== undefined && !dead.has(planned.provider) && !skip.has(planned.provider)) push(planned);
  const withCap = alive
    .filter((c) => (callsMade.get(c.provider) ?? 0) < capFor(c.provider, variant))
    .sort(byRatingCallsId);
  for (const c of withCap) push(c);
  const anyCount = [...alive].sort(variant === 'e' ? byBudgetCallsRatingId : byRatingCallsId);
  for (const c of anyCount) push(c);
  if (input.includeDead === true) {
    const deadCands = candidates.filter((c) => dead.has(c.provider) && !skip.has(c.provider)).sort(byRatingCallsId);
    for (const c of deadCands) push(c);
  }
  return order;
}

/** The provider directory entry for an id, or null. */
export function providerEntryOf(providers: readonly ProviderEntry[], id: string): ProviderEntry | null {
  return providers.find((p) => p.id === id) ?? null;
}
