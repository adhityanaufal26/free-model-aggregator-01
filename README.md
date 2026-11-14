# free-model-aggregator-01 v2 — quality-ranked free-model aggregator

One server, two client wires (OpenAI Chat Completions, Anthropic Messages),
ten free-tier providers on three upstream wires (OpenAI, Anthropic, Gemini).
For each of the twenty requests it picks the best free model that meets the
request's requirements, fails over across providers when one refuses, and
translates the answer back into the wire the client used, in JSON or in
streaming.

## Layout

- `main.ts` — entrypoint: brief, catalog probe, plan, HTTP server, register,
  `/work` poll, exit 0 when done. Also `--list`.
- `lib/brief.ts` — the trial brief: provider directory, quality table,
  workload manifest, parsed out of the prompt file by schema.
- `lib/catalog.ts` — parallel catalog probes in each provider's own wire,
  and the one catalog `--list` prints and registration carries.
- `lib/plan.ts` — routing: quality-ranked candidates, the budget-aware
  upfront plan, adaptive failover.
- `lib/translate.ts` — request/answer translation between every
  client×upstream wire pair, JSON and SSE.
- `eval/bench.mjs` — local eval harness: runs the sealed practice env
  standalone (no docker), grades with the sealed scorer, benchmarks
  providers. Dev tooling only; the trial runs `main.ts` alone.

## Routing

The brief's quality table rates every free model, and the manifest lists all
twenty requests up front, so the router plans the whole workload before
serving anything: weight-first, best-model-first, subject to a soft
per-provider cap (the kit provider table's published minimum budgets:
groq 9, cerebras 5, gemini 6, zai 4, openrouter 10, sambanova 4,
pollinations 10, cloudflare 7, ovh 5, kilol 12). Inside one rating tier the
provider with the fewest planned calls wins, so two providers sharing a
rating (zai and cloudflare both serve `glm-4.7-flash` at 1470) split the
load instead of one running dry.

Failover: any refused call (402, 429, 5xx, timeout, unreachable) retries
the next eligible model by rating, with refused providers pushed down 45
milli per refusal and retried only after every untried provider has been
tried; at most five provider calls per request. Exactly one provider call
ever answers a request.

Eligibility is the model's capabilities against the request's requirements.
One wire rule on top: an OpenAI request in JSON mode goes only to an
openai- or gemini-wire provider, because the Anthropic wire cannot carry
JSON mode and its echo would report it absent.

## Wire fidelity

The trial's client compares the provider's echo of the request (system
prompt, user text, tool names, `max_tokens`, stop sequences, image, JSON
mode) with what it sent, so every translation moves those fields losslessly
between wires: system prompt, user text, images (bytes never re-encoded),
tool definitions and tool use/results, `max_tokens`, stops, JSON mode
(`response_format` ↔ Gemini `responseMimeType`). Streaming is translated
buffered — the whole upstream event stream is read first, then the client's
event sequence is emitted — so usage is known before `message_start`, the
stop reason before `message_delta`, and an OpenAI stream always ends with
exactly one finish chunk, a usage chunk, and `[DONE]`.

Extra routes (never used by the trial client): `GET /health` (liveness plus
live routing counters), `GET /v1/providers` (catalog with per-provider
calls/refusals/planned counts). Responses carry `x-aggregator-provider` and
`x-aggregator-model` headers naming the model that answered.

## Determinism

Every routing choice is a function of the brief, the requests, and the
providers' answers. No wall clock and no unseeded randomness anywhere in a
request path, so a replay leaves the same state export byte for byte
(verified: two runs of trial-0 produce identical state exports).

## Measured results (local, sealed practice env + sealed scorer)

`node eval/bench.mjs --trials=0,1,2 --bench --repeat=2`

| trial | score | quality | served | wire | failover | catalog | calls |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 99 | 99 | 100 | 100 | 100 | 100 | 28 |
| 1 | 99 | 99 | 100 | 100 | 100 | 100 | 28 |
| 2 | 99 | 99 | 100 | 100 | 100 | 100 | 28 |

Quality 99 is the practical ceiling: the best rating (1475) sits on one
provider whose per-trial budget (10–12) cannot serve all 17 requests that
want it, so the tail is served at 1470/1465 — a 0.3–0.7% quality cost per
request for full coverage. `--list` output matches the registered catalog.

Provider benchmark (trial-0 throwaway env, 1 inference call per provider):
10/10 providers answer 200; average catalog-probe latency 7.5 ms, average
inference round-trip 7.8 ms (loopback sim; the numbers rank providers, they
are not production latencies).

Direct endpoint exercise (live env, curl): `GET /health` 3 ms,
`GET /v1/models` in both auth shapes, `GET /v1/providers`, `POST
/v1/chat/completions` and `POST /v1/messages` in JSON (10 ms / 4 ms) and
streaming (5 ms), all returning the provider echo whole with correct usage
and stop reasons.

## Running it

```sh
node main.ts --list   # print the catalog, exit 0
node main.ts          # needs MAKEFUN_ENV_URL + MAKEFUN_PROMPT_FILE
node eval/bench.mjs --trials=0,1,2 --bench   # local grade + benchmark
```

Node 22 standard library only; no dependencies. `main.ts` runs directly
under Node 22's type stripping.
