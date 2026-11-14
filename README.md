# free-model-aggregator-01

One server that serves the OpenAI Chat Completions API and the Anthropic
Messages API (JSON and SSE) by routing each request to the best free model
that can serve it, across ten simulated free-tier providers that speak the
real OpenAI, Anthropic and Gemini wires — with failover when a provider
refuses, and a `--list` of the catalog it read.

## Run

Node 22, standard library only. Nothing is installed.

```sh
node main.ts            # trial mode: reads $MAKEFUN_PROMPT_FILE, probes the
                        # providers under $MAKEFUN_ENV_URL, registers, serves
                        # the 20 requests, exits 0 when GET /work says done
node main.ts --list     # print the catalog document and exit 0
```

Outside a trial (no `MAKEFUN_PROMPT_FILE` and no `MAKEFUN_ENV_URL`),
`--list` prints `{"providers":[]}`.

## How it routes

- At startup it probes every provider's catalog route in that provider's own
  wire and auth scheme (`Authorization: Bearer`, `x-api-key`,
  `x-goog-api-key`, or none) and builds one catalog: provider id, base URL
  (the provider's own path, e.g. `/p/groq`), key variable, wire, and each
  model with its free flag and context length. The catalog `--list` prints
  is the same document the register body carries, byte for byte.
- For each request it ranks every free model that meets the request's
  requirements (tools, vision, JSON mode, minimum context) by the quality
  table's rating, highest first, spreading ties across the least-used
  providers. A paid model is never picked.
- A provider that answers 402, 429, 5xx, or is unreachable is failed over
  at once: the request is retried on the next-best free model that meets
  it. A provider that fails twice in a row is treated as spent for the rest
  of the trial (and retried only if nothing else can serve a request).
  A 400/401 spends no budget and is skipped without a strike.
- An OpenAI JSON-mode request is never routed to an Anthropic-wire
  provider: that wire cannot carry `response_format`, so its echo would
  report JSON mode absent. Gemini carries it as
  `generationConfig.responseMimeType`.

## How it translates

Every provider answer is reduced to one normalized shape — text, tool
calls, usage, stop family — then rendered in the wire the client used, so
all six combinations (OpenAI/Anthropic client × OpenAI/Anthropic/Gemini
provider) share one path, JSON and streaming alike:

- OpenAI client → OpenAI provider: passthrough (model replaced,
  `stream_options.include_usage` added on streams so usage is reported).
- Anthropic client → OpenAI provider: system prompt, image blocks, tool
  definitions/results and tool choice rewritten to Chat Completions, and
  the answer rewritten back to a Messages object or the
  `message_start` / `content_block_*` / `message_delta` / `message_stop`
  event sequence, with usage and stop reason mapped.
- Either client → Gemini provider: `generateContent` /
  `streamGenerateContent?alt=sse` with `systemInstruction`, `contents`,
  `functionDeclarations` and `generationConfig` (`maxOutputTokens`,
  `stopSequences`, `responseMimeType` for JSON mode); the answer's
  candidates and `usageMetadata` mapped back to the client's wire.
- The reverse directions are symmetric. System prompt, user text (images
  included, as the same bytes, so the image hash matches), tool names in
  order, `max_tokens`, stop sequences and JSON mode are passed through
  untouched on every wire: the provider echoes what it parsed, and the
  client compares that echo with what it sent.

## Determinism

No wall clock, no unseeded randomness. Fixed timestamps in model lists and
responses, fixed response ids, one request at a time in arrival order, and
every routing choice a function of the brief, the requests and the
providers' answers — so a replayed trial leaves the same state export,
byte for byte (verified: two runs of practice trial-0 produce identical
`state.json`).

## Local grading

Practice trials from the kit (`grade-local` needs Docker; the equivalent
manual run does not):

```sh
KIT=~/workspace/makefun/free-model-aggregator-01-kit
# trial 0: start the env, run the server against it, export, score
MAKEFUN_STATE_DIR=/tmp/fma-state0 \
  MAKEFUN_ENV_DATA_DIR=$KIT/practice/trial-0/input/env \
  MAKEFUN_ENV_PORT=18080 node $KIT/program/env/server.mjs &
MAKEFUN_ENV_URL=http://127.0.0.1:18080 \
  MAKEFUN_PROMPT_FILE=$KIT/practice/trial-0/input/prompt.txt \
  node main.ts
kill -TERM <env pid>   # env writes /tmp/fma-state0/export/state.json
MAKEFUN_OUTPUT_DIR=/tmp/fma-state0/export \
  MAKEFUN_REFERENCE_DIR=$KIT/practice/trial-0/reference \
  MAKEFUN_SCORES_DIR=/tmp/fma-scores0 \
  node $KIT/program/scorer/score.mjs
```

Results on the kit's practice instances (20 requests each):

| trial | valid | score | quality | served | wire | failover | catalog | upstream calls |
|-------|-------|-------|---------|--------|------|----------|---------|----------------|
| 0 | 20/20 | 99 | 99 | 100 | 100 | 100 | 100 | 26 |
| 1 | 20/20 | 99 | 99 | 100 | 100 | 100 | 100 | 30 |
| 2 | 20/20 | 99 | 99 | 100 | 100 | 100 | 100 | 29 |

Quality is 99 rather than 100 because the top-rated provider's free tier
runs dry partway through a trial (e.g. OpenRouter's budget), so the last
requests fall back to the next-best free model (1470 vs 1475) — the
failover that keeps them served.

## Layout

- `main.ts` — the whole server: brief parsing, catalog probe, routing and
  failover, all six wire translations, the HTTP server, register/poll.
- `package.json` — `{"type": "module"}` so Node 22 runs `main.ts` as ESM
  without a reparsing warning.
- `LICENSE` — MIT.

## License

MIT. See `LICENSE`.
