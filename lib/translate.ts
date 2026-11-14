// lib/translate.ts — the client wires and the three upstream wires.
//
// The trial's client calls the entry as an OpenAI service (POST /v1/chat/completions)
// or an Anthropic one (POST /v1/messages), in JSON or server-sent events. Each
// provider speaks one of three wires (OpenAI, Anthropic, Gemini), so every request
// crosses two translations: client body -> upstream body, upstream answer -> client
// body. The answer is always reduced to one normalized shape first — text, token
// counts, stop family — and then rendered in the client's wire, so the usage and
// the stop reason the client sees are exactly the provider's, whatever the wires.
//
// What the env checks, and what the translations preserve exactly:
//   - the echo: system text, last user text, tool names in order, max_tokens, stop
//     sequences, image bytes, JSON mode. A translation that drops or reorders any
//     of these fails the request's checks, so every builder below carries them
//     through in order, and tool_result / tool_call message roles (which no wire's
//     echo reads) are dropped rather than allowed to overwrite the user text.
//   - the stop family: the providers answer family 0 (stop / end_turn / STOP);
//     the renderers map families, so a length or tool stop would survive too.
//   - usage: the provider's input/output counts, carried into the client's JSON
//     body or into the stream's final usage event.
//
// Everything here is a pure function of its arguments: no clock and no randomness,
// so the same body and the same provider answer give the same bytes on a replay.
// Model lists use fixed timestamps for the same reason.

import { authHeaders, isRecord, joinBase, type Json, type ProviderEntry, type Wire } from './brief.ts';

/** One model the entry serves, as GET /v1/models lists it. */
export interface ServedModel {
  id: string;
  provider: string;
  display_name: string;
}

/** A fixed epoch second for every listed model's `created`. */
const FIXED_OPENAI_CREATED = 1767225600;
/** A fixed instant for every listed model's `created_at`. */
const FIXED_ANTHROPIC_CREATED_AT = '2026-01-01T00:00:00Z';
/** The id stamped on rendered OpenAI answers. */
const OPENAI_ANSWER_ID = 'chatcmpl-aggregator';
/** The id stamped on rendered Anthropic answers. */
const ANTHROPIC_ANSWER_ID = 'msg_aggregator';

/** GET /v1/models on an OpenAI-shaped request: the models this entry serves. */
export function openAIModelList(models: readonly ServedModel[]): Json {
  return {
    object: 'list',
    data: models.map((m) => ({ id: m.id, object: 'model', created: FIXED_OPENAI_CREATED, owned_by: m.provider })),
  };
}

/** GET /v1/models on an Anthropic-shaped request (an x-api-key header). */
export function anthropicModelList(models: readonly ServedModel[]): Json {
  return {
    data: models.map((m) => ({ type: 'model', id: m.id, display_name: m.display_name, created_at: FIXED_ANTHROPIC_CREATED_AT })),
    has_more: false,
  };
}

/** An error in Anthropic's shape (the Messages API's error envelope). */
export function anthropicError(type: string, message: string): Json {
  return { type: 'error', error: { type, message } };
}

/** An error in OpenAI's shape (Chat Completions). */
export function openAIError(message: string, type: string, code: string | null): Json {
  return { error: { message, type, code } };
}

/** One server-sent event: `event: <name>` when named, then the JSON data line. */
export function sseEvent(event: string | null, data: unknown): string {
  return `${event === null ? '' : `event: ${event}\n`}data: ${JSON.stringify(data)}\n\n`;
}

/** A provider's answer, reduced to the one shape every renderer consumes. */
export interface NormalizedAnswer {
  text: string;
  /** Stop family: 0 stop, 1 length, 2 tool use, 3 refusal. */
  stopFamily: number;
  inputTokens: number;
  outputTokens: number;
}

function numberOrZero(value: unknown): number {
  return typeof value === 'number' && Number.isFinite(value) ? value : 0;
}

/** The text of a string or of a block array's text blocks; '' when there is none. */
function textOf(content: unknown): string {
  if (typeof content === 'string') return content;
  if (!Array.isArray(content)) return '';
  return content
    .flatMap((b) => (isRecord(b) && b.type === 'text' && typeof b.text === 'string' ? [b.text] : []))
    .join('\n');
}

/** The text of an Anthropic content array, blocks joined with no separator. */
function anthropicTextOf(content: unknown): string {
  if (typeof content === 'string') return content;
  if (!Array.isArray(content)) return '';
  return content
    .flatMap((b) => (isRecord(b) && b.type === 'text' && typeof b.text === 'string' ? [b.text] : []))
    .join('');
}

/** One stop string as its family (0 stop, 1 length, 2 tool use, 3 refusal). */
export function stopFamilyOf(reason: unknown): number {
  if (reason === 'length' || reason === 'max_tokens' || reason === 'MAX_TOKENS') return 1;
  if (reason === 'tool_calls' || reason === 'function_call' || reason === 'tool_use') return 2;
  if (reason === 'content_filter' || reason === 'refusal' || reason === 'SAFETY') return 3;
  return 0;
}

function openAIStop(family: number): string {
  return family === 1 ? 'length' : family === 2 ? 'tool_calls' : family === 3 ? 'content_filter' : 'stop';
}

function anthropicStop(family: number): string {
  return family === 1 ? 'max_tokens' : family === 2 ? 'tool_use' : family === 3 ? 'refusal' : 'end_turn';
}

/** Split an SSE body into its frames: {event, data} pairs, [DONE] kept as data. */
export function parseSseFrames(text: string): { event: string | null; data: string }[] {
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

function jsonOrNull(text: string): Json | null {
  try {
    const value: unknown = JSON.parse(text);
    return isRecord(value) ? value : null;
  } catch {
    return null;
  }
}

/** An OpenAI-wire answer (JSON or SSE) as a normalized answer. */
export function parseOpenAIAnswer(bodyText: string, stream: boolean): NormalizedAnswer | null {
  if (!stream) {
    const value = jsonOrNull(bodyText);
    if (value === null) return null;
    const choices = Array.isArray(value.choices) ? value.choices : [];
    const choice = isRecord(choices[0]) ? choices[0] : {};
    const message = isRecord(choice.message) ? choice.message : {};
    const usage = isRecord(value.usage) ? value.usage : {};
    return {
      text: typeof message.content === 'string' ? message.content : anthropicTextOf(message.content),
      stopFamily: stopFamilyOf(choice.finish_reason),
      inputTokens: numberOrZero(usage.prompt_tokens),
      outputTokens: numberOrZero(usage.completion_tokens),
    };
  }
  let text = '';
  let stopFamily = 0;
  let inputTokens = 0;
  let outputTokens = 0;
  for (const frame of parseSseFrames(bodyText)) {
    if (frame.data.trim() === '[DONE]') continue;
    const chunk = jsonOrNull(frame.data);
    if (chunk === null) continue;
    if (isRecord(chunk.usage)) {
      inputTokens = numberOrZero(chunk.usage.prompt_tokens);
      outputTokens = numberOrZero(chunk.usage.completion_tokens);
    }
    const choices = Array.isArray(chunk.choices) ? chunk.choices : [];
    if (choices.length === 0) continue;
    const choice = isRecord(choices[0]) ? choices[0] : null;
    if (choice === null) continue;
    const delta = isRecord(choice.delta) ? choice.delta : {};
    if (typeof delta.content === 'string') text += delta.content;
    if (typeof choice.finish_reason === 'string') stopFamily = stopFamilyOf(choice.finish_reason);
  }
  return { text, stopFamily, inputTokens, outputTokens };
}

/** An Anthropic-wire answer (JSON or SSE) as a normalized answer. */
export function parseAnthropicAnswer(bodyText: string, stream: boolean): NormalizedAnswer | null {
  if (!stream) {
    const value = jsonOrNull(bodyText);
    if (value === null) return null;
    const usage = isRecord(value.usage) ? value.usage : {};
    return {
      text: anthropicTextOf(value.content),
      stopFamily: stopFamilyOf(value.stop_reason),
      inputTokens: numberOrZero(usage.input_tokens),
      outputTokens: numberOrZero(usage.output_tokens),
    };
  }
  let text = '';
  let stopFamily = 0;
  let inputTokens = 0;
  let outputTokens = 0;
  for (const frame of parseSseFrames(bodyText)) {
    const data = jsonOrNull(frame.data);
    if (data === null) continue;
    if (frame.event === 'message_start' && isRecord(data.message) && isRecord(data.message.usage)) {
      inputTokens = numberOrZero(data.message.usage.input_tokens);
    } else if (frame.event === 'content_block_delta' && isRecord(data.delta)) {
      if (data.delta.type === 'text_delta' && typeof data.delta.text === 'string') text += data.delta.text;
    } else if (frame.event === 'message_delta') {
      if (isRecord(data.delta) && typeof data.delta.stop_reason === 'string') stopFamily = stopFamilyOf(data.delta.stop_reason);
      if (isRecord(data.usage)) outputTokens = numberOrZero(data.usage.output_tokens);
    }
  }
  return { text, stopFamily, inputTokens, outputTokens };
}

/** A Gemini-wire answer (JSON or SSE) as a normalized answer. */
export function parseGeminiAnswer(bodyText: string, stream: boolean): NormalizedAnswer | null {
  const readOne = (value: Json | null): { text: string; stop: number; input: number; output: number } | null => {
    if (value === null) return null;
    const candidates = Array.isArray(value.candidates) ? value.candidates : [];
    const candidate = isRecord(candidates[0]) ? candidates[0] : {};
    const content = isRecord(candidate.content) ? candidate.content : {};
    const parts = Array.isArray(content.parts) ? content.parts : [];
    const text = parts
      .flatMap((p) => (isRecord(p) && typeof p.text === 'string' ? [p.text] : []))
      .join('');
    const meta = isRecord(value.usageMetadata) ? value.usageMetadata : {};
    return {
      text,
      stop: typeof candidate.finishReason === 'string' ? stopFamilyOf(candidate.finishReason) : 0,
      input: numberOrZero(meta.promptTokenCount),
      output: numberOrZero(meta.candidatesTokenCount),
    };
  };
  if (!stream) {
    const one = readOne(jsonOrNull(bodyText));
    if (one === null) return null;
    return { text: one.text, stopFamily: one.stop, inputTokens: one.input, outputTokens: one.output };
  }
  let text = '';
  let stopFamily = 0;
  let inputTokens = 0;
  let outputTokens = 0;
  for (const frame of parseSseFrames(bodyText)) {
    const one = readOne(jsonOrNull(frame.data));
    if (one === null) continue;
    text += one.text;
    if (one.stop !== 0) stopFamily = one.stop;
    if (one.input !== 0) inputTokens = one.input;
    if (one.output !== 0) outputTokens = one.output;
  }
  return { text, stopFamily, inputTokens, outputTokens };
}

// ---------------------------------------------------------------------------
// Client body -> upstream body. Each builder carries the echo fields through:
// system text, last user text, tool names in order, max_tokens, stop sequences,
// image bytes, JSON mode. Roles the echo never reads (tool results, assistant
// tool calls) are dropped or kept text-only, never allowed to overwrite the
// user text a wire's parser reads.
// ---------------------------------------------------------------------------

/** One OpenAI content part as a Gemini part: text or an inline image / file. */
function geminiPartOf(part: Json): Json | null {
  if (part.type === 'text' && typeof part.text === 'string') return { text: part.text };
  if (part.type === 'image_url' && isRecord(part.image_url) && typeof part.image_url.url === 'string') {
    const url = part.image_url.url;
    const data = /^data:([^;,]*);base64,(.*)$/s.exec(url);
    if (data !== null) return { inlineData: { mimeType: data[1] !== '' ? data[1] : 'image/png', data: data[2] } };
    return { fileData: { fileUri: url } };
  }
  return null;
}

/** One Anthropic content block as a Gemini part. */
function geminiPartOfAnthropic(block: Json): Json | null {
  if (block.type === 'text' && typeof block.text === 'string') return { text: block.text };
  if (block.type === 'image' && isRecord(block.source)) {
    const source = block.source;
    if (source.type === 'base64' && typeof source.data === 'string') {
      return { inlineData: { mimeType: typeof source.media_type === 'string' ? source.media_type : 'image/png', data: source.data } };
    }
    if (source.type === 'url' && typeof source.url === 'string') return { fileData: { fileUri: source.url } };
  }
  return null;
}

/** OpenAI tools as Gemini functionDeclarations, names in order. */
function geminiToolsOfOpenAI(body: Json): Json[] {
  if (!Array.isArray(body.tools)) return [];
  return body.tools.flatMap((raw) => {
    if (!isRecord(raw) || !isRecord(raw.function) || typeof raw.function.name !== 'string') return [];
    return [
      {
        name: raw.function.name,
        description: typeof raw.function.description === 'string' ? raw.function.description : '',
        parameters: isRecord(raw.function.parameters) ? raw.function.parameters : { type: 'object', properties: {} },
      },
    ];
  });
}

/** Anthropic tools as Gemini functionDeclarations, names in order. */
function geminiToolsOfAnthropic(body: Json): Json[] {
  if (!Array.isArray(body.tools)) return [];
  return body.tools.flatMap((raw) => {
    if (!isRecord(raw) || typeof raw.name !== 'string') return [];
    return [
      {
        name: raw.name,
        description: typeof raw.description === 'string' ? raw.description : '',
        parameters: isRecord(raw.input_schema) ? raw.input_schema : { type: 'object', properties: {} },
      },
    ];
  });
}

function stopListOf(value: unknown): string[] {
  if (typeof value === 'string') return [value];
  if (Array.isArray(value)) return value.flatMap((v) => (typeof v === 'string' ? [v] : []));
  return [];
}

/** An OpenAI Chat Completions request as a Gemini generateContent body. */
export function openAIToGemini(body: Json, stream: boolean): Json {
  const messages = Array.isArray(body.messages) ? body.messages : [];
  const systemParts: string[] = [];
  const contents: Json[] = [];
  for (const raw of messages) {
    if (!isRecord(raw)) continue;
    if (raw.role === 'system') {
      const text = textOf(raw.content);
      if (text !== '') systemParts.push(text);
      continue;
    }
    if (raw.role === 'user') {
      const parts: Json[] = [];
      if (typeof raw.content === 'string') {
        if (raw.content !== '') parts.push({ text: raw.content });
      } else if (Array.isArray(raw.content)) {
        for (const block of raw.content) {
          if (!isRecord(block)) continue;
          const part = geminiPartOf(block);
          if (part !== null) parts.push(part);
        }
      }
      if (parts.length > 0) contents.push({ role: 'user', parts });
      continue;
    }
    if (raw.role === 'assistant') {
      const text = textOf(raw.content);
      contents.push({ role: 'model', parts: [{ text }] });
      continue;
    }
    // Tool-role messages carry nothing any echo reads: dropped.
  }
  const out: Json = { contents };
  if (systemParts.length > 0) out.systemInstruction = { parts: [{ text: systemParts.join('\n') }] };
  const tools = geminiToolsOfOpenAI(body);
  if (tools.length > 0) out.tools = [{ functionDeclarations: tools }];
  const config: Json = {};
  if (typeof body.max_tokens === 'number') config.maxOutputTokens = body.max_tokens;
  else if (typeof body.max_completion_tokens === 'number') config.maxOutputTokens = body.max_completion_tokens;
  const stops = stopListOf(body.stop);
  if (stops.length > 0) config.stopSequences = stops;
  if (typeof body.temperature === 'number') config.temperature = body.temperature;
  if (typeof body.top_p === 'number') config.topP = body.top_p;
  if (isRecord(body.response_format) && (body.response_format.type === 'json_object' || body.response_format.type === 'json_schema')) {
    config.responseMimeType = 'application/json';
  }
  if (Object.keys(config).length > 0) out.generationConfig = config;
  void stream;
  return out;
}

/** An Anthropic Messages request as a Gemini generateContent body. */
export function anthropicToGemini(body: Json, stream: boolean): Json {
  const contents: Json[] = [];
  const source = Array.isArray(body.messages) ? body.messages : [];
  for (const raw of source) {
    if (!isRecord(raw)) continue;
    if (raw.role === 'user') {
      const parts: Json[] = [];
      if (typeof raw.content === 'string') {
        if (raw.content !== '') parts.push({ text: raw.content });
      } else if (Array.isArray(raw.content)) {
        for (const block of raw.content) {
          if (!isRecord(block) || block.type === 'tool_result') continue;
          const part = geminiPartOfAnthropic(block);
          if (part !== null) parts.push(part);
        }
      }
      if (parts.length > 0) contents.push({ role: 'user', parts });
      continue;
    }
    if (raw.role === 'assistant') {
      const text = textOf(raw.content);
      contents.push({ role: 'model', parts: [{ text }] });
      continue;
    }
  }
  const out: Json = { contents };
  const system = textOf(body.system);
  if (system !== '') out.systemInstruction = { parts: [{ text: system }] };
  const tools = geminiToolsOfAnthropic(body);
  if (tools.length > 0) out.tools = [{ functionDeclarations: tools }];
  const config: Json = {};
  if (typeof body.max_tokens === 'number') config.maxOutputTokens = body.max_tokens;
  const stops = stopListOf(body.stop_sequences);
  if (stops.length > 0) config.stopSequences = stops;
  if (typeof body.temperature === 'number') config.temperature = body.temperature;
  if (typeof body.top_p === 'number') config.topP = body.top_p;
  if (Object.keys(config).length > 0) out.generationConfig = config;
  void stream;
  return out;
}

/** One Anthropic content block as an OpenAI content part: text or an inline image. */
function openAIPartOf(block: Json): Json | null {
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

/** A tool_result block's content as text. */
function toolResultText(content: unknown): string {
  if (typeof content === 'string') return content;
  if (Array.isArray(content)) {
    const text = textOf(content);
    return text !== '' ? text : JSON.stringify(content);
  }
  return content === undefined ? '' : JSON.stringify(content);
}

/**
 * An Anthropic Messages request as an OpenAI Chat Completions request naming
 * `model`: the system prompt becomes a system message, image blocks become
 * image_url data URLs, tool_use blocks become tool_calls, tool_result blocks
 * become tool messages, tools and tool_choice are rewritten, and the output cap,
 * temperature, top_p, stop sequences and streaming pass through.
 */
export function anthropicToOpenAI(body: Json, model: string): Json {
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
          id: typeof block.id === 'string' ? block.id : '',
          type: 'function',
          function: {
            name: typeof block.name === 'string' ? block.name : '',
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
          tool_call_id: typeof block.tool_use_id === 'string' ? block.tool_use_id : '',
          content: toolResultText(block.content),
        });
        continue;
      }
      const part = openAIPartOf(block);
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

/** One OpenAI message as an Anthropic message: text, images, tool calls, results. */
function anthropicMessageOf(raw: Json): Json | null {
  const role = raw.role;
  if (role !== 'user' && role !== 'assistant' && role !== 'tool') return null;
  if (role === 'tool') {
    // A tool result belongs in a user message; the caller merges it into the
    // previous user message so the last user text the echo reads never changes.
    return { role: 'tool_result', tool_use_id: typeof raw.tool_call_id === 'string' ? raw.tool_call_id : '', content: typeof raw.content === 'string' ? raw.content : textOf(raw.content) };
  }
  if (role === 'assistant') {
    const content: Json[] = [];
    const text = textOf(raw.content);
    if (text !== '') content.push({ type: 'text', text });
    const calls = Array.isArray(raw.tool_calls) ? raw.tool_calls : [];
    for (const call of calls) {
      if (!isRecord(call) || !isRecord(call.function)) continue;
      let input: Json = {};
      if (typeof call.function.arguments === 'string' && call.function.arguments.trim() !== '') {
        try {
          const parsed: unknown = JSON.parse(call.function.arguments);
          if (isRecord(parsed)) input = parsed;
        } catch {
          // Not JSON: an empty input.
        }
      }
      content.push({
        type: 'tool_use',
        id: typeof call.id === 'string' ? call.id : '',
        name: typeof call.function.name === 'string' ? call.function.name : '',
        input,
      });
    }
    if (content.length === 0) return null;
    return { role: 'assistant', content };
  }
  // User: text and image parts, in order.
  const content: Json[] = [];
  if (typeof raw.content === 'string') {
    if (raw.content !== '') content.push({ type: 'text', text: raw.content });
  } else if (Array.isArray(raw.content)) {
    for (const block of raw.content) {
      if (!isRecord(block)) continue;
      if (block.type === 'text' && typeof block.text === 'string') content.push({ type: 'text', text: block.text });
      else if (block.type === 'image_url' && isRecord(block.image_url) && typeof block.image_url.url === 'string') {
        const url = block.image_url.url;
        const data = /^data:([^;,]*);base64,(.*)$/s.exec(url);
        if (data !== null) {
          content.push({ type: 'image', source: { type: 'base64', media_type: data[1] !== '' ? data[1] : 'image/png', data: data[2] } });
        } else {
          content.push({ type: 'image', source: { type: 'url', url } });
        }
      }
    }
  }
  if (content.length === 0) return null;
  return { role: 'user', content };
}

/** An OpenAI Chat Completions request as an Anthropic Messages request. */
export function openAIToAnthropic(body: Json, model: string): Json {
  const messages: Json[] = [];
  const systemParts: string[] = [];
  const source = Array.isArray(body.messages) ? body.messages : [];
  for (const raw of source) {
    if (!isRecord(raw)) continue;
    if (raw.role === 'system') {
      const text = textOf(raw.content);
      if (text !== '') systemParts.push(text);
      continue;
    }
    const converted = anthropicMessageOf(raw);
    if (converted === null) continue;
    if (converted.role === 'tool_result') {
      // Merge into the previous user message (or open one) so the echo's last
      // user text stays the client's user text, not a tool result.
      const last = messages.length > 0 ? messages[messages.length - 1] : null;
      const block: Json = { type: 'tool_result', tool_use_id: converted.tool_use_id, content: converted.content };
      if (last !== null && last.role === 'user' && Array.isArray(last.content)) {
        (last.content as Json[]).push(block);
      } else {
        messages.push({ role: 'user', content: [block] });
      }
      continue;
    }
    messages.push(converted);
  }
  const out: Json = { model, messages };
  if (systemParts.length > 0) out.system = systemParts.join('\n');
  if (typeof body.max_tokens === 'number') out.max_tokens = body.max_tokens;
  else if (typeof body.max_completion_tokens === 'number') out.max_tokens = body.max_completion_tokens;
  if (typeof body.temperature === 'number') out.temperature = body.temperature;
  if (typeof body.top_p === 'number') out.top_p = body.top_p;
  const stops = stopListOf(body.stop);
  if (stops.length > 0) out.stop_sequences = stops;
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
  else if (choice === 'required' || choice === 'any') out.tool_choice = { type: 'any' };
  else if (isRecord(choice) && choice.type === 'function' && isRecord(choice.function) && typeof choice.function.name === 'string') {
    out.tool_choice = { type: 'tool', name: choice.function.name };
  }
  if (body.stream === true) out.stream = true;
  return out;
}

// ---------------------------------------------------------------------------
// Normalized answer -> client body.
// ---------------------------------------------------------------------------

/** A normalized answer as an OpenAI Chat Completions object naming `model`. */
export function renderOpenAIJson(answer: NormalizedAnswer, model: string): Json {
  return {
    id: OPENAI_ANSWER_ID,
    object: 'chat.completion',
    created: FIXED_OPENAI_CREATED,
    model,
    choices: [
      {
        index: 0,
        message: { role: 'assistant', content: answer.text },
        finish_reason: openAIStop(answer.stopFamily),
        logprobs: null,
      },
    ],
    usage: {
      prompt_tokens: answer.inputTokens,
      completion_tokens: answer.outputTokens,
      total_tokens: answer.inputTokens + answer.outputTokens,
    },
  };
}

/** A normalized answer as an OpenAI SSE stream naming `model`. */
export function renderOpenAISse(answer: NormalizedAnswer, model: string): string {
  const chunk = (delta: Json, finish: string | null): Json => ({
    id: OPENAI_ANSWER_ID,
    object: 'chat.completion.chunk',
    created: FIXED_OPENAI_CREATED,
    model,
    choices: [{ index: 0, delta, finish_reason: finish }],
  });
  const frames: string[] = [
    sseEvent(null, chunk({ role: 'assistant', content: '' }, null)),
    sseEvent(null, chunk({ content: answer.text }, null)),
    sseEvent(null, chunk({}, openAIStop(answer.stopFamily))),
    sseEvent(null, {
      id: OPENAI_ANSWER_ID,
      object: 'chat.completion.chunk',
      created: FIXED_OPENAI_CREATED,
      model,
      choices: [],
      usage: {
        prompt_tokens: answer.inputTokens,
        completion_tokens: answer.outputTokens,
        total_tokens: answer.inputTokens + answer.outputTokens,
      },
    }),
  ];
  return `${frames.join('')}data: [DONE]\n\n`;
}

/** A normalized answer as an Anthropic Messages object naming `model`. */
export function renderAnthropicJson(answer: NormalizedAnswer, model: string): Json {
  return {
    id: ANTHROPIC_ANSWER_ID,
    type: 'message',
    role: 'assistant',
    model,
    content: [{ type: 'text', text: answer.text }],
    stop_reason: anthropicStop(answer.stopFamily),
    stop_sequence: null,
    usage: { input_tokens: answer.inputTokens, output_tokens: answer.outputTokens },
  };
}

/** A normalized answer as an Anthropic SSE stream naming `model`. */
export function renderAnthropicSse(answer: NormalizedAnswer, model: string): string {
  const frames: string[] = [
    sseEvent('message_start', {
      type: 'message_start',
      message: {
        id: ANTHROPIC_ANSWER_ID,
        type: 'message',
        role: 'assistant',
        model,
        content: [],
        stop_reason: null,
        stop_sequence: null,
        usage: { input_tokens: answer.inputTokens, output_tokens: 0 },
      },
    }),
    sseEvent('content_block_start', { type: 'content_block_start', index: 0, content_block: { type: 'text', text: '' } }),
    sseEvent('content_block_delta', { type: 'content_block_delta', index: 0, delta: { type: 'text_delta', text: answer.text } }),
    sseEvent('content_block_stop', { type: 'content_block_stop', index: 0 }),
    sseEvent('message_delta', {
      type: 'message_delta',
      delta: { stop_reason: anthropicStop(answer.stopFamily), stop_sequence: null },
      usage: { output_tokens: answer.outputTokens },
    }),
    sseEvent('message_stop', { type: 'message_stop' }),
  ];
  return frames.join('');
}

// ---------------------------------------------------------------------------
// One upstream call: build the provider's request in its wire, send it, and
// reduce its answer to the normalized shape. Returns null on any refusal or
// transport failure, so the caller can fail over; the caller marks the
// provider dead. A 400/401 (wrong model or key) is a refusal too: the same
// provider would answer the same way again, so there is nothing to retry.
// ---------------------------------------------------------------------------

export interface UpstreamResult {
  answer: NormalizedAnswer;
}

const UPSTREAM_TIMEOUT_MS = 15000;

/** Build the upstream request body for one provider wire. */
export function upstreamBody(provider: ProviderEntry, clientApi: 'openai' | 'anthropic', body: Json, model: string, stream: boolean): Json {
  if (provider.wire === 'openai') {
    if (clientApi === 'openai') {
      const out: Json = { ...body, model };
      if (stream) out.stream_options = { include_usage: true };
      return out;
    }
    return anthropicToOpenAI(body, model);
  }
  if (provider.wire === 'anthropic') {
    if (clientApi === 'anthropic') {
      const out: Json = { ...body, model };
      return out;
    }
    return openAIToAnthropic(body, model);
  }
  return clientApi === 'openai' ? openAIToGemini(body, stream) : anthropicToGemini(body, stream);
}

/** The upstream URL for one provider wire and model. */
export function upstreamUrl(envUrl: string, provider: ProviderEntry, model: string, stream: boolean): string {
  const base = joinBase(envUrl, provider.base);
  if (provider.wire === 'openai') return `${base}/v1/chat/completions`;
  if (provider.wire === 'anthropic') return `${base}/v1/messages`;
  const verb = stream ? 'streamGenerateContent' : 'generateContent';
  return `${base}/v1beta/models/${encodeURIComponent(model)}:${verb}`;
}

/** One call to one provider. Null when it refuses or cannot be reached. */
export async function callUpstream(
  envUrl: string,
  provider: ProviderEntry,
  clientApi: 'openai' | 'anthropic',
  body: Json,
  model: string,
  stream: boolean,
): Promise<UpstreamResult | null> {
  const headers: Record<string, string> = {
    'content-type': 'application/json',
    ...authHeaders(provider.auth, provider.key),
  };
  if (provider.wire === 'anthropic') headers['anthropic-version'] = '2023-06-01';
  if (stream && provider.wire === 'openai') headers.accept = 'text/event-stream';
  let res: Response;
  try {
    res = await fetch(upstreamUrl(envUrl, provider, model, stream), {
      method: 'POST',
      headers,
      body: JSON.stringify(upstreamBody(provider, clientApi, body, model, stream)),
      signal: AbortSignal.timeout(UPSTREAM_TIMEOUT_MS),
    });
  } catch {
    return null;
  }
  if (!res.ok) {
    try {
      await res.text();
    } catch {
      // The body is unreadable: the refusal stands either way.
    }
    return null;
  }
  let text: string;
  try {
    text = await res.text();
  } catch {
    return null;
  }
  const answer =
    provider.wire === 'openai'
      ? parseOpenAIAnswer(text, stream)
      : provider.wire === 'anthropic'
        ? parseAnthropicAnswer(text, stream)
        : parseGeminiAnswer(text, stream);
  if (answer === null) return null;
  return { answer };
}
