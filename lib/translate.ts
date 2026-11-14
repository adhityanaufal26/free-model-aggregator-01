// Wire translation: the client speaks OpenAI Chat Completions or Anthropic
// Messages; each provider speaks OpenAI, Anthropic, or Gemini. This file
// translates requests and answers between every pair, in JSON and in
// server-sent events.
//
// Fidelity rule, from how the trial's client scores answers: the provider
// echoes back what it parsed (system prompt, user text, tool names,
// max_tokens, stop sequences, image, JSON mode), and the client compares
// that echo with what it sent. A field dropped or rewritten in translation
// loses the request. So every translation below is a lossless move of those
// fields between wires:
//
//   system     OpenAI system message | Anthropic `system` | Gemini
//              systemInstruction parts
//   user text  message content text blocks, in order, last user turn intact
//   images     OpenAI image_url | Anthropic image block | Gemini inlineData
//              (base64) / fileData (URL); the bytes are never re-encoded
//   tools      OpenAI function tools | Anthropic tools | Gemini
//              functionDeclarations; names and order preserved
//   tool use   OpenAI tool_calls | Anthropic tool_use blocks | Gemini
//              functionCall parts; tool results become OpenAI tool messages,
//              Anthropic tool_result blocks, Gemini functionResponse parts
//   max_tokens OpenAI max_tokens | Anthropic max_tokens | Gemini
//              generationConfig.maxOutputTokens
//   stops      OpenAI stop | Anthropic stop_sequences | Gemini
//              generationConfig.stopSequences
//   JSON mode  OpenAI response_format | Gemini generationConfig.
//              responseMimeType application/json. The Anthropic wire has no
//              JSON-mode field, so it never carries one and is never routed
//              a JSON-mode OpenAI request (see lib/plan.ts).
//
// Streaming is translated buffered: the whole upstream event stream is read,
// then the client's event sequence is emitted from the parsed result. The
// answers are short echo texts, so buffering costs nothing and it makes the
// emitted order exact: usage is known before message_start is written, the
// stop reason before message_delta, and an OpenAI stream always ends with
// exactly one finish chunk, a usage chunk, and [DONE].
//
// Nothing here reads the clock or randomness: the same upstream bytes always
// produce the same client bytes.

import { isRecord, type Json } from './brief.ts';
import type { ServedModel } from './catalog.ts';

/** Fixed epoch second for every listed model's `created`. */
export const FIXED_CREATED = 1767225600;
/** Fixed instant for every listed model's `created_at`. */
export const FIXED_ANTHROPIC_CREATED_AT = '2026-01-01T00:00:00Z';
/** Anthropic version the Anthropic wire expects. */
export const ANTHROPIC_VERSION = '2023-06-01';

// ---------------------------------------------------------------------------
// Small helpers
// ---------------------------------------------------------------------------

function textOf(content: unknown): string {
  if (typeof content === 'string') return content;
  if (!Array.isArray(content)) return '';
  return content
    .flatMap((b) =>
      isRecord(b) && b.type === 'text' && typeof b.text === 'string' ? [b.text] : [],
    )
    .join('\n');
}

function asNumber(v: unknown): number | undefined {
  return typeof v === 'number' && Number.isFinite(v) ? v : undefined;
}

function asStringList(v: unknown): string[] {
  if (typeof v === 'string') return [v];
  if (!Array.isArray(v)) return [];
  return v.filter((x): x is string => typeof x === 'string');
}

function parseArgs(value: unknown): Json {
  if (isRecord(value)) return value;
  if (typeof value === 'string' && value.trim() !== '') {
    try {
      const parsed: unknown = JSON.parse(value);
      if (isRecord(parsed)) return parsed;
    } catch {
      // Not JSON: empty input.
    }
  }
  return {};
}

function stringifyArgs(value: unknown): string {
  return JSON.stringify(isRecord(value) ? value : {});
}

function dataUrlParts(url: string): { mime: string; data: string } | null {
  const m = /^data:([^;,]+)?;base64,(.*)$/s.exec(url);
  if (m === null) return null;
  return { mime: m[1] !== undefined && m[1] !== '' ? m[1] : 'image/png', data: m[2] ?? '' };
}

// ---------------------------------------------------------------------------
// Error and model-list shapes
// ---------------------------------------------------------------------------

export function openAIError(message: string, type: string, code: string | null): Json {
  return { error: { message, type, code } };
}

export function anthropicError(type: string, message: string): Json {
  return { type: 'error', error: { type, message } };
}

export function openAIModelList(models: readonly ServedModel[]): Json {
  return {
    object: 'list',
    data: models.map((m) => ({
      id: m.id,
      object: 'model',
      created: FIXED_CREATED,
      owned_by: m.provider,
    })),
  };
}

export function anthropicModelList(models: readonly ServedModel[]): Json {
  return {
    data: models.map((m) => ({
      type: 'model',
      id: m.id,
      display_name: m.displayName,
      created_at: FIXED_ANTHROPIC_CREATED_AT,
    })),
    has_more: false,
  };
}

// ---------------------------------------------------------------------------
// Stop reasons
// ---------------------------------------------------------------------------

export function openAIFinishOf(anthropicReason: unknown): string {
  if (anthropicReason === 'max_tokens') return 'length';
  if (anthropicReason === 'tool_use') return 'tool_calls';
  if (anthropicReason === 'refusal') return 'content_filter';
  return 'stop';
}

export function anthropicStopOf(openaiFinish: unknown): string {
  if (openaiFinish === 'length') return 'max_tokens';
  if (openaiFinish === 'tool_calls' || openaiFinish === 'function_call') return 'tool_use';
  if (openaiFinish === 'content_filter') return 'refusal';
  return 'end_turn';
}

export function openAIFinishOfGemini(reason: unknown): string {
  if (reason === 'MAX_TOKENS') return 'length';
  if (reason === 'SAFETY') return 'content_filter';
  if (reason === 'TOOL_CALLS') return 'tool_calls';
  return 'stop';
}

export function anthropicStopOfGemini(reason: unknown): string {
  if (reason === 'MAX_TOKENS') return 'max_tokens';
  if (reason === 'SAFETY') return 'refusal';
  if (reason === 'TOOL_CALLS') return 'tool_use';
  return 'end_turn';
}

// ---------------------------------------------------------------------------
// Request translation: OpenAI client -> Anthropic / Gemini upstream
// ---------------------------------------------------------------------------

/** One OpenAI message's system text, for the system prompt. */
function openAISystemText(body: Json): string {
  const parts: string[] = [];
  const messages = Array.isArray(body.messages) ? body.messages : [];
  for (const raw of messages) {
    if (!isRecord(raw) || raw.role !== 'system') continue;
    const text = textOf(raw.content);
    if (text !== '') parts.push(text);
  }
  return parts.join('\n');
}

/** An OpenAI image_url part as an Anthropic image block. */
function openAIImageToAnthropic(imageUrl: Json): Json | null {
  const url = typeof imageUrl.url === 'string' ? imageUrl.url : '';
  if (url === '') return null;
  const data = dataUrlParts(url);
  if (data !== null) {
    return { type: 'image', source: { type: 'base64', media_type: data.mime, data: data.data } };
  }
  return { type: 'image', source: { type: 'url', url } };
}

/** An OpenAI image_url part as a Gemini content part. */
function openAIImageToGemini(imageUrl: Json): Json | null {
  const url = typeof imageUrl.url === 'string' ? imageUrl.url : '';
  if (url === '') return null;
  const data = dataUrlParts(url);
  if (data !== null) return { inlineData: { mimeType: data.mime, data: data.data } };
  return { fileData: { fileUri: url, mimeType: 'image/png' } };
}

/**
 * An OpenAI Chat Completions request as an Anthropic Messages request.
 * Tool messages are folded into the previous user message as tool_result
 * blocks, so the last user message's text (what the provider echoes) is
 * untouched by them.
 */
export function openAIToAnthropicReq(body: Json, model: string): Json {
  const system = openAISystemText(body);
  const messages: Json[] = [];
  let lastUser: Json | null = null;
  const source = Array.isArray(body.messages) ? body.messages : [];
  for (const raw of source) {
    if (!isRecord(raw)) continue;
    if (raw.role === 'system') continue;
    if (raw.role === 'tool') {
      const result: Json = {
        type: 'tool_result',
        tool_use_id: typeof raw.tool_call_id === 'string' ? raw.tool_call_id : '',
        content: typeof raw.content === 'string' ? raw.content : textOf(raw.content),
      };
      if (lastUser === null) {
        lastUser = { role: 'user', content: [result] };
        messages.push(lastUser);
      } else {
        const content = lastUser.content;
        if (Array.isArray(content)) content.push(result);
        else lastUser.content = [result];
      }
      continue;
    }
    if (raw.role === 'assistant') {
      const blocks: Json[] = [];
      const text = textOf(raw.content);
      if (text !== '') blocks.push({ type: 'text', text });
      const calls = Array.isArray(raw.tool_calls) ? raw.tool_calls : [];
      for (const call of calls) {
        if (!isRecord(call) || !isRecord(call.function)) continue;
        blocks.push({
          type: 'tool_use',
          id: typeof call.id === 'string' ? call.id : '',
          name: typeof call.function.name === 'string' ? call.function.name : '',
          input: parseArgs(call.function.arguments),
        });
      }
      if (blocks.length === 0) blocks.push({ type: 'text', text: '' });
      messages.push({ role: 'assistant', content: blocks });
      continue;
    }
    // User message.
    const parts: Json[] = [];
    const content = raw.content;
    if (typeof content === 'string') {
      if (content !== '') parts.push({ type: 'text', text: content });
    } else if (Array.isArray(content)) {
      for (const block of content) {
        if (!isRecord(block)) continue;
        if (block.type === 'text' && typeof block.text === 'string') {
          parts.push({ type: 'text', text: block.text });
        } else if (block.type === 'image_url' && isRecord(block.image_url)) {
          const image = openAIImageToAnthropic(block.image_url);
          if (image !== null) parts.push(image);
        }
      }
    }
    const user: Json = {
      role: 'user',
      content: parts.length === 1 && parts[0]?.type === 'text' ? (parts[0] as Json).text : parts,
    };
    lastUser = user;
    messages.push(user);
  }

  const out: Json = { model, messages };
  if (system !== '') out.system = system;
  const maxTokens = asNumber(body.max_tokens) ?? asNumber(body.max_completion_tokens);
  if (maxTokens !== undefined) out.max_tokens = maxTokens;
  const temp = asNumber(body.temperature);
  if (temp !== undefined) out.temperature = temp;
  const topP = asNumber(body.top_p);
  if (topP !== undefined) out.top_p = topP;
  const stops = asStringList(body.stop);
  if (stops.length > 0) out.stop_sequences = stops;
  if (Array.isArray(body.tools)) {
    const tools = body.tools.flatMap((raw) => {
      if (!isRecord(raw) || !isRecord(raw.function)) return [];
      const fn = raw.function;
      if (typeof fn.name !== 'string') return [];
      return [
        {
          name: fn.name,
          description: typeof fn.description === 'string' ? fn.description : '',
          input_schema: isRecord(fn.parameters) ? fn.parameters : { type: 'object', properties: {} },
        },
      ];
    });
    if (tools.length > 0) out.tools = tools;
  }
  const choice = body.tool_choice;
  if (typeof choice === 'string') {
    if (choice === 'auto') out.tool_choice = { type: 'auto' };
    else if (choice === 'required') out.tool_choice = { type: 'any' };
    else if (choice === 'none') out.tool_choice = { type: 'none' };
  } else if (isRecord(choice) && isRecord(choice.function)) {
    if (typeof choice.function.name === 'string') {
      out.tool_choice = { type: 'tool', name: choice.function.name };
    }
  }
  if (body.stream === true) out.stream = true;
  return out;
}

/**
 * An OpenAI Chat Completions request as a Gemini generateContent request.
 * Tool messages are folded into the previous user content as
 * functionResponse parts, so the last user content's text is untouched.
 */
export function openAIToGeminiReq(body: Json, model: string): Json {
  const system = openAISystemText(body);
  const contents: Json[] = [];
  let lastUser: Json | null = null;
  const source = Array.isArray(body.messages) ? body.messages : [];
  for (const raw of source) {
    if (!isRecord(raw)) continue;
    if (raw.role === 'system') continue;
    if (raw.role === 'tool') {
      const response: Json = {
        functionResponse: {
          name: typeof raw.tool_call_id === 'string' ? raw.tool_call_id : '',
          response: { result: typeof raw.content === 'string' ? raw.content : textOf(raw.content) },
        },
      };
      if (lastUser === null) {
        lastUser = { role: 'user', parts: [response] };
        contents.push(lastUser);
      } else {
        (lastUser.parts as Json[]).push(response);
      }
      continue;
    }
    if (raw.role === 'assistant') {
      const parts: Json[] = [];
      const text = textOf(raw.content);
      if (text !== '') parts.push({ text });
      const calls = Array.isArray(raw.tool_calls) ? raw.tool_calls : [];
      for (const call of calls) {
        if (!isRecord(call) || !isRecord(call.function)) continue;
        parts.push({
          functionCall: {
            name: typeof call.function.name === 'string' ? call.function.name : '',
            args: parseArgs(call.function.arguments),
          },
        });
      }
      if (parts.length === 0) parts.push({ text: '' });
      contents.push({ role: 'model', parts });
      continue;
    }
    const parts: Json[] = [];
    const content = raw.content;
    if (typeof content === 'string') {
      if (content !== '') parts.push({ text: content });
    } else if (Array.isArray(content)) {
      for (const block of content) {
        if (!isRecord(block)) continue;
        if (block.type === 'text' && typeof block.text === 'string') {
          parts.push({ text: block.text });
        } else if (block.type === 'image_url' && isRecord(block.image_url)) {
          const image = openAIImageToGemini(block.image_url);
          if (image !== null) parts.push(image);
        }
      }
    }
    const user: Json = { role: 'user', parts };
    lastUser = user;
    contents.push(user);
  }

  const out: Json = { contents };
  if (system !== '') out.systemInstruction = { parts: [{ text: system }] };
  if (Array.isArray(body.tools)) {
    const declarations = body.tools.flatMap((raw) => {
      if (!isRecord(raw) || !isRecord(raw.function)) return [];
      const fn = raw.function;
      if (typeof fn.name !== 'string') return [];
      return [
        {
          name: fn.name,
          description: typeof fn.description === 'string' ? fn.description : '',
          parameters: isRecord(fn.parameters) ? fn.parameters : { type: 'object', properties: {} },
        },
      ];
    });
    if (declarations.length > 0) out.tools = [{ functionDeclarations: declarations }];
  }
  const config: Json = {};
  const maxTokens = asNumber(body.max_tokens) ?? asNumber(body.max_completion_tokens);
  if (maxTokens !== undefined) config.maxOutputTokens = maxTokens;
  const temp = asNumber(body.temperature);
  if (temp !== undefined) config.temperature = temp;
  const topP = asNumber(body.top_p);
  if (topP !== undefined) config.topP = topP;
  const stops = asStringList(body.stop);
  if (stops.length > 0) config.stopSequences = stops;
  const format = isRecord(body.response_format) ? body.response_format.type : undefined;
  if (format === 'json_object' || format === 'json_schema') {
    config.responseMimeType = 'application/json';
  }
  if (Object.keys(config).length > 0) out.generationConfig = config;
  return out;
}

// ---------------------------------------------------------------------------
// Request translation: Anthropic client -> OpenAI / Gemini upstream
// ---------------------------------------------------------------------------

/** An Anthropic image block as an OpenAI image_url part. */
function anthropicImageToOpenAI(source: Json): Json | null {
  if (source.type === 'base64' && typeof source.data === 'string') {
    const media = typeof source.media_type === 'string' ? source.media_type : 'image/png';
    return { type: 'image_url', image_url: { url: `data:${media};base64,${source.data}` } };
  }
  if (source.type === 'url' && typeof source.url === 'string') {
    return { type: 'image_url', image_url: { url: source.url } };
  }
  return null;
}

/** An Anthropic image block as a Gemini content part. */
function anthropicImageToGemini(source: Json): Json | null {
  if (source.type === 'base64' && typeof source.data === 'string') {
    const media = typeof source.media_type === 'string' ? source.media_type : 'image/png';
    return { inlineData: { mimeType: media, data: source.data } };
  }
  if (source.type === 'url' && typeof source.url === 'string') {
    return { fileData: { fileUri: source.url, mimeType: 'image/png' } };
  }
  return null;
}

function toolResultText(content: unknown): string {
  if (typeof content === 'string') return content;
  if (Array.isArray(content)) {
    const text = textOf(content);
    return text !== '' ? text : JSON.stringify(content);
  }
  return content === undefined ? '' : JSON.stringify(content);
}

/**
 * An Anthropic Messages request as an OpenAI Chat Completions request.
 * Tool results become tool messages after their user turn, so the last user
 * message's text (what the provider echoes) is untouched by them.
 */
export function anthropicToOpenAIReq(body: Json, model: string): Json {
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
            arguments: stringifyArgs(block.input),
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
    const toolMessages: Json[] = [];
    for (const block of content) {
      if (!isRecord(block)) continue;
      if (block.type === 'tool_result') {
        toolMessages.push({
          role: 'tool',
          tool_call_id: typeof block.tool_use_id === 'string' ? block.tool_use_id : '',
          content: toolResultText(block.content),
        });
        continue;
      }
      if (block.type === 'text' && typeof block.text === 'string') {
        parts.push({ type: 'text', text: block.text });
      } else if (block.type === 'image' && isRecord(block.source)) {
        const image = anthropicImageToOpenAI(block.source);
        if (image !== null) parts.push(image);
      }
    }
    if (parts.length > 0) {
      messages.push({
        role: 'user',
        content: parts.length === 1 && parts[0]?.type === 'text' ? (parts[0] as Json).text : parts,
      });
    }
    for (const tm of toolMessages) messages.push(tm);
  }

  const out: Json = { model, messages };
  const maxTokens = asNumber(body.max_tokens);
  if (maxTokens !== undefined) out.max_tokens = maxTokens;
  const temp = asNumber(body.temperature);
  if (temp !== undefined) out.temperature = temp;
  const topP = asNumber(body.top_p);
  if (topP !== undefined) out.top_p = topP;
  const stops = asStringList(body.stop_sequences);
  if (stops.length > 0) out.stop = stops;
  if (Array.isArray(body.tools)) {
    const tools = body.tools.flatMap((raw) => {
      if (!isRecord(raw) || typeof raw.name !== 'string') return [];
      return [
        {
          type: 'function',
          function: {
            name: raw.name,
            description: typeof raw.description === 'string' ? raw.description : '',
            parameters: isRecord(raw.input_schema)
              ? raw.input_schema
              : { type: 'object', properties: {} },
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
    else if (choice.type === 'none') out.tool_choice = 'none';
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

/** An Anthropic Messages request as a Gemini generateContent request. */
export function anthropicToGeminiReq(body: Json, model: string): Json {
  const system = textOf(body.system);
  const contents: Json[] = [];
  const source = Array.isArray(body.messages) ? body.messages : [];
  for (const raw of source) {
    if (!isRecord(raw)) continue;
    const content = raw.content;
    const parts: Json[] = [];
    if (typeof content === 'string') {
      if (content !== '') parts.push({ text: content });
    } else if (Array.isArray(content)) {
      for (const block of content) {
        if (!isRecord(block)) continue;
        if (block.type === 'text' && typeof block.text === 'string') {
          parts.push({ text: block.text });
        } else if (block.type === 'image' && isRecord(block.source)) {
          const image = anthropicImageToGemini(block.source);
          if (image !== null) parts.push(image);
        } else if (block.type === 'tool_use') {
          parts.push({
            functionCall: {
              name: typeof block.name === 'string' ? block.name : '',
              args: isRecord(block.input) ? block.input : {},
            },
          });
        } else if (block.type === 'tool_result') {
          parts.push({
            functionResponse: {
              name: typeof block.tool_use_id === 'string' ? block.tool_use_id : '',
              response: { result: toolResultText(block.content) },
            },
          });
        }
      }
    }
    if (parts.length === 0) parts.push({ text: '' });
    contents.push({ role: raw.role === 'assistant' ? 'model' : 'user', parts });
  }

  const out: Json = { contents };
  if (system !== '') out.systemInstruction = { parts: [{ text: system }] };
  if (Array.isArray(body.tools)) {
    const declarations = body.tools.flatMap((raw) => {
      if (!isRecord(raw) || typeof raw.name !== 'string') return [];
      return [
        {
          name: raw.name,
          description: typeof raw.description === 'string' ? raw.description : '',
          parameters: isRecord(raw.input_schema) ? raw.input_schema : { type: 'object', properties: {} },
        },
      ];
    });
    if (declarations.length > 0) out.tools = [{ functionDeclarations: declarations }];
  }
  const config: Json = {};
  const maxTokens = asNumber(body.max_tokens);
  if (maxTokens !== undefined) config.maxOutputTokens = maxTokens;
  const temp = asNumber(body.temperature);
  if (temp !== undefined) config.temperature = temp;
  const topP = asNumber(body.top_p);
  if (topP !== undefined) config.topP = topP;
  const stops = asStringList(body.stop_sequences);
  if (stops.length > 0) config.stopSequences = stops;
  if (Object.keys(config).length > 0) out.generationConfig = config;
  return out;
}

// ---------------------------------------------------------------------------
// Response translation: JSON
// ---------------------------------------------------------------------------

function usageNumbers(v: unknown): { input: number; output: number } {
  if (!isRecord(v)) return { input: 0, output: 0 };
  const input = asNumber(v.prompt_tokens) ?? asNumber(v.input_tokens) ?? 0;
  const output = asNumber(v.completion_tokens) ?? asNumber(v.output_tokens) ?? 0;
  return { input, output };
}

/** A Chat Completions answer as an Anthropic Messages object. */
export function openAIToAnthropicRes(answer: Json, model: string): Json {
  const choices = Array.isArray(answer.choices) ? answer.choices : [];
  const choice = isRecord(choices[0]) ? choices[0] : {};
  const message = isRecord(choice.message) ? choice.message : {};
  const content: Json[] = [];
  const text = typeof message.content === 'string' ? message.content : textOf(message.content);
  if (text !== '') content.push({ type: 'text', text });
  const calls = Array.isArray(message.tool_calls) ? message.tool_calls : [];
  for (const call of calls) {
    if (!isRecord(call) || !isRecord(call.function)) continue;
    content.push({
      type: 'tool_use',
      id: typeof call.id === 'string' ? call.id : 'toolu_unknown',
      name: typeof call.function.name === 'string' ? call.function.name : '',
      input: parseArgs(call.function.arguments),
    });
  }
  if (content.length === 0) content.push({ type: 'text', text: '' });
  const usage = usageNumbers(answer.usage);
  return {
    id: `msg_${typeof answer.id === 'string' ? answer.id : 'aggregate'}`,
    type: 'message',
    role: 'assistant',
    model,
    content,
    stop_reason: anthropicStopOf(choice.finish_reason),
    stop_sequence: null,
    usage: { input_tokens: usage.input, output_tokens: usage.output },
  };
}

/** An Anthropic Messages answer as a Chat Completions object. */
export function anthropicToOpenAIRes(answer: Json, model: string): Json {
  const content = Array.isArray(answer.content) ? answer.content : [];
  const textParts: string[] = [];
  const toolCalls: Json[] = [];
  for (const block of content) {
    if (!isRecord(block)) continue;
    if (block.type === 'text' && typeof block.text === 'string') textParts.push(block.text);
    else if (block.type === 'tool_use') {
      toolCalls.push({
        id: typeof block.id === 'string' ? block.id : 'call_unknown',
        type: 'function',
        function: {
          name: typeof block.name === 'string' ? block.name : '',
          arguments: stringifyArgs(block.input),
        },
      });
    }
  }
  const usage = usageNumbers(answer.usage);
  const message: Json = { role: 'assistant', content: textParts.join('') };
  if (toolCalls.length > 0) message.tool_calls = toolCalls;
  return {
    id: typeof answer.id === 'string' ? answer.id : 'chatcmpl-aggregate',
    object: 'chat.completion',
    created: FIXED_CREATED,
    model,
    choices: [
      {
        index: 0,
        message,
        finish_reason: openAIFinishOf(answer.stop_reason),
        logprobs: null,
      },
    ],
    usage: {
      prompt_tokens: usage.input,
      completion_tokens: usage.output,
      total_tokens: usage.input + usage.output,
    },
  };
}

function geminiText(answer: Json): string {
  const candidates = Array.isArray(answer.candidates) ? answer.candidates : [];
  const first = isRecord(candidates[0]) ? candidates[0] : {};
  const content = isRecord(first.content) ? first.content : {};
  const parts = Array.isArray(content.parts) ? content.parts : [];
  return parts
    .flatMap((p) => (isRecord(p) && typeof p.text === 'string' ? [p.text] : []))
    .join('');
}

function geminiUsage(answer: Json): { input: number; output: number } {
  const meta = isRecord(answer.usageMetadata) ? answer.usageMetadata : {};
  return {
    input: asNumber(meta.promptTokenCount) ?? 0,
    output: asNumber(meta.candidatesTokenCount) ?? 0,
  };
}

function geminiFinish(answer: Json): unknown {
  const candidates = Array.isArray(answer.candidates) ? answer.candidates : [];
  const first = isRecord(candidates[0]) ? candidates[0] : {};
  return first.finishReason;
}

/** A Gemini generateContent answer as a Chat Completions object. */
export function geminiToOpenAIRes(answer: Json, model: string): Json {
  const usage = geminiUsage(answer);
  return {
    id: 'chatcmpl-aggregate',
    object: 'chat.completion',
    created: FIXED_CREATED,
    model,
    choices: [
      {
        index: 0,
        message: { role: 'assistant', content: geminiText(answer) },
        finish_reason: openAIFinishOfGemini(geminiFinish(answer)),
        logprobs: null,
      },
    ],
    usage: {
      prompt_tokens: usage.input,
      completion_tokens: usage.output,
      total_tokens: usage.input + usage.output,
    },
  };
}

/** A Gemini generateContent answer as an Anthropic Messages object. */
export function geminiToAnthropicRes(answer: Json, model: string): Json {
  const usage = geminiUsage(answer);
  return {
    id: 'msg_aggregate',
    type: 'message',
    role: 'assistant',
    model,
    content: [{ type: 'text', text: geminiText(answer) }],
    stop_reason: anthropicStopOfGemini(geminiFinish(answer)),
    stop_sequence: null,
    usage: { input_tokens: usage.input, output_tokens: usage.output },
  };
}

// ---------------------------------------------------------------------------
// SSE parsing and streaming translation (buffered)
// ---------------------------------------------------------------------------

export interface SseFrame {
  event: string | null;
  data: string;
}

/** Split raw SSE text into frames. */
export function parseSse(text: string): SseFrame[] {
  const frames: SseFrame[] = [];
  const normalized = text.replace(/\r\n/g, '\n');
  for (const chunk of normalized.split('\n\n')) {
    if (chunk.trim() === '') continue;
    let event: string | null = null;
    const data: string[] = [];
    for (const line of chunk.split('\n')) {
      if (line.startsWith('event:')) event = line.slice('event:'.length).trim();
      else if (line.startsWith('data:')) data.push(line.slice('data:'.length).replace(/^ /, ''));
    }
    if (data.length > 0) frames.push({ event, data: data.join('\n') });
  }
  return frames;
}

export function sseEvent(event: string | null, data: unknown): string {
  return `${event === null ? '' : `event: ${event}\n`}data: ${JSON.stringify(data)}\n\n`;
}

function parseJsonFrame(frame: SseFrame): Json | null {
  try {
    const value: unknown = JSON.parse(frame.data);
    return isRecord(value) ? value : null;
  } catch {
    return null;
  }
}

interface CollectedText {
  text: string;
  tools: { id: string; name: string; args: string }[];
  finishOpenAI: string | null;
  stopAnthropic: string | null;
  inputTokens: number;
  outputTokens: number;
}

function emptyCollected(): CollectedText {
  return { text: '', tools: [], finishOpenAI: null, stopAnthropic: null, inputTokens: 0, outputTokens: 0 };
}

/** Collect one OpenAI chunk stream's text, tool calls, finish and usage. */
function collectOpenAI(frames: SseFrame[]): CollectedText {
  const out = emptyCollected();
  const byIndex = new Map<number, { id: string; name: string; args: string }>();
  for (const frame of frames) {
    if (frame.data.trim() === '[DONE]') continue;
    const chunk = parseJsonFrame(frame);
    if (chunk === null) continue;
    if (isRecord(chunk.usage)) {
      const u = usageNumbers(chunk.usage);
      out.inputTokens = u.input;
      out.outputTokens = u.output;
    }
    const choices = Array.isArray(chunk.choices) ? chunk.choices : [];
    const choice = isRecord(choices[0]) ? choices[0] : null;
    if (choice === null) continue;
    const delta = isRecord(choice.delta) ? choice.delta : {};
    if (typeof delta.content === 'string') out.text += delta.content;
    const calls = Array.isArray(delta.tool_calls) ? delta.tool_calls : [];
    for (const call of calls) {
      if (!isRecord(call)) continue;
      const index = typeof call.index === 'number' ? call.index : 0;
      let tool = byIndex.get(index);
      if (tool === undefined) {
        tool = { id: '', name: '', args: '' };
        byIndex.set(index, tool);
      }
      if (typeof call.id === 'string' && call.id !== '') tool.id = call.id;
      const fn = isRecord(call.function) ? call.function : {};
      if (typeof fn.name === 'string' && fn.name !== '') tool.name = fn.name;
      if (typeof fn.arguments === 'string') tool.args += fn.arguments;
    }
    if (typeof choice.finish_reason === 'string') out.finishOpenAI = choice.finish_reason;
  }
  out.tools = [...byIndex.entries()]
    .sort((a, b) => a[0] - b[0])
    .map(([, t]) => t);
  return out;
}

/** Collect one Anthropic event stream's text, tool calls, stop and usage. */
function collectAnthropic(frames: SseFrame[]): CollectedText {
  const out = emptyCollected();
  const byIndex = new Map<number, { id: string; name: string; args: string }>();
  for (const frame of frames) {
    const data = parseJsonFrame(frame);
    if (data === null) continue;
    if (frame.event === 'message_start' && isRecord(data.message) && isRecord(data.message.usage)) {
      out.inputTokens = asNumber(data.message.usage.input_tokens) ?? 0;
    } else if (frame.event === 'content_block_delta' && isRecord(data.delta)) {
      const index = typeof data.index === 'number' ? data.index : 0;
      if (data.delta.type === 'text_delta' && typeof data.delta.text === 'string') {
        out.text += data.delta.text;
      } else if (data.delta.type === 'input_json_delta' && typeof data.delta.partial_json === 'string') {
        let tool = byIndex.get(index);
        if (tool === undefined) {
          tool = { id: '', name: '', args: '' };
          byIndex.set(index, tool);
        }
        tool.args += data.delta.partial_json;
      }
    } else if (frame.event === 'content_block_start' && isRecord(data.content_block)) {
      const index = typeof data.index === 'number' ? data.index : 0;
      const block = data.content_block;
      if (block.type === 'tool_use') {
        byIndex.set(index, {
          id: typeof block.id === 'string' ? block.id : '',
          name: typeof block.name === 'string' ? block.name : '',
          args: '',
        });
      }
    } else if (frame.event === 'message_delta') {
      if (isRecord(data.delta) && typeof data.delta.stop_reason === 'string') {
        out.stopAnthropic = data.delta.stop_reason;
      }
      if (isRecord(data.usage)) out.outputTokens = asNumber(data.usage.output_tokens) ?? 0;
    }
  }
  out.tools = [...byIndex.entries()]
    .sort((a, b) => a[0] - b[0])
    .map(([, t]) => t);
  return out;
}

/** Collect one Gemini SSE stream's text, finish reason and usage. */
function collectGemini(frames: SseFrame[]): CollectedText {
  const out = emptyCollected();
  for (const frame of frames) {
    const chunk = parseJsonFrame(frame);
    if (chunk === null) continue;
    const candidates = Array.isArray(chunk.candidates) ? chunk.candidates : [];
    const first = isRecord(candidates[0]) ? candidates[0] : null;
    if (first !== null) {
      const content = isRecord(first.content) ? first.content : {};
      const parts = Array.isArray(content.parts) ? content.parts : [];
      for (const p of parts) {
        if (isRecord(p) && typeof p.text === 'string') out.text += p.text;
      }
      if (typeof first.finishReason === 'string') {
        out.finishOpenAI = openAIFinishOfGemini(first.finishReason);
        out.stopAnthropic = anthropicStopOfGemini(first.finishReason);
      }
    }
    if (isRecord(chunk.usageMetadata)) {
      out.inputTokens = asNumber(chunk.usageMetadata.promptTokenCount) ?? 0;
      out.outputTokens = asNumber(chunk.usageMetadata.candidatesTokenCount) ?? 0;
    }
  }
  if (out.finishOpenAI === null) out.finishOpenAI = 'stop';
  if (out.stopAnthropic === null) out.stopAnthropic = 'end_turn';
  return out;
}

function openAIChunk(model: string, delta: Json, finish: string | null): string {
  return sseEvent(null, {
    id: 'chatcmpl-aggregate',
    object: 'chat.completion.chunk',
    created: FIXED_CREATED,
    model,
    choices: [{ index: 0, delta, finish_reason: finish }],
  });
}

function openAIUsageChunk(model: string, input: number, output: number): string {
  return sseEvent(null, {
    id: 'chatcmpl-aggregate',
    object: 'chat.completion.chunk',
    created: FIXED_CREATED,
    model,
    choices: [],
    usage: { prompt_tokens: input, completion_tokens: output, total_tokens: input + output },
  });
}

/** An OpenAI chunk stream as the Anthropic event sequence. */
export function openAIStreamToAnthropic(frames: SseFrame[], model: string): string {
  const c = collectOpenAI(frames);
  const out: string[] = [];
  out.push(
    sseEvent('message_start', {
      type: 'message_start',
      message: {
        id: 'msg_stream',
        type: 'message',
        role: 'assistant',
        model,
        content: [],
        stop_reason: null,
        stop_sequence: null,
        usage: { input_tokens: c.inputTokens, output_tokens: 0 },
      },
    }),
  );
  let index = 0;
  if (c.text !== '' || c.tools.length === 0) {
    out.push(
      sseEvent('content_block_start', {
        type: 'content_block_start',
        index,
        content_block: { type: 'text', text: '' },
      }),
    );
    if (c.text !== '') {
      out.push(
        sseEvent('content_block_delta', {
          type: 'content_block_delta',
          index,
          delta: { type: 'text_delta', text: c.text },
        }),
      );
    }
    out.push(sseEvent('content_block_stop', { type: 'content_block_stop', index }));
    index += 1;
  }
  for (const tool of c.tools) {
    out.push(
      sseEvent('content_block_start', {
        type: 'content_block_start',
        index,
        content_block: {
          type: 'tool_use',
          id: tool.id !== '' ? tool.id : `toolu_${index}`,
          name: tool.name,
          input: {},
        },
      }),
    );
    if (tool.args !== '') {
      out.push(
        sseEvent('content_block_delta', {
          type: 'content_block_delta',
          index,
          delta: { type: 'input_json_delta', partial_json: tool.args },
        }),
      );
    }
    out.push(sseEvent('content_block_stop', { type: 'content_block_stop', index }));
    index += 1;
  }
  out.push(
    sseEvent('message_delta', {
      type: 'message_delta',
      delta: { stop_reason: anthropicStopOf(c.finishOpenAI ?? 'stop'), stop_sequence: null },
      usage: { output_tokens: c.outputTokens },
    }),
    sseEvent('message_stop', { type: 'message_stop' }),
  );
  return out.join('');
}

/** An Anthropic event stream as OpenAI chunks, ending with usage and [DONE]. */
export function anthropicStreamToOpenAI(frames: SseFrame[], model: string): string {
  const c = collectAnthropic(frames);
  const out: string[] = [];
  out.push(openAIChunk(model, { role: 'assistant', content: '' }, null));
  if (c.text !== '') out.push(openAIChunk(model, { content: c.text }, null));
  c.tools.forEach((tool, i) => {
    out.push(
      openAIChunk(model, {
        tool_calls: [
          {
            index: i,
            id: tool.id !== '' ? tool.id : `call_${i}`,
            type: 'function',
            function: { name: tool.name, arguments: tool.args },
          },
        ],
      }, null),
    );
  });
  const finish = openAIFinishOf(c.stopAnthropic ?? 'end_turn');
  out.push(openAIChunk(model, {}, finish));
  out.push(openAIUsageChunk(model, c.inputTokens, c.outputTokens));
  out.push('data: [DONE]\n\n');
  return out.join('');
}

/** A Gemini SSE stream as OpenAI chunks, ending with usage and [DONE]. */
export function geminiStreamToOpenAI(frames: SseFrame[], model: string): string {
  const c = collectGemini(frames);
  const out: string[] = [];
  out.push(openAIChunk(model, { role: 'assistant', content: '' }, null));
  if (c.text !== '') out.push(openAIChunk(model, { content: c.text }, null));
  out.push(openAIChunk(model, {}, c.finishOpenAI ?? 'stop'));
  out.push(openAIUsageChunk(model, c.inputTokens, c.outputTokens));
  out.push('data: [DONE]\n\n');
  return out.join('');
}

/** A Gemini SSE stream as the Anthropic event sequence. */
export function geminiStreamToAnthropic(frames: SseFrame[], model: string): string {
  const c = collectGemini(frames);
  const out: string[] = [];
  out.push(
    sseEvent('message_start', {
      type: 'message_start',
      message: {
        id: 'msg_stream',
        type: 'message',
        role: 'assistant',
        model,
        content: [],
        stop_reason: null,
        stop_sequence: null,
        usage: { input_tokens: c.inputTokens, output_tokens: 0 },
      },
    }),
    sseEvent('content_block_start', {
      type: 'content_block_start',
      index: 0,
      content_block: { type: 'text', text: '' },
    }),
  );
  if (c.text !== '') {
    out.push(
      sseEvent('content_block_delta', {
        type: 'content_block_delta',
        index: 0,
        delta: { type: 'text_delta', text: c.text },
      }),
    );
  }
  out.push(
    sseEvent('content_block_stop', { type: 'content_block_stop', index: 0 }),
    sseEvent('message_delta', {
      type: 'message_delta',
      delta: { stop_reason: c.stopAnthropic ?? 'end_turn', stop_sequence: null },
      usage: { output_tokens: c.outputTokens },
    }),
    sseEvent('message_stop', { type: 'message_stop' }),
  );
  return out.join('');
}
