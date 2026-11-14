// Local eval harness for free-model-aggregator-01 v2.
//
// Runs the sealed practice env standalone (its process-driver mode, no
// docker), runs the entry against it, scores the export with the sealed
// scorer, and reports per-request routing, quality vs the best rating,
// latency, and failover events. A `--bench` phase first measures every
// provider's catalog-probe latency and one inference round-trip against a
// throwaway env, so the numbers below are measured, not asserted.
//
//   node eval/bench.mjs [--trials 0,1,2] [--bench] [--repeat 2]
//
// The harness is dev tooling only: it never ships into the trial (the trial
// runs main.ts alone) and its timings never reach the entry.

import { spawn } from 'node:child_process';
import { existsSync, mkdirSync, readFileSync, rmSync, writeFileSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';

const ROOT = join(dirname(fileURLToPath(import.meta.url)), '..');
const KIT = join(ROOT, '..', 'free-model-aggregator-01-kit');
const ENV_SERVER = join(KIT, 'program', 'env', 'server.mjs');
const SCORER = join(KIT, 'program', 'scorer', 'score.mjs');
const SEEDS = {
  0: '4f751e962e1eb1d5b461859662944e82c2ce329459e7bc6842a5ab35170c3f81',
  1: '01fc82338daa99de8447d0258d87cb2fc1f5eb386cc8b6237c3c86259b73e448',
  2: '21086a502f48ccc50a24c68de145b40cbb8cb4877b2c8c7d3cb2159c180a13dab6410cd67194c7d12880fa070f8bd5',
};

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

async function waitForFile(path, timeoutMs = 15000) {
  const start = Date.now();
  while (Date.now() - start < timeoutMs) {
    if (existsSync(path)) return true;
    await sleep(100);
  }
  return false;
}

function run(cmd, args, env, opts = {}) {
  return new Promise((resolve) => {
    const child = spawn(cmd, args, {
      env: { ...process.env, ...env },
      stdio: ['ignore', 'pipe', 'pipe'],
      ...opts,
    });
    let stdout = '';
    let stderr = '';
    child.stdout.on('data', (d) => (stdout += d));
    child.stderr.on('data', (d) => (stderr += d));
    child.on('exit', (code, signal) => resolve({ code, signal, stdout, stderr }));
    child.on('error', (err) => resolve({ code: -1, signal: null, stdout, stderr: String(err) }));
  });
}

function runDetached(cmd, args, env, cwd) {
  return spawn(cmd, args, { env: { ...process.env, ...env }, stdio: ['ignore', 'pipe', 'pipe'], cwd });
}

async function waitExit(child, timeoutMs) {
  return new Promise((resolve) => {
    const timer = setTimeout(() => resolve(false), timeoutMs);
    child.on('exit', () => {
      clearTimeout(timer);
      resolve(true);
    });
  });
}

/** Start the practice env for one trial; resolves with its base URL. */
async function startEnv(trial, port, stateDir) {
  const dataDir = join(KIT, 'practice', `trial-${trial}`, 'input', 'env');
  rmSync(stateDir, { recursive: true, force: true });
  mkdirSync(stateDir, { recursive: true });
  const env = runDetached('node', [ENV_SERVER], {
    MAKEFUN_STATE_DIR: stateDir,
    MAKEFUN_ENV_DATA_DIR: dataDir,
    MAKEFUN_ENV_PORT: String(port),
  });
  let envLog = '';
  env.stdout.on('data', (d) => (envLog += d));
  env.stderr.on('data', (d) => (envLog += d));
  const readyPath = join(stateDir, 'ready');
  if (!(await waitForFile(readyPath, 15000))) {
    env.kill('SIGKILL');
    throw new Error(`env did not become ready for trial ${trial}:\n${envLog}`);
  }
  const ready = JSON.parse(readFileSync(readyPath, 'utf8'));
  return { child: env, url: `http://${ready.host}:${ready.port}`, log: () => envLog };
}

/** One trial: env + entry + scorer. Returns scores and the state export. */
async function runTrial(trial, workRoot) {
  const port = 18080 + trial * 10 + Math.floor(Math.random() * 5);
  const stateDir = join(workRoot, `trial-${trial}-state`);
  const { child: envChild, url, log } = await startEnv(trial, port, stateDir);
  const promptFile = join(KIT, 'practice', `trial-${trial}`, 'input', 'prompt.txt');
  const entry = runDetached('node', ['main.ts'], {
    MAKEFUN_ENV_URL: url,
    MAKEFUN_PROMPT_FILE: promptFile,
    MAKEFUN_TRIAL_SEED: SEEDS[trial],
  }, ROOT);
  let entryLog = '';
  entry.stdout.on('data', (d) => (entryLog += d));
  entry.stderr.on('data', (d) => (entryLog += d));
  const exited = await waitExit(entry, 120000);
  if (!exited) {
    entry.kill('SIGKILL');
    envChild.kill('SIGKILL');
    throw new Error(`entry did not exit for trial ${trial}:\n${entryLog}\n${log()}`);
  }
  envChild.kill('SIGTERM');
  const exportPath = join(stateDir, 'export', 'state.json');
  if (!(await waitForFile(exportPath, 10000))) {
    throw new Error(`env wrote no state export for trial ${trial}:\n${log()}`);
  }
  await sleep(300);
  const scoresDir = join(workRoot, `trial-${trial}-scores`);
  mkdirSync(scoresDir, { recursive: true });
  const scored = await run('node', [SCORER], {
    MAKEFUN_OUTPUT_DIR: join(stateDir, 'export'),
    MAKEFUN_REFERENCE_DIR: join(KIT, 'practice', `trial-${trial}`, 'reference'),
    MAKEFUN_SCORES_DIR: scoresDir,
  });
  if (scored.code !== 0) throw new Error(`scorer failed for trial ${trial}: ${scored.stderr}`);
  const state = JSON.parse(readFileSync(exportPath, 'utf8'));
  const scores = JSON.parse(readFileSync(join(scoresDir, 'scores.json'), 'utf8'));
  return { scores, state, entryLog, envLog: log(), scorerOut: scored.stdout.trim() };
}

/** Benchmark phase: probe + one inference round-trip per provider. */
async function runBenchmark(trial, workRoot) {
  const port = 18200 + trial;
  const stateDir = join(workRoot, `trial-${trial}-bench-state`);
  const { child: envChild, url } = await startEnv(trial, port, stateDir);
  const providers = JSON.parse(
    readFileSync(join(KIT, 'practice', `trial-${trial}`, 'input', 'env', 'providers.json'), 'utf8'),
  ).providers;
  const rows = [];
  try {
    for (const p of providers) {
      const base = `${url}${p.path}`;
      const auth =
        p.auth === 'bearer'
          ? { authorization: `Bearer ${p.key}` }
          : p.auth === 'x-api-key'
            ? { 'x-api-key': p.key }
            : p.auth === 'x-goog-api-key'
              ? { 'x-goog-api-key': p.key }
              : {};
      const catalogUrl = p.wire === 'gemini' ? `${base}/v1beta/models` : `${base}/v1/models`;
      const t0 = Date.now();
      const catRes = await fetch(catalogUrl, {
        headers: { ...auth, ...(p.wire === 'anthropic' ? { 'anthropic-version': '2023-06-01' } : {}) },
      });
      const probeMs = Date.now() - t0;
      const catBody = await catRes.json().catch(() => null);
      const listed = catBody === null ? 0 : p.wire === 'gemini' ? (catBody.models ?? []).length : (catBody.data ?? []).length;
      const topFree = p.models.filter((m) => m.free).sort((a, b) => b.rating_milli - a.rating_milli)[0];
      let inferMs = null;
      let inferStatus = null;
      if (topFree !== undefined) {
        let inferUrl;
        let inferBody;
        if (p.wire === 'openai') {
          inferUrl = `${base}/v1/chat/completions`;
          inferBody = { model: topFree.id, messages: [{ role: 'user', content: 'benchmark ping' }], max_tokens: 8 };
        } else if (p.wire === 'anthropic') {
          inferUrl = `${base}/v1/messages`;
          inferBody = { model: topFree.id, messages: [{ role: 'user', content: 'benchmark ping' }], max_tokens: 8 };
        } else {
          inferUrl = `${base}/v1beta/models/${encodeURIComponent(topFree.id)}:generateContent`;
          inferBody = { contents: [{ role: 'user', parts: [{ text: 'benchmark ping' }] }], generationConfig: { maxOutputTokens: 8 } };
        }
        const t1 = Date.now();
        const inferRes = await fetch(inferUrl, {
          method: 'POST',
          headers: { 'content-type': 'application/json', ...auth, ...(p.wire === 'anthropic' ? { 'anthropic-version': '2023-06-01' } : {}) },
          body: JSON.stringify(inferBody),
        });
        inferMs = Date.now() - t1;
        inferStatus = inferRes.status;
        await inferRes.text().catch(() => '');
      }
      rows.push({
        provider: p.id,
        wire: p.wire,
        probeMs,
        listed,
        topModel: topFree?.id ?? null,
        topRating: topFree?.rating_milli ?? null,
        inferMs,
        inferStatus,
      });
    }
  } finally {
    envChild.kill('SIGKILL');
  }
  return rows;
}

async function main() {
  const args = process.argv.slice(2);
  const trialsArg = args.find((a) => a.startsWith('--trials='));
  const trials = trialsArg ? trialsArg.slice('--trials='.length).split(',').map(Number) : [0];
  const doBench = args.includes('--bench');
  const repeatArg = args.find((a) => a.startsWith('--repeat='));
  const repeat = repeatArg ? Number(repeatArg.slice('--repeat='.length)) : 1;
  const workRoot = join(ROOT, 'eval', 'out');
  mkdirSync(workRoot, { recursive: true });

  const runs = [];
  for (const trial of trials) {
    if (doBench) {
      const bench = await runBenchmark(trial, workRoot);
      console.log(`\n== benchmark trial-${trial} (throwaway env; 1 inference call per provider) ==`);
      console.log('provider      wire       probeMs  listed  top model (rating)            inferMs  status');
      for (const r of bench) {
        console.log(
          `${r.provider.padEnd(13)} ${r.wire.padEnd(10)} ${String(r.probeMs).padStart(6)}  ${String(r.listed).padStart(6)}  ${(r.topModel ?? '').padEnd(28)} ${String(r.inferMs ?? '-').padStart(6)}  ${r.inferStatus ?? '-'}`,
        );
      }
      const probeAvg = bench.reduce((n, r) => n + r.probeMs, 0) / bench.length;
      const inferAvg = bench.filter((r) => r.inferMs !== null).reduce((n, r) => n + r.inferMs, 0) / bench.filter((r) => r.inferMs !== null).length;
      console.log(`avg probe ${probeAvg.toFixed(1)} ms, avg inference ${inferAvg.toFixed(1)} ms, ${bench.length} providers, ${bench.filter((r) => r.inferStatus === 200).length} inference calls 200`);
    }
    for (let rep = 0; rep < repeat; rep++) {
      const { scores, state, entryLog } = await runTrial(trial, workRoot);
      runs.push({ trial, rep, scores, state });
      console.log(`\n== trial-${trial} rep ${rep}: score ${scores.score} (quality ${scores.quality}, served ${scores.served}, wire ${scores.wire}, failover ${scores.failover}, catalog ${scores.catalog}, calls ${scores.upstream_calls}, working ${scores.working}) ==`);
      const ratings = JSON.parse(readFileSync(join(KIT, 'practice', `trial-${trial}`, 'reference', 'ratings.json'), 'utf8'));
      const best = new Map(ratings.best.map((b) => [b.id, b.rating_milli]));
      for (const r of state.requests) {
        const fails = r.failures.length > 0 ? ` FAILURES=${r.failures.map((f) => `${f.provider}:${f.status}`).join(',')}` : '';
        const bad = r.checks.filter((c) => !c.pass).map((c) => c.name).join(',');
        console.log(
          ` ${r.id} valid=${r.valid ? '1' : '0'} ${String(r.provider ?? '-').padEnd(11)} ${(r.model ?? '-').padEnd(30)} rating ${r.rating_milli}/${best.get(r.id)} calls=${r.upstream_calls}${fails}${bad ? ` BAD=[${bad}]` : ''}`,
        );
      }
      const refused = entryLog.split('\n').filter((l) => l.includes('refused:'));
      if (refused.length > 0) {
        console.log(' entry refusals:');
        for (const l of refused) console.log(`  ${l.trim()}`);
      }
    }
  }
  if (repeat > 1) {
    const byTrial = new Map();
    for (const r of runs) {
      const list = byTrial.get(r.trial) ?? [];
      list.push(r);
      byTrial.set(r.trial, list);
    }
    for (const [trial, list] of byTrial) {
      const first = JSON.stringify(list[0].state);
      const same = list.every((r) => JSON.stringify(r.state) === first);
      console.log(`\ntrial-${trial} determinism across ${list.length} runs: state exports ${same ? 'IDENTICAL' : 'DIFFER'}`);
    }
  }
  writeFileSync(join(workRoot, 'last-run.json'), JSON.stringify(runs.map((r) => ({ trial: r.trial, rep: r.rep, scores: r.scores })), null, 2));
}

await main();
