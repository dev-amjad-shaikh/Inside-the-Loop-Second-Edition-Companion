// Minimal agent-runtime loop for Chapter 23 (Node.js, async/await + built-in JSON).
// Same work as ../py/runtime_py.py: see that file for the step definition.
import { readFileSync, writeFileSync, openSync, writeSync, fsyncSync, closeSync, renameSync, mkdirSync } from "node:fs";
import { performance, PerformanceObserver } from "node:perf_hooks";

const perf = () => performance.now() / 1000;
const tStart = perf();

function args() {
  const a = { n: 1000, steps: 5, medianMs: 2000, sigma: 0.6, ckpt: "full", seed: 23, reps: 100, writeReps: 30,
              tmpdir: "/tmp/ch23_ckpt_node", staggerMs: 0 };
  const v = process.argv.slice(2);
  for (let i = 0; i < v.length; i += 2) {
    const k = v[i], val = v[i + 1];
    if (k === "--mode") a.mode = val;
    else if (k === "--fixtures") a.fixtures = val;
    else if (k === "--n") a.n = +val;
    else if (k === "--steps") a.steps = +val;
    else if (k === "--median-ms") a.medianMs = +val;
    else if (k === "--sigma") a.sigma = +val;
    else if (k === "--ckpt") a.ckpt = val;
    else if (k === "--seed") a.seed = +val;
    else if (k === "--reps") a.reps = +val;
    else if (k === "--write-reps") a.writeReps = +val;
    else if (k === "--tmpdir") a.tmpdir = val;
    else if (k === "--out") a.out = val;
    else if (k === "--stagger-ms") a.staggerMs = +val;
    else throw new Error("unknown argument " + k);
  }
  return a;
}

function statusKiB(key) {
  for (const line of readFileSync("/proc/self/status", "utf8").split("\n")) {
    if (line.startsWith(key)) return parseInt(line.split(/\s+/)[1], 10);
  }
  return -1;
}

// Small seeded PRNG (mulberry32) and Box-Muller for the lognormal model latency.
function rng32(seed) {
  let t = seed >>> 0;
  return () => {
    t = (t + 0x6d2b79f5) >>> 0;
    let r = Math.imul(t ^ (t >>> 15), 1 | t);
    r ^= r + Math.imul(r ^ (r >>> 7), 61 | r);
    return ((r ^ (r >>> 14)) >>> 0) / 4294967296;
  };
}
function normal(r) {
  const u = 1 - r(), v = r();
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
}

function load(d) {
  return {
    transcript: readFileSync(d + "/transcript.json", "utf8"),
    reply: readFileSync(d + "/model_reply.json", "utf8"),
    tool: readFileSync(d + "/tool_result.txt", "utf8"),
    state1mb: d + "/state_1mb.json",
  };
}

function registry(toolText) {
  const reg = new Map();
  reg.set("read_file", (sid, step, a) => ({ path: a.path, step, session: sid, content: toolText }));
  reg.set("run_tests", (sid, step, a) => ({ path: a.path, step, session: sid, passed: 14, failed: 0, content: toolText }));
  for (const n of ["replace_text", "list_files", "search_code", "git_diff", "export_report", "ask_user"]) {
    reg.set(n, (sid, step, a) => ({ path: a.path, step, session: sid, content: "" }));
  }
  return reg;
}

function checkArgs(a) {
  if (a === null || typeof a !== "object" || Array.isArray(a)) throw new Error("arguments must be an object");
  if (typeof a.path !== "string" || a.path.length === 0) throw new Error("path must be a non-empty string");
}

function doStep(st, step, fx, reg, full, ph) {
  const t0 = perf();
  const reply = JSON.parse(fx.reply);
  const t1 = perf();
  const call = reply.tool_calls[0];
  const h = reg.get(call.name);
  if (!h) throw new Error("unknown tool");
  checkArgs(call.arguments);
  const toolMsg = { role: "tool", tool_call_id: call.id, content: JSON.stringify(h(st.session_id, step, call.arguments)) };
  const t2 = perf();
  st.transcript.push(reply, toolMsg);
  st.step = step;
  const t3 = perf();
  let blob;
  if (full) blob = Buffer.from(JSON.stringify(st), "utf8");
  else blob = Buffer.from(JSON.stringify({ session_id: st.session_id, step, append: [reply, toolMsg] }) + "\n", "utf8");
  const t4 = perf();
  ph[0] += t1 - t0; ph[1] += t2 - t1; ph[2] += t3 - t2; ph[3] += t4 - t3;
  return blob.length;
}

const sleep = (s) => new Promise((res) => setTimeout(res, s * 1000));
const yieldNow = () => new Promise((res) => setImmediate(res));

async function runSessions(fx, a) {
  const reg = registry(fx.tool);
  const gcPauses = [];
  const obs = new PerformanceObserver((list) => { for (const e of list.getEntries()) gcPauses.push(e.duration); });
  if (global.gc) global.gc();
  const rssBase = statusKiB("VmRSS:");
  const c0 = perf();
  const states = [];
  for (let i = 0; i < a.n; i++) {
    states.push({ session_id: "s" + String(i).padStart(5, "0"), step: 0, transcript: JSON.parse(fx.transcript) });
  }
  const createS = perf() - c0;
  let openGate;
  const gate = new Promise((res) => { openGate = res; });
  let parked = 0;
  const sched = [], work = [];
  const ph = [0, 0, 0, 0];
  let bytes = 0;
  const mu = a.medianMs > 0 ? Math.log(a.medianMs / 1000) : 0;
  const full = a.ckpt === "full";
  async function session(i, st) {
    const r = rng32(a.seed * 1000003 + i);
    parked++;
    await gate;
    if (a.staggerMs > 0) await sleep(Math.round(r() * a.staggerMs) / 1000);
    for (let step = 1; step <= a.steps; step++) {
      const d = a.medianMs > 0 ? Math.round(Math.exp(mu + a.sigma * normal(r)) * 1000) / 1000 : 0;
      const ts = perf();
      if (d > 0) await sleep(d); else await yieldNow();
      const tw = perf();
      bytes += doStep(st, step, fx, reg, full, ph);
      const td = perf();
      sched.push(Math.round((tw - ts - d) * 1e7) / 10);
      work.push(Math.round((td - tw) * 1e7) / 10);
    }
  }
  const tasks = states.map((st, i) => session(i, st));
  while (parked < a.n) await yieldNow();
  states.length = 0;
  const rssParked = statusKiB("VmRSS:");
  obs.observe({ entryTypes: ["gc"] });
  const t0 = perf();
  openGate();
  await Promise.all(tasks);
  const elapsed = perf() - t0;
  await yieldNow();
  obs.disconnect();
  return {
    lang: "node", n: a.n, steps: a.steps, median_ms: a.medianMs, sigma: a.sigma, ckpt: a.ckpt, stagger_ms: a.staggerMs,
    rss_base_kib: rssBase, rss_parked_kib: rssParked, rss_hwm_kib: statusKiB("VmHWM:"),
    create_s: createS, elapsed_s: elapsed, total_steps: work.length, bytes_out: bytes,
    phase_s: { parse: ph[0], dispatch: ph[1], append: ph[2], checkpoint: ph[3] },
    sched_us: sched, work_us: work,
    gc: { count: gcPauses.length, total_ms: gcPauses.reduce((x, y) => x + y, 0),
          max_ms: gcPauses.length ? Math.max(...gcPauses) : 0 },
    version: process.version,
  };
}

function runCheckpoint(fx, a) {
  const state = JSON.parse(readFileSync(fx.state1mb, "utf8"));
  const ser = [], de = [], wr = [];
  let size = 0;
  for (let i = 0; i < a.reps + 5; i++) {
    const t0 = perf();
    const blob = Buffer.from(JSON.stringify(state), "utf8");
    const t1 = perf();
    const back = JSON.parse(blob.toString("utf8"));
    const t2 = perf();
    if (back.transcript.length !== state.transcript.length) throw new Error("mismatch");
    if (i >= 5) { ser.push((t1 - t0) * 1e6); de.push((t2 - t1) * 1e6); }
    size = blob.length;
  }
  mkdirSync(a.tmpdir, { recursive: true });
  const blob = Buffer.from(JSON.stringify(state), "utf8");
  for (let i = 0; i < a.writeReps; i++) {
    const path = a.tmpdir + "/ckpt.json", tmp = path + ".tmp";
    const t0 = perf();
    const fd = openSync(tmp, "w");
    writeSync(fd, blob);
    fsyncSync(fd);
    closeSync(fd);
    renameSync(tmp, path);
    const dfd = openSync(a.tmpdir, "r");
    fsyncSync(dfd);
    closeSync(dfd);
    wr.push((perf() - t0) * 1e6);
  }
  return { lang: "node", bytes: size, messages: state.transcript.length,
           serialize_us: ser, deserialize_us: de, write_fsync_us: wr, version: process.version };
}

const a = args();
const fx = load(a.fixtures);
if (a.mode === "startup") {
  const reg = registry(fx.tool);
  JSON.parse(fx.reply);
  console.log(JSON.stringify({ ready: true, tools: reg.size, init_ms: (perf() - tStart) * 1e3 }));
} else {
  const res = a.mode === "sessions" ? await runSessions(fx, a) : runCheckpoint(fx, a);
  writeFileSync(a.out, JSON.stringify(res));
  const small = {};
  for (const [k, v] of Object.entries(res)) if (!Array.isArray(v)) small[k] = v;
  console.log(JSON.stringify(small));
}
