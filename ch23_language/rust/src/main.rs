//! Minimal agent-runtime loop for Chapter 23 (Rust, tokio + serde_json).
//! Same work as ../py/runtime_py.py: see that file for the step definition.
//!
//! --repr typed  transcript held as Vec<Message> (derive(Serialize, Deserialize))
//! --repr value  transcript held as serde_json::Value (dynamic JSON, as many engines do)
//! --threads 1   tokio current_thread runtime; >1 multi_thread with that many workers
use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};
use rand_distr::{Distribution, LogNormal};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::collections::HashMap;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

#[cfg(feature = "mimalloc")]
#[global_allocator]
static GLOBAL: mimalloc::MiMalloc = mimalloc::MiMalloc;

#[derive(Serialize, Deserialize, Clone)]
struct ToolCall {
    id: String,
    name: String,
    arguments: Value,
}

#[derive(Serialize, Deserialize, Clone)]
struct Message {
    role: String,
    content: String,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    tool_calls: Vec<ToolCall>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    tool_call_id: Option<String>,
}

#[derive(Serialize, Deserialize)]
struct State {
    session_id: String,
    step: u32,
    transcript: Vec<Message>,
}

#[derive(Serialize)]
struct Delta<'a> {
    session_id: &'a str,
    step: u32,
    append: [&'a Message; 2],
}

#[derive(Serialize)]
struct ToolResult<'a> {
    path: &'a str,
    step: u32,
    session: &'a str,
    #[serde(skip_serializing_if = "Option::is_none")]
    passed: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    failed: Option<u32>,
    content: &'a str,
}

struct Fixture {
    transcript_text: String,
    reply_text: String,
    tool_text: String,
    state_1mb_path: String,
}

type Handler = fn(&str, u32, &str, &str) -> String;

fn read_file(sid: &str, step: u32, path: &str, tool: &str) -> String {
    serde_json::to_string(&ToolResult { path, step, session: sid, passed: None, failed: None, content: tool }).unwrap()
}
fn run_tests(sid: &str, step: u32, path: &str, tool: &str) -> String {
    serde_json::to_string(&ToolResult { path, step, session: sid, passed: Some(14), failed: Some(0), content: tool }).unwrap()
}
fn other(sid: &str, step: u32, path: &str, _tool: &str) -> String {
    serde_json::to_string(&ToolResult { path, step, session: sid, passed: None, failed: None, content: "" }).unwrap()
}

fn registry() -> HashMap<&'static str, Handler> {
    let mut m: HashMap<&'static str, Handler> = HashMap::new();
    m.insert("read_file", read_file);
    m.insert("run_tests", run_tests);
    for n in ["replace_text", "list_files", "search_code", "git_diff", "export_report", "ask_user"] {
        m.insert(n, other);
    }
    m
}

fn check_args(args: &Value) -> Result<&str, String> {
    let obj = args.as_object().ok_or("arguments must be an object")?;
    match obj.get("path") {
        Some(Value::String(s)) if !s.is_empty() => Ok(s.as_str()),
        _ => Err("path must be a non-empty string".into()),
    }
}

fn status_kib(key: &str) -> i64 {
    let s = std::fs::read_to_string("/proc/self/status").unwrap_or_default();
    for line in s.lines() {
        if line.starts_with(key) {
            return line.split_whitespace().nth(1).and_then(|v| v.parse().ok()).unwrap_or(-1);
        }
    }
    -1
}

#[derive(Clone)]
struct Args {
    mode: String,
    fixtures: String,
    n: usize,
    steps: u32,
    median_ms: f64,
    sigma: f64,
    ckpt: String,
    seed: u64,
    reps: usize,
    write_reps: usize,
    tmpdir: String,
    out: String,
    repr: String,
    threads: usize,
    stagger_ms: f64,
}

fn parse_args() -> Args {
    let mut a = Args {
        mode: String::new(), fixtures: String::new(), n: 1000, steps: 5, median_ms: 2000.0, sigma: 0.6,
        ckpt: "full".into(), seed: 23, reps: 100, write_reps: 30, tmpdir: "/tmp/ch23_ckpt_rs".into(),
        out: String::new(), repr: "typed".into(), threads: 1, stagger_ms: 0.0,
    };
    let v: Vec<String> = std::env::args().collect();
    let mut i = 1;
    while i < v.len() {
        let val = v.get(i + 1).cloned().unwrap_or_default();
        match v[i].as_str() {
            "--mode" => a.mode = val,
            "--fixtures" => a.fixtures = val,
            "--n" => a.n = val.parse().unwrap(),
            "--steps" => a.steps = val.parse().unwrap(),
            "--median-ms" => a.median_ms = val.parse().unwrap(),
            "--sigma" => a.sigma = val.parse().unwrap(),
            "--ckpt" => a.ckpt = val,
            "--seed" => a.seed = val.parse().unwrap(),
            "--reps" => a.reps = val.parse().unwrap(),
            "--write-reps" => a.write_reps = val.parse().unwrap(),
            "--tmpdir" => a.tmpdir = val,
            "--out" => a.out = val,
            "--repr" => a.repr = val,
            "--threads" => a.threads = val.parse().unwrap(),
            "--stagger-ms" => a.stagger_ms = val.parse().unwrap(),
            other => panic!("unknown argument {other}"),
        }
        i += 2;
    }
    a
}

fn load(d: &str) -> Fixture {
    let r = |f: &str| std::fs::read_to_string(format!("{d}/{f}")).unwrap();
    Fixture {
        transcript_text: r("transcript.json"),
        reply_text: r("model_reply.json"),
        tool_text: r("tool_result.txt"),
        state_1mb_path: format!("{d}/state_1mb.json"),
    }
}

#[derive(Default)]
struct Samples {
    sched: Vec<f64>,
    work: Vec<f64>,
    phases: [f64; 4],
    bytes: usize,
}

enum Session {
    Typed(State),
    Dynamic(Value),
}

fn step_typed(st: &mut State, step: u32, fx: &Fixture, reg: &HashMap<&'static str, Handler>, full: bool, ph: &mut [f64; 4]) -> usize {
    let t0 = Instant::now();
    let reply: Message = serde_json::from_str(&fx.reply_text).unwrap();
    let t1 = Instant::now();
    let call = &reply.tool_calls[0];
    let h = reg.get(call.name.as_str()).expect("unknown tool");
    let path = check_args(&call.arguments).unwrap();
    let content = h(&st.session_id, step, path, &fx.tool_text);
    let tool_msg = Message { role: "tool".into(), content, tool_calls: vec![], tool_call_id: Some(call.id.clone()) };
    let t2 = Instant::now();
    st.transcript.push(reply);
    st.transcript.push(tool_msg);
    st.step = step;
    let t3 = Instant::now();
    let blob = if full {
        serde_json::to_vec(&*st).unwrap()
    } else {
        let n = st.transcript.len();
        let mut b = serde_json::to_vec(&Delta { session_id: &st.session_id, step, append: [&st.transcript[n - 2], &st.transcript[n - 1]] }).unwrap();
        b.push(b'\n');
        b
    };
    let t4 = Instant::now();
    ph[0] += (t1 - t0).as_secs_f64();
    ph[1] += (t2 - t1).as_secs_f64();
    ph[2] += (t3 - t2).as_secs_f64();
    ph[3] += (t4 - t3).as_secs_f64();
    blob.len()
}

fn step_value(st: &mut Value, step: u32, fx: &Fixture, reg: &HashMap<&'static str, Handler>, full: bool, ph: &mut [f64; 4]) -> usize {
    let t0 = Instant::now();
    let reply: Value = serde_json::from_str(&fx.reply_text).unwrap();
    let t1 = Instant::now();
    let call = &reply["tool_calls"][0];
    let name = call["name"].as_str().unwrap();
    let h = reg.get(name).expect("unknown tool");
    let path = check_args(&call["arguments"]).unwrap();
    let sid = st["session_id"].as_str().unwrap().to_string();
    let content = h(&sid, step, path, &fx.tool_text);
    let tool_msg = json!({"role": "tool", "tool_call_id": call["id"].clone(), "content": content});
    let t2 = Instant::now();
    let tr = st["transcript"].as_array_mut().unwrap();
    tr.push(reply);
    tr.push(tool_msg);
    st["step"] = json!(step);
    let t3 = Instant::now();
    let blob = if full {
        serde_json::to_vec(&*st).unwrap()
    } else {
        let tr = st["transcript"].as_array().unwrap();
        let n = tr.len();
        let mut b = serde_json::to_vec(&json!({"session_id": sid, "step": step, "append": [&tr[n - 2], &tr[n - 1]]})).unwrap();
        b.push(b'\n');
        b
    };
    let t4 = Instant::now();
    ph[0] += (t1 - t0).as_secs_f64();
    ph[1] += (t2 - t1).as_secs_f64();
    ph[2] += (t3 - t2).as_secs_f64();
    ph[3] += (t4 - t3).as_secs_f64();
    blob.len()
}

async fn run_sessions(fx: Arc<Fixture>, a: Args) -> Value {
    let reg = Arc::new(registry());
    let rss_base = status_kib("VmRSS:");
    let c0 = Instant::now();
    let mut states = Vec::with_capacity(a.n);
    for i in 0..a.n {
        let sid = format!("s{:05}", i);
        if a.repr == "typed" {
            let tr: Vec<Message> = serde_json::from_str(&fx.transcript_text).unwrap();
            states.push(Session::Typed(State { session_id: sid, step: 0, transcript: tr }));
        } else {
            let tr: Value = serde_json::from_str(&fx.transcript_text).unwrap();
            states.push(Session::Dynamic(json!({"session_id": sid, "step": 0, "transcript": tr})));
        }
    }
    let create_s = c0.elapsed().as_secs_f64();
    let (gate_tx, gate_rx) = tokio::sync::watch::channel(false);
    let parked = Arc::new(AtomicUsize::new(0));
    let mu = if a.median_ms > 0.0 { (a.median_ms / 1000.0).ln() } else { 0.0 };
    let full = a.ckpt == "full";
    let mut handles = Vec::with_capacity(a.n);
    for (i, mut st) in states.into_iter().enumerate() {
        let fx = fx.clone();
        let reg = reg.clone();
        let parked = parked.clone();
        let mut rx = gate_rx.clone();
        let a = a.clone();
        handles.push(tokio::spawn(async move {
            let mut rng = SmallRng::seed_from_u64(a.seed * 1_000_003 + i as u64);
            let dist = LogNormal::new(mu, a.sigma).unwrap();
            let mut s = Samples::default();
            parked.fetch_add(1, Ordering::SeqCst);
            while !*rx.borrow_and_update() {
                rx.changed().await.unwrap();
            }
            if a.stagger_ms > 0.0 {
                let off = (rng.gen_range(0.0..a.stagger_ms / 1000.0) * 1000.0f64).round() / 1000.0;
                tokio::time::sleep(Duration::from_secs_f64(off)).await;
            }
            for step in 1..=a.steps {
                let d = if a.median_ms > 0.0 { (dist.sample(&mut rng) * 1000.0).round() / 1000.0 } else { 0.0 };
                let ts = Instant::now();
                if d > 0.0 {
                    tokio::time::sleep(Duration::from_secs_f64(d)).await;
                } else {
                    tokio::task::yield_now().await;
                }
                let tw = Instant::now();
                s.bytes += match &mut st {
                    Session::Typed(t) => step_typed(t, step, &fx, &reg, full, &mut s.phases),
                    Session::Dynamic(v) => step_value(v, step, &fx, &reg, full, &mut s.phases),
                };
                let td = Instant::now();
                s.sched.push((tw - ts).as_secs_f64() - d);
                s.work.push((td - tw).as_secs_f64());
            }
            s
        }));
    }
    while parked.load(Ordering::SeqCst) < a.n {
        tokio::task::yield_now().await;
    }
    let rss_parked = status_kib("VmRSS:");
    let t0 = Instant::now();
    gate_tx.send(true).unwrap();
    let mut sched = Vec::new();
    let mut work = Vec::new();
    let mut ph = [0.0f64; 4];
    let mut bytes = 0usize;
    for h in handles {
        let s = h.await.unwrap();
        sched.extend(s.sched.iter().map(|x| (x * 1e7).round() / 10.0));
        work.extend(s.work.iter().map(|x| (x * 1e7).round() / 10.0));
        for k in 0..4 {
            ph[k] += s.phases[k];
        }
        bytes += s.bytes;
    }
    let elapsed = t0.elapsed().as_secs_f64();
    json!({
        "lang": format!("rust-{}", a.repr), "n": a.n, "steps": a.steps, "median_ms": a.median_ms, "sigma": a.sigma,
        "ckpt": a.ckpt, "threads": a.threads, "stagger_ms": a.stagger_ms, "rss_base_kib": rss_base, "rss_parked_kib": rss_parked,
        "rss_hwm_kib": status_kib("VmHWM:"), "create_s": create_s, "elapsed_s": elapsed,
        "total_steps": work.len(), "bytes_out": bytes,
        "phase_s": {"parse": ph[0], "dispatch": ph[1], "append": ph[2], "checkpoint": ph[3]},
        "sched_us": sched, "work_us": work,
        "gc": {"count": 0, "total_ms": 0.0, "max_ms": 0.0},
        "version": "rustc (see build log)"
    })
}

fn run_checkpoint(fx: &Fixture, a: &Args) -> Value {
    let text = std::fs::read_to_string(&fx.state_1mb_path).unwrap();
    let (mut ser, mut de, mut wr) = (Vec::new(), Vec::new(), Vec::new());
    let mut size = 0;
    let msgs;
    if a.repr == "typed" {
        let state: State = serde_json::from_str(&text).unwrap();
        msgs = state.transcript.len();
        for i in 0..a.reps + 5 {
            let t0 = Instant::now();
            let blob = serde_json::to_vec(&state).unwrap();
            let t1 = Instant::now();
            let back: State = serde_json::from_slice(&blob).unwrap();
            let t2 = Instant::now();
            assert_eq!(back.transcript.len(), msgs);
            if i >= 5 {
                ser.push((t1 - t0).as_secs_f64() * 1e6);
                de.push((t2 - t1).as_secs_f64() * 1e6);
            }
            size = blob.len();
        }
    } else {
        let state: Value = serde_json::from_str(&text).unwrap();
        msgs = state["transcript"].as_array().unwrap().len();
        for i in 0..a.reps + 5 {
            let t0 = Instant::now();
            let blob = serde_json::to_vec(&state).unwrap();
            let t1 = Instant::now();
            let back: Value = serde_json::from_slice(&blob).unwrap();
            let t2 = Instant::now();
            assert_eq!(back["transcript"].as_array().unwrap().len(), msgs);
            if i >= 5 {
                ser.push((t1 - t0).as_secs_f64() * 1e6);
                de.push((t2 - t1).as_secs_f64() * 1e6);
            }
            size = blob.len();
        }
    }
    std::fs::create_dir_all(&a.tmpdir).unwrap();
    let state: Value = serde_json::from_str(&text).unwrap();
    let blob = serde_json::to_vec(&state).unwrap();
    for _ in 0..a.write_reps {
        use std::io::Write;
        let path = format!("{}/ckpt.json", a.tmpdir);
        let tmp = format!("{path}.tmp");
        let t0 = Instant::now();
        let mut f = std::fs::File::create(&tmp).unwrap();
        f.write_all(&blob).unwrap();
        f.sync_all().unwrap();
        drop(f);
        std::fs::rename(&tmp, &path).unwrap();
        std::fs::File::open(&a.tmpdir).unwrap().sync_all().unwrap();
        wr.push(t0.elapsed().as_secs_f64() * 1e6);
    }
    json!({"lang": format!("rust-{}", a.repr), "bytes": size, "messages": msgs,
           "serialize_us": ser, "deserialize_us": de, "write_fsync_us": wr, "version": "rustc"})
}

fn main() {
    let t_start = Instant::now();
    let a = parse_args();
    let fx = Arc::new(load(&a.fixtures));
    match a.mode.as_str() {
        "startup" => {
            let reg = registry();
            let _: Message = serde_json::from_str(&fx.reply_text).unwrap();
            println!("{}", json!({"ready": true, "tools": reg.len(), "init_ms": t_start.elapsed().as_secs_f64() * 1e3}));
        }
        "sessions" => {
            let rt = if a.threads <= 1 {
                tokio::runtime::Builder::new_current_thread().enable_time().build().unwrap()
            } else {
                tokio::runtime::Builder::new_multi_thread().worker_threads(a.threads).enable_time().build().unwrap()
            };
            let res = rt.block_on(run_sessions(fx.clone(), a.clone()));
            std::fs::write(&a.out, serde_json::to_vec(&res).unwrap()).unwrap();
            let mut small = res.clone();
            small.as_object_mut().unwrap().retain(|_, v| !v.is_array());
            println!("{}", small);
        }
        "checkpoint" => {
            let res = run_checkpoint(&fx, &a);
            std::fs::write(&a.out, serde_json::to_vec(&res).unwrap()).unwrap();
            println!("{}", json!({"lang": res["lang"], "bytes": res["bytes"]}));
        }
        m => panic!("unknown mode {m}"),
    }
}
