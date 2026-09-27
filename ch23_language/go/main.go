// Minimal agent-runtime loop for Chapter 23 (Go, goroutines + encoding/json).
// Same work as ../py/runtime_py.py: see that file for the step definition.
package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"math"
	"math/rand"
	"os"
	"runtime"
	"runtime/debug"
	"strconv"
	"strings"
	"sync"
	"sync/atomic"
	"time"
)

type ToolCall struct {
	ID        string         `json:"id"`
	Name      string         `json:"name"`
	Arguments map[string]any `json:"arguments"`
}

type Message struct {
	Role       string     `json:"role"`
	Content    string     `json:"content"`
	ToolCalls  []ToolCall `json:"tool_calls,omitempty"`
	ToolCallID string     `json:"tool_call_id,omitempty"`
}

type State struct {
	SessionID  string    `json:"session_id"`
	Step       int       `json:"step"`
	Transcript []Message `json:"transcript"`
}

type Delta struct {
	SessionID string     `json:"session_id"`
	Step      int        `json:"step"`
	Append    [2]Message `json:"append"`
}

type ToolResult struct {
	Path    string `json:"path"`
	Step    int    `json:"step"`
	Session string `json:"session"`
	Passed  *int   `json:"passed,omitempty"`
	Failed  *int   `json:"failed,omitempty"`
	Content string `json:"content"`
}

type Handler func(sid string, step int, path string, tool string) string

func mustJSON(v any) []byte {
	b, err := json.Marshal(v)
	if err != nil {
		panic(err)
	}
	return b
}

func registry() map[string]Handler {
	p, f := 14, 0
	m := map[string]Handler{
		"read_file": func(sid string, step int, path, tool string) string {
			return string(mustJSON(ToolResult{Path: path, Step: step, Session: sid, Content: tool}))
		},
		"run_tests": func(sid string, step int, path, tool string) string {
			return string(mustJSON(ToolResult{Path: path, Step: step, Session: sid, Passed: &p, Failed: &f, Content: tool}))
		},
	}
	for _, n := range []string{"replace_text", "list_files", "search_code", "git_diff", "export_report", "ask_user"} {
		m[n] = func(sid string, step int, path, tool string) string {
			return string(mustJSON(ToolResult{Path: path, Step: step, Session: sid, Content: ""}))
		}
	}
	return m
}

func checkArgs(args map[string]any) (string, error) {
	if args == nil {
		return "", fmt.Errorf("arguments must be an object")
	}
	s, ok := args["path"].(string)
	if !ok || s == "" {
		return "", fmt.Errorf("path must be a non-empty string")
	}
	return s, nil
}

func statusKiB(key string) int64 {
	b, _ := os.ReadFile("/proc/self/status")
	for _, line := range strings.Split(string(b), "\n") {
		if strings.HasPrefix(line, key) {
			f := strings.Fields(line)
			v, _ := strconv.ParseInt(f[1], 10, 64)
			return v
		}
	}
	return -1
}

type Fixture struct {
	transcript, reply, tool string
	state1mb                string
}

func load(d string) Fixture {
	r := func(f string) string {
		b, err := os.ReadFile(d + "/" + f)
		if err != nil {
			panic(err)
		}
		return string(b)
	}
	return Fixture{r("transcript.json"), r("model_reply.json"), r("tool_result.txt"), d + "/state_1mb.json"}
}

func doStep(st *State, step int, fx *Fixture, reg map[string]Handler, full bool, ph *[4]float64) int {
	t0 := time.Now()
	var reply Message
	if err := json.Unmarshal([]byte(fx.reply), &reply); err != nil {
		panic(err)
	}
	t1 := time.Now()
	call := reply.ToolCalls[0]
	h, ok := reg[call.Name]
	if !ok {
		panic("unknown tool")
	}
	path, err := checkArgs(call.Arguments)
	if err != nil {
		panic(err)
	}
	toolMsg := Message{Role: "tool", ToolCallID: call.ID, Content: h(st.SessionID, step, path, fx.tool)}
	t2 := time.Now()
	st.Transcript = append(st.Transcript, reply, toolMsg)
	st.Step = step
	t3 := time.Now()
	var n int
	if full {
		n = len(mustJSON(st))
	} else {
		n = len(mustJSON(Delta{st.SessionID, step, [2]Message{reply, toolMsg}})) + 1
	}
	t4 := time.Now()
	ph[0] += t1.Sub(t0).Seconds()
	ph[1] += t2.Sub(t1).Seconds()
	ph[2] += t3.Sub(t2).Seconds()
	ph[3] += t4.Sub(t3).Seconds()
	return n
}

type samples struct {
	sched, work []float64
	ph          [4]float64
	bytes       int
}

func runSessions(fx *Fixture, n, steps int, medianMs, sigma float64, ckpt string, seed int64, staggerMs float64) map[string]any {
	reg := registry()
	runtime.GC()
	rssBase := statusKiB("VmRSS:")
	c0 := time.Now()
	states := make([]*State, n)
	for i := 0; i < n; i++ {
		var tr []Message
		if err := json.Unmarshal([]byte(fx.transcript), &tr); err != nil {
			panic(err)
		}
		states[i] = &State{SessionID: fmt.Sprintf("s%05d", i), Transcript: tr}
	}
	createS := time.Since(c0).Seconds()
	gate := make(chan struct{})
	var parked int64
	var wg sync.WaitGroup
	res := make([]samples, n)
	mu := 0.0
	if medianMs > 0 {
		mu = math.Log(medianMs / 1000.0)
	}
	full := ckpt == "full"
	for i := 0; i < n; i++ {
		wg.Add(1)
		go func(i int) {
			defer wg.Done()
			rng := rand.New(rand.NewSource(seed*1000003 + int64(i)))
			s := &res[i]
			st := states[i]
			atomic.AddInt64(&parked, 1)
			<-gate
			if staggerMs > 0 {
				off := math.Round(rng.Float64()*staggerMs) / 1000
				time.Sleep(time.Duration(off * float64(time.Second)))
			}
			for step := 1; step <= steps; step++ {
				d := 0.0
				if medianMs > 0 {
					d = math.Round(math.Exp(mu+sigma*rng.NormFloat64())*1000) / 1000
				}
				ts := time.Now()
				if d > 0 {
					time.Sleep(time.Duration(d * float64(time.Second)))
				} else {
					runtime.Gosched()
				}
				tw := time.Now()
				s.bytes += doStep(st, step, fx, reg, full, &s.ph)
				td := time.Now()
				s.sched = append(s.sched, math.Round((tw.Sub(ts).Seconds()-d)*1e7)/10)
				s.work = append(s.work, math.Round(td.Sub(tw).Seconds()*1e7)/10)
			}
		}(i)
	}
	for atomic.LoadInt64(&parked) < int64(n) {
		runtime.Gosched()
	}
	rssParked := statusKiB("VmRSS:")
	var ms0 runtime.MemStats
	runtime.ReadMemStats(&ms0)
	t0 := time.Now()
	close(gate)
	wg.Wait()
	elapsed := time.Since(t0).Seconds()
	var ms1 runtime.MemStats
	runtime.ReadMemStats(&ms1)
	sched, work := []float64{}, []float64{}
	var ph [4]float64
	bytes := 0
	for i := range res {
		sched = append(sched, res[i].sched...)
		work = append(work, res[i].work...)
		for k := 0; k < 4; k++ {
			ph[k] += res[i].ph[k]
		}
		bytes += res[i].bytes
	}
	// Stop-the-world pause durations recorded by the runtime during the step phase.
	numGC := ms1.NumGC - ms0.NumGC
	var maxPause, totPause float64
	for g := ms0.NumGC; g < ms1.NumGC; g++ {
		p := float64(ms1.PauseNs[g%256]) / 1e6
		totPause += p
		if p > maxPause {
			maxPause = p
		}
	}
	return map[string]any{
		"lang": "go", "n": n, "steps": steps, "median_ms": medianMs, "sigma": sigma, "ckpt": ckpt, "stagger_ms": staggerMs,
		"gomaxprocs": runtime.GOMAXPROCS(0), "rss_base_kib": rssBase, "rss_parked_kib": rssParked,
		"rss_hwm_kib": statusKiB("VmHWM:"), "create_s": createS, "elapsed_s": elapsed,
		"total_steps": len(work), "bytes_out": bytes,
		"phase_s":   map[string]float64{"parse": ph[0], "dispatch": ph[1], "append": ph[2], "checkpoint": ph[3]},
		"sched_us":  sched, "work_us": work,
		"gc":        map[string]any{"count": numGC, "total_ms": totPause, "max_ms": maxPause, "cpu_fraction": ms1.GCCPUFraction},
		"version":   runtime.Version(),
	}
}

func runCheckpoint(fx *Fixture, reps, writeReps int, tmpdir string) map[string]any {
	b, _ := os.ReadFile(fx.state1mb)
	var st State
	if err := json.Unmarshal(b, &st); err != nil {
		panic(err)
	}
	var ser, de, wr []float64
	size := 0
	for i := 0; i < reps+5; i++ {
		t0 := time.Now()
		blob := mustJSON(&st)
		t1 := time.Now()
		var back State
		if err := json.Unmarshal(blob, &back); err != nil {
			panic(err)
		}
		t2 := time.Now()
		if len(back.Transcript) != len(st.Transcript) {
			panic("mismatch")
		}
		if i >= 5 {
			ser = append(ser, t1.Sub(t0).Seconds()*1e6)
			de = append(de, t2.Sub(t1).Seconds()*1e6)
		}
		size = len(blob)
	}
	os.MkdirAll(tmpdir, 0o755)
	blob := mustJSON(&st)
	for i := 0; i < writeReps; i++ {
		path := tmpdir + "/ckpt.json"
		tmp := path + ".tmp"
		t0 := time.Now()
		f, err := os.Create(tmp)
		if err != nil {
			panic(err)
		}
		f.Write(blob)
		f.Sync()
		f.Close()
		os.Rename(tmp, path)
		dfd, _ := os.Open(tmpdir)
		dfd.Sync()
		dfd.Close()
		wr = append(wr, time.Since(t0).Seconds()*1e6)
	}
	return map[string]any{"lang": "go", "bytes": size, "messages": len(st.Transcript),
		"serialize_us": ser, "deserialize_us": de, "write_fsync_us": wr, "version": runtime.Version()}
}

func main() {
	tStart := time.Now()
	mode := flag.String("mode", "", "")
	fixtures := flag.String("fixtures", "", "")
	n := flag.Int("n", 1000, "")
	steps := flag.Int("steps", 5, "")
	median := flag.Float64("median-ms", 2000, "")
	sigma := flag.Float64("sigma", 0.6, "")
	ckpt := flag.String("ckpt", "full", "")
	seed := flag.Int64("seed", 23, "")
	reps := flag.Int("reps", 100, "")
	writeReps := flag.Int("write-reps", 30, "")
	tmpdir := flag.String("tmpdir", "/tmp/ch23_ckpt_go", "")
	out := flag.String("out", "", "")
	stagger := flag.Float64("stagger-ms", 0, "")
	flag.Parse()
	_ = debug.SetGCPercent
	fx := load(*fixtures)
	switch *mode {
	case "startup":
		reg := registry()
		var m Message
		json.Unmarshal([]byte(fx.reply), &m)
		fmt.Println(string(mustJSON(map[string]any{"ready": true, "tools": len(reg), "init_ms": time.Since(tStart).Seconds() * 1e3})))
	case "sessions", "checkpoint":
		var res map[string]any
		if *mode == "sessions" {
			res = runSessions(&fx, *n, *steps, *median, *sigma, *ckpt, *seed, *stagger)
		} else {
			res = runCheckpoint(&fx, *reps, *writeReps, *tmpdir)
		}
		os.WriteFile(*out, mustJSON(res), 0o644)
		small := map[string]any{}
		for k, v := range res {
			if _, isList := v.([]float64); !isList {
				small[k] = v
			}
		}
		fmt.Println(string(mustJSON(small)))
	default:
		panic("unknown mode")
	}
}
