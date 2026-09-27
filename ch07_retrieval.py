"""Chapter 7: retrieval.

Builds a small support-policy index twice (fixed-size chunks and heading/table-
aware chunks), runs access-filtered hybrid queries, reproduces a missed answer,
and evaluates the fix. Also shows effective-date selection, withdrawal
propagation, an embedding-model mismatch and agentic search through a tool.

    python3 ch07_retrieval.py --output runs/ch07
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from itl.core import ScriptedModel, ToolRegistry, run_agent, write_run  # noqa: E402
from itl import ch07_retrieval as R  # noqa: E402

FIX = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fixtures", "support_policy")
FILES = {(d["doc_id"], d["version"]): d["file"]
         for d in json.load(open(os.path.join(FIX, "manifest.json"), encoding="utf-8"))}

CUSTOMER = {"id": "customer:c-4471", "audiences": ["public"]}
STAFF = {"id": "staff:agent-12", "audiences": ["public", "staff"]}

Q1 = "Can I return an opened blender that arrived defective?"

# The evaluation set. subject/qualifiers are what a model would extract from
# the question; the stand-in answerer receives them directly.
EVAL = [
    {"id": "q1-opened-defective", "question": Q1, "order_date": "2026-05-10",
     "subject": "opened", "qualifiers": ["defective"], "expected": "yes",
     "needed": ("returns-policy", 2, ["| Condition | Returnable | Window |",
                                      "| Defective or damaged on arrival | Yes | 90 days from delivery |"])},
    {"id": "q2-opened-changed-mind", "question": "I opened the kettle and changed my mind. Can I return it?",
     "order_date": "2026-05-10", "subject": "opened", "qualifiers": [], "expected": "no",
     "needed": ("returns-policy", 2, ["Opened products cannot be returned."])},
    {"id": "q3-old-order-opened", "question": "I opened the kettle and changed my mind. Can I return it?",
     "order_date": "2026-02-14", "subject": "opened", "qualifiers": [], "expected": "yes",
     "needed": ("returns-policy", 1, ["Opened products may be returned within 30 days of delivery, "
                                      "less a 15% restocking fee."])},
    {"id": "q4-unopened-window", "question": "How long do I have to return an unopened product?",
     "order_date": "2026-05-10", "subject": "unopened", "qualifiers": [], "expected": "yes",
     "needed": ("returns-policy", 2, ["Unopened products may be returned within 30 days of delivery"])},
    {"id": "q5-missing-parts", "question": "The mixer I opened is missing parts. Can I send it back?",
     "order_date": "2026-05-10", "subject": "opened", "qualifiers": ["missing parts"], "expected": "yes",
     "needed": ("returns-policy", 2, ["| Condition | Returnable | Window |",
                                      "| Missing parts or accessories | Yes | 30 days from delivery |"])},
    {"id": "q6-misspelled", "question": "can i retrun an opend blender thats defectve",
     "order_date": "2026-05-10", "subject": "opened", "qualifiers": ["defective"], "expected": "yes",
     "needed": ("returns-policy", 2, ["| Condition | Returnable | Window |",
                                      "| Defective or damaged on arrival | Yes | 90 days from delivery |"])},
    {"id": "q7-reason-code", "question": "Which reason code goes on a goodwill return?", "principal": "staff",
     "order_date": "2026-05-10", "subject": "goodwill", "qualifiers": [], "expected": "retrieval-only",
     "needed": ("staff-returns-playbook", 3, ["Record reason code GW-2 on the return."])},
    {"id": "q8-goodwill-customer", "question": "Can support approve a goodwill return of my opened product?",
     "order_date": "2026-05-10", "subject": "goodwill", "qualifiers": [], "expected": "insufficient",
     "needed": None},
]


def show(title):
    print("\n== " + title)


def build(out, chunker):
    name = "index_fixed.sqlite" if chunker.startswith("fixed") else "index_structured.sqlite"
    db = R.open_index(os.path.join(out, name))
    R.ingest(db, FIX, chunker)
    return db


def recall(result_top, item):
    """True when one retrieved chunk contains every needed span. A table row
    and its header in different chunks do not count: neither chunk can be read
    as evidence on its own."""
    if item["needed"] is None:
        return None
    doc, ver, snippets = item["needed"]
    spans = R.needed_spans(FIX, FILES[(doc, ver)], snippets)
    return any(all(R.covers([c], doc, ver, s) for s in spans) for c in result_top)


def needed_rank(ranked_chunks, item):
    for i, c in enumerate(ranked_chunks, start=1):
        if recall([c], item):
            return i
    return None


def run_item(index, item, **kw):
    principal = STAFF if item.get("principal") == "staff" else CUSTOMER
    res = index.search(item["question"], principal, item["order_date"], **kw)
    ev = R.assemble(res)
    ans = R.answer(ev, item["subject"], item["qualifiers"])
    sup = R.support_check(ans, ev, item["expected"])
    return res, ev, ans, sup


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", default="runs/ch07")
    out = ap.parse_args().output
    os.makedirs(out, exist_ok=True)
    summary = {}

    # 1. the second edition's cosine example
    show("Cosine example")
    q, a, b = [1, 0], [0.8, 0.2], [0.2, 0.9]
    ca, cb = R.cosine(q, a), R.cosine(q, b)
    print(f"cos(q, A) = {ca:.4f}   cos(q, B) = {cb:.4f}")
    summary["cosine_example"] = {"A": round(ca, 4), "B": round(cb, 4)}

    # 2. ingestion with lineage, two chunkers
    show("Ingestion")
    dbs = {"fixed": build(out, "fixed-200/20"), "structured": build(out, "structured-700")}
    lineage = {}
    for mode, db in dbs.items():
        rows = R.chunk_rows(db)
        lineage[mode] = rows
        v2 = [r for r in rows if r["doc_id"] == "returns-policy" and r["version"] == 2]
        print(f"{mode:10s} chunks={len(rows):3d}  returns-policy v2 chunks={len(v2)}")
        write_run(out, f"chunks_{mode}.json", rows)
    for r in lineage["fixed"]:
        if r["doc_id"] == "returns-policy" and r["version"] == 2:
            print(f"  fixed {r['chunk_id']:24s} chars {r['start']:4d}-{r['end']:4d} sha {r['content_hash']} "
                  f"| {r['text'][:38]!r} ... {r['text'][-30:]!r}")
    for r in lineage["structured"]:
        if r["doc_id"] == "returns-policy" and r["version"] == 2:
            print(f"  struct {r['chunk_id']:23s} chars {r['start']:4d}-{r['end']:4d} sha {r['content_hash']} "
                  f"| {r['section']}")
    summary["chunk_counts"] = {m: len(v) for m, v in lineage.items()}

    idx = {m: R.Index(db) for m, db in dbs.items()}

    # 3. the missed answer, then the fix
    for mode in ("fixed", "structured"):
        show(f"Q1 with {mode} chunks, customer, order 2026-05-10")
        res, ev, ans, sup = run_item(idx[mode], EVAL[0])
        for name, n in res["stages"]:
            print(f"  {name:40s} {n}")
        print("  rank  chunk_id                  bm25(r)        cos(r)        rrf     rerank")
        for row in res["ranking"]:
            print(f"  {row['final_rank']:>4}  {row['chunk_id']:24s} {row['bm25']:6.3f}({row['bm25_rank'] or '-'})"
                  f"  {row['cosine']:6.3f}({row['cosine_rank'] or '-'})  {row['rrf']:.5f}  {row['rerank']:.4f}")
        for e in ev:
            print(f"  [{e['cite']}] {e['citation']}")
            if e["unreadable_fragments"]:
                print(f"       unreadable fragments: {e['unreadable_fragments']}")
        print(f"  answer: {ans['text']}")
        print(f"  recall@3 of needed chunk: {recall(res['top'], EVAL[0])}   outcome: {sup['outcome']} ({sup['reason']})")
        write_run(out, f"q1_{mode}.json", {"stages": res["stages"], "ranking": res["ranking"],
                                           "evidence": ev, "answer": ans, "support": sup})

    # raising k does not repair a split table
    res8, ev8, ans8, sup8 = run_item(idx["fixed"], EVAL[0], k=8, pool=12)
    frags = [f for e in ev8 for f in e["unreadable_fragments"]]
    print(f"  fixed chunks with k=8: answer: {ans8['text']}  outcome={sup8['outcome']}")
    print(f"  unreadable table fragments in those 8 chunks: {frags}")
    summary["q1_fixed_k8"] = {"answer": ans8["text"], "fragments": frags}

    # 4. evaluation before and after
    show("Evaluation: recall@3 and answer outcome")
    evaluation = {}
    for mode in ("fixed", "structured"):
        rows = []
        for item in EVAL:
            res, ev, ans, sup = run_item(idx[mode], item)
            rows.append({"id": item["id"], "recall@3": recall(res["top"], item),
                         "verdict": ans["verdict"], "expected": item["expected"],
                         "outcome": sup["outcome"] if item["expected"] != "retrieval-only" else "n/a"})
        rec = [r["recall@3"] for r in rows if r["recall@3"] is not None]
        counts = {o: sum(r["outcome"] == o for r in rows) for o in ("supported", "declined", "wrong")}
        answered = sum(r["outcome"] != "n/a" for r in rows)
        evaluation[mode] = {"rows": rows, "recall@3": f"{sum(rec)}/{len(rec)}", "answers": answered, **counts}
        print(f"  {mode}: recall@3 {sum(rec)}/{len(rec)}  answers: supported {counts['supported']}, "
              f"declined {counts['declined']}, wrong {counts['wrong']} (of {answered})")
        for r in rows:
            print(f"    {r['id']:22s} recall@3={str(r['recall@3']):5s} verdict={r['verdict']:12s} "
                  f"expected={r['expected']:14s} outcome={r['outcome']}")
    # ranker comparison on the structured index: rank of the needed chunk
    show("Rank of the needed chunk by method (structured chunks)")
    comp = {}
    by = {r["chunk_id"]: r for r in lineage["structured"]}
    methods = ("keyword", "vector", "hybrid (RRF)", "hybrid + rerank")
    print("  " + f"{'question':22s}" + "".join(f"{m:>17s}" for m in methods))
    for item in EVAL:
        if item["needed"] is None:
            continue
        principal = STAFF if item.get("principal") == "staff" else CUSTOMER
        res = idx["structured"].search(item["question"], principal, item["order_date"])
        ranks = {m: needed_rank([by[c] for c in res["orders"][m]], item) for m in methods}
        comp[item["id"]] = ranks
        print("  " + f"{item['id']:22s}" + "".join(f"{str(ranks[m] or '-'):>17s}" for m in methods))
    for m in methods:
        hit = sum(1 for r in comp.values() if r[m] is not None and r[m] <= 3)
        print(f"  recall@3 {m:16s} {hit}/{len(comp)}")
        evaluation.setdefault("recall@3_by_method", {})[m] = f"{hit}/{len(comp)}"
    evaluation["needed_rank_by_method"] = comp
    write_run(out, "evaluation.json", evaluation)
    summary["evaluation"] = {m: {k: evaluation[m][k] for k in ("recall@3", "supported", "declined", "wrong")}
                             for m in ("fixed", "structured")}

    # what the vector stand-in measures: shared character 4-grams, plus collisions
    show("Vector stand-in sanity check")
    text = open(os.path.join(FIX, "returns-policy.v2.md"), encoding="utf-8").read()
    sect = text[text.index("## Opened products"):text.index("## Refunds")]
    sanity = {}
    for qtext in ("can i retrun an opend blender thats defectve",
                  "Can I get my money back for a kettle I unboxed last week?"):
        shared = sorted(set(R.grams(qtext)) & set(R.grams(sect)))
        c256 = R.cosine(R.hashvec(qtext, 256), R.hashvec(sect, 256))
        c1024 = R.cosine(R.hashvec(qtext, 1024), R.hashvec(sect, 1024))
        sanity[qtext] = {"shared_4grams": shared, "cos_256": round(c256, 4), "cos_1024": round(c1024, 4)}
        print(f"  {qtext!r}: shared 4-grams={len(shared)} {shared[:8]} cos@256={c256:.4f} cos@1024={c1024:.4f}")
    write_run(out, "vector_sanity.json", sanity)

    # 5. access control: filter before ranking
    show("Access filter")
    s = idx["structured"]
    filtered = s.search(Q1, CUSTOMER, "2026-05-10")
    unfiltered = s.search(Q1, CUSTOMER, "2026-05-10", apply_access=False)
    staff = s.search(Q1, STAFF, "2026-05-10")
    for label, res in (("customer, filter applied", filtered), ("customer, NO access filter", unfiltered),
                       ("staff principal, filter applied", staff)):
        print(f"  {label}: " + ", ".join(f"{r['final_rank']}. {r['chunk_id']} ({r['rerank']})"
                                          for r in res["ranking"][:3]))
    # the post-filter anti-pattern: rank everything, take top 3, then remove
    post = [c for c in unfiltered["top"] if c["audience"] in CUSTOMER["audiences"]]
    print(f"  post-filter (rank all, keep top 3, then drop unauthorized): {len(post)} of 3 remain: "
          + ", ".join(c["chunk_id"] for c in post))
    write_run(out, "access.json", {
        "filtered": {"stages": filtered["stages"], "top3": [r["chunk_id"] for r in filtered["ranking"][:3]]},
        "unfiltered": {"stages": unfiltered["stages"], "top3": [r["chunk_id"] for r in unfiltered["ranking"][:3]]},
        "staff": {"stages": staff["stages"], "top3": [r["chunk_id"] for r in staff["ranking"][:3]]},
        "post_filter_remaining": [c["chunk_id"] for c in post]})
    summary["access"] = {"filtered_top1": filtered["ranking"][0]["chunk_id"],
                         "unfiltered_top1": unfiltered["ranking"][0]["chunk_id"]}
    summary["q1_structured_stages"] = filtered["stages"]

    # 6. effective dates versus ingestion dates
    show("Effective date versus ingestion date")
    eff = {}
    for item in (EVAL[1], EVAL[2]):
        for rule in ("effective", "latest-ingested"):
            res, ev, ans, sup = run_item(s, item, date_rule=rule)
            versions = sorted({(c["doc_id"], c["version"]) for c in res["top"] if c["doc_id"] == "returns-policy"})
            print(f"  order {item['order_date']} rule={rule:15s} policy versions in evidence={versions} "
                  f"answer={ans['verdict']:4s} expected={item['expected']:4s} outcome={sup['outcome']}")
            eff[f"{item['id']}|{rule}"] = {"versions": versions, "answer": ans["text"],
                                           "outcome": sup["outcome"]}
    write_run(out, "effective_dates.json", eff)

    # 7. withdrawal propagates to chunks and to the evidence cache
    show("Withdrawal")
    hq = "Can I return an opened holiday gift?"
    before = s.search(hq, CUSTOMER, "2025-12-20")
    key = f"{hq}|{CUSTOMER['id']}|2025-12-20"
    s.db.execute("INSERT INTO evidence_cache VALUES (?,?,?)",
                 (key, json.dumps([c["chunk_id"] for c in before["top"]]), "2026-01-09"))
    print("  before: " + ", ".join(c["chunk_id"] for c in before["top"]))
    s.db.execute("UPDATE documents SET status='withdrawn', withdrawn_at='2026-01-10', "
                 "withdrawn_reason='published in error; extension never approved' "
                 "WHERE doc_id='holiday-returns-2025'")
    affected = [r[0] for r in s.db.execute("SELECT chunk_id FROM chunks WHERE doc_id='holiday-returns-2025'")]
    stale = [r[0] for r in s.db.execute("SELECT cache_key, chunk_ids FROM evidence_cache").fetchall()
             if any(c in json.loads(r[1]) for c in affected)]
    for k in stale:
        s.db.execute("DELETE FROM evidence_cache WHERE cache_key=?", (k,))
    s.db.execute("INSERT INTO events (kind, detail) VALUES (?,?)", ("withdraw", json.dumps(
        {"doc_id": "holiday-returns-2025", "chunks": affected, "cache_entries_invalidated": len(stale)})))
    s.db.commit()
    after = s.search(hq, CUSTOMER, "2025-12-20")
    print(f"  withdrawn doc chunks={affected} cache entries invalidated={len(stale)}")
    print("  after:  " + ", ".join(c["chunk_id"] for c in after["top"]))
    events = [dict(zip(("seq", "kind", "detail"), r)) for r in s.db.execute("SELECT * FROM events")]
    write_run(out, "withdrawal.json", {"before": [c["chunk_id"] for c in before["top"]],
                                       "after": [c["chunk_id"] for c in after["top"]],
                                       "events": events})

    # 8. embedding model change
    show("Embedding model change")
    refused = s.search(Q1, CUSTOMER, "2026-05-10", query_embedder="hashvec-4gram-2048/v2")
    print(f"  {refused['status']}: {refused['reason']} (chunks needing re-embedding: "
          f"{refused['chunks_needing_reindex']})")
    write_run(out, "embedder_mismatch.json", refused)

    # 9. lineage verification
    show("Lineage verification")
    ev = R.assemble(filtered)
    checks = []
    for e, c in zip(ev, filtered["top"]):
        ok, src = R.verify_lineage(FIX, FILES[(c["doc_id"], c["version"])], c)
        checks.append({"cite": e["cite"], "chunk_id": c["chunk_id"], "verified": ok,
                       "source_hash_now": src, "source_hash_indexed": c["source_hash"]})
        print(f"  [{e['cite']}] {c['chunk_id']:24s} slice hash matches={ok} "
              f"source {c['source_hash']} -> {src}")
    # edit the source in place (a typo fix that shifts offsets) without a new version
    text = open(os.path.join(FIX, "returns-policy.v2.md"), encoding="utf-8").read()
    edited = text.replace("Version 2. Applies", "Version 2 (corrected). Applies")
    c = filtered["top"][0]
    tampered = R.sha(edited[c["start"]:c["end"]]) == c["content_hash"]
    print(f"  after an in-place edit of returns-policy.v2.md: slice hash matches={tampered} "
          f"source {c['source_hash']} -> {R.sha(edited)}")
    write_run(out, "lineage_check.json", {"checks": checks, "in_place_edit_matches": tampered})

    # 10. agentic search through a tool versus pre-retrieval
    show("Agentic search")
    tools = ToolRegistry()
    principal = CUSTOMER          # trusted context: from the session, never from the model
    log = []

    def search_policy(query, order_date):
        res = s.search(query, principal, order_date)
        e = R.assemble(res)
        log.append({"query": query, "principal": principal["id"], "order_date": order_date,
                    "stages": res["stages"], "chunks": [c["chunk_id"] for c in res["top"]]})
        return {"status": "ok", "evidence": [{"cite": x["cite"], "citation": x["citation"],
                                              "text": c["text"]} for x, c in zip(e, res["top"])]}

    tools.register("search_policy",
                   "Search published support policies that apply to an order. Returns cited passages.",
                   {"type": "object", "properties": {"query": {"type": "string"},
                                                     "order_date": {"type": "string"}},
                    "required": ["query", "order_date"], "additionalProperties": False}, search_policy)
    model = ScriptedModel([
        {"role": "assistant", "content": "", "tool_calls": [
            {"id": "s1", "name": "search_policy",
             "arguments": {"query": "opened blender defective", "order_date": "2026-05-10",
                           "audience": "staff"}}]},
        {"role": "assistant", "content": "", "tool_calls": [
            {"id": "s2", "name": "search_policy",
             "arguments": {"query": "opened blender defective", "order_date": "2026-05-10"}}]},
        {"role": "assistant", "content": "", "tool_calls": [
            {"id": "s3", "name": "search_policy",
             "arguments": {"query": "defective on arrival return window", "order_date": "2026-05-10"}}]},
        {"role": "assistant", "content": "Yes. Opened products are normally not returnable, but a product "
                                         "that is defective or damaged on arrival can be returned within 90 "
                                         "days of delivery [E1].", "tool_calls": []},
    ])
    messages = [{"role": "system", "content": "Answer from cited policy evidence only."},
                {"role": "user", "content": Q1 + " Order placed 2026-05-10."}]
    final = run_agent(model, tools, messages)
    for m in messages:
        if m["role"] == "tool":
            c = m["content"]
            brief = c.get("message") if c["status"] != "ok" else [x["citation"] for x in c["evidence"]]
            print(f"  {m['tool_call_id']}: {c['status']} {brief}")
    print(f"  model calls={len(model.requests)} executed searches={len(tools.executions)} final: {final}")
    write_run(out, "agentic_trace.json", {"messages": messages, "search_log": log,
                                          "model_calls": len(model.requests)})

    # pre-retrieval: the runtime searches with the user's words before the first model call
    pre_q = Q1 + " Order placed 2026-05-10."
    res = s.search(pre_q, principal, "2026-05-10")
    ev = R.assemble(res)
    context = "\n\n".join(f"[{e['cite']}] {e['citation']}\n{c['text']}" for e, c in zip(ev, res["top"]))
    cite = next(e["cite"] for e in ev if e["chunk_id"] == "returns-policy@v2#2")
    pre_model = ScriptedModel([{"role": "assistant", "content": "Yes. A product that is defective or damaged "
                                f"on arrival can be returned within 90 days of delivery [{cite}].", "tool_calls": []}])
    pre_messages = [{"role": "system", "content": "Answer from cited policy evidence only.\n\nEvidence:\n" + context},
                    {"role": "user", "content": pre_q}]
    pre_final = run_agent(pre_model, ToolRegistry(), pre_messages)
    print(f"  pre-retrieval: model calls={len(pre_model.requests)} searches=1 "
          f"evidence={[e['chunk_id'] for e in ev]} final: {pre_final}")
    write_run(out, "preretrieval_trace.json", {"query": pre_q, "stages": res["stages"],
                                               "messages": pre_messages, "model_calls": len(pre_model.requests)})

    write_run(out, "summary.json", summary)
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
