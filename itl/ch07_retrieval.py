"""Retrieval pieces for Chapter 7.

Everything here is deterministic and uses the standard library only.

    ingest()            documents -> chunks with lineage, stored in sqlite
    fixed_chunks()      fixed-size character windows (the flawed chunker)
    structured_chunks() heading- and table-aware chunks (the revised chunker)
    Index.search()      filter first (access, status, effective date), then
                        BM25 + hashed-vector cosine, fused with reciprocal rank
                        fusion, then a deterministic reranker
    assemble()          evidence units with citations
    answer()            an extractive stand-in for the model's answer
    support_check()     does the cited evidence contain what the answer needs?

The vector representation is a hashed bag of character 4-grams. It stands in
for a learned embedding so that the pipeline has two different rankers to
fuse. It measures surface similarity of spelling, not meaning.
"""
import hashlib
import json
import math
import os
import re
import sqlite3

STOP = set("""a an and are as at be by can for from has have i if in is it its my of on or
that the this to was were will with our your you do does did not""".split())

EMBEDDER = "hashvec-4gram-1024/v1"
DIMS = 1024


def sha(text, n=12):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:n]


def stem(word):
    for suffix, repl, minlen in (("ies", "y", 5), ("ing", "", 6), ("ed", "", 5), ("s", "", 4)):
        if word.endswith(suffix) and len(word) >= minlen and not word.endswith("ss"):
            return word[: len(word) - len(suffix)] + repl
    return word


def terms(text):
    return [stem(w) for w in re.findall(r"[a-z0-9]+", text.lower()) if w not in STOP]


# ---------------------------------------------------------------- chunkers

def fixed_chunks(text, size=200, overlap=20):
    """Character windows, the same shape as a size/overlap splitter."""
    out = []
    step = size - overlap
    for start in range(0, len(text), step):
        end = min(len(text), start + size)
        out.append({"start": start, "end": end, "section": "(none)"})
        if end == len(text):
            break
    return out


def structured_chunks(text, max_chars=700):
    """One chunk per section. A table stays whole and stays with the paragraph
    that introduces it; a long section splits only at paragraph boundaries."""
    lines = text.splitlines(keepends=True)
    offsets, pos = [], 0
    for line in lines:
        offsets.append(pos)
        pos += len(line)
    path = []
    sections = []          # (section path, start, end)
    cur_start, cur_path = 0, "(preamble)"
    for i, line in enumerate(lines):
        m = re.match(r"^(#+)\s+(.*)", line)
        if m:
            if offsets[i] > cur_start and text[cur_start:offsets[i]].strip():
                sections.append((cur_path, cur_start, offsets[i]))
            level = len(m.group(1))
            path = path[: level - 1] + [m.group(2).strip()]
            cur_path = " > ".join(path)
            cur_start = offsets[i]
    if text[cur_start:].strip():
        sections.append((cur_path, cur_start, len(text)))
    out = []
    for spath, s, e in sections:
        if e - s <= max_chars:
            out.append({"start": s, "end": e, "section": spath})
            continue
        # split at blank lines, but never between a paragraph ending in ':'
        # and the table that follows it, and never inside a table
        blocks, bstart = [], s
        for m in re.finditer(r"\n\n", text[s:e]):
            cut = s + m.end()
            before = text[bstart:cut].rstrip()
            after = text[cut:e].lstrip()
            if before.endswith(":") or before.splitlines()[-1].startswith("|") and after.startswith("|"):
                continue
            blocks.append((bstart, cut))
            bstart = cut
        blocks.append((bstart, e))
        acc = None
        for bs, be in blocks:
            if acc and be - acc[0] > max_chars:
                out.append({"start": acc[0], "end": acc[1], "section": spath})
                acc = None
            acc = (acc[0], be) if acc else (bs, be)
        out.append({"start": acc[0], "end": acc[1], "section": spath})
    return out


# ------------------------------------------------------------- vectors

def grams(text):
    out = []
    for w in re.findall(r"[a-z0-9]+", text.lower()):
        if w in STOP:
            continue
        g = f"#{w}#"
        out += [g[i:i + 4] for i in range(max(1, len(g) - 3))]
    return out


def hashvec(text, dims=DIMS):
    v = [0.0] * dims
    for w in re.findall(r"[a-z0-9]+", text.lower()):
        if w in STOP:
            continue
        g = f"#{w}#"
        for i in range(max(1, len(g) - 3)):
            h = int(hashlib.sha1(g[i:i + 4].encode()).hexdigest(), 16)
            v[h % dims] += 1.0
    norm = math.sqrt(sum(x * x for x in v)) or 1.0
    return [round(x / norm, 6) for x in v]


def cosine(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb) if na and nb else 0.0


# ------------------------------------------------------------- ingestion

SCHEMA = """
CREATE TABLE documents (doc_id TEXT, version INTEGER, title TEXT, owner TEXT, audience TEXT,
  status TEXT, effective_from TEXT, effective_to TEXT, ingested_at TEXT, source_hash TEXT,
  withdrawn_at TEXT, withdrawn_reason TEXT, PRIMARY KEY (doc_id, version));
CREATE TABLE chunks (chunk_id TEXT PRIMARY KEY, doc_id TEXT, version INTEGER, chunker TEXT,
  seq INTEGER, section TEXT, start INTEGER, end INTEGER, content_hash TEXT, text TEXT,
  embedder TEXT, vector TEXT);
CREATE TABLE evidence_cache (cache_key TEXT PRIMARY KEY, chunk_ids TEXT, created_at TEXT);
CREATE TABLE events (seq INTEGER PRIMARY KEY AUTOINCREMENT, kind TEXT, detail TEXT);
"""


def open_index(path):
    if os.path.exists(path):
        os.remove(path)
    db = sqlite3.connect(path)
    db.executescript(SCHEMA)
    return db


def ingest(db, fixture_dir, chunker):
    manifest = json.load(open(os.path.join(fixture_dir, "manifest.json"), encoding="utf-8"))
    split = fixed_chunks if chunker == "fixed-200/20" else structured_chunks
    for doc in manifest:
        text = open(os.path.join(fixture_dir, doc["file"]), encoding="utf-8").read()
        source_hash = sha(text)
        db.execute("INSERT INTO documents VALUES (?,?,?,?,?,?,?,?,?,?,NULL,NULL)",
                   (doc["doc_id"], doc["version"], doc["title"], doc["owner"], doc["audience"],
                    doc["status"], doc["effective_from"], doc["effective_to"], doc["ingested_at"],
                    source_hash))
        for seq, c in enumerate(split(text)):
            body = text[c["start"]:c["end"]]
            cid = f"{doc['doc_id']}@v{doc['version']}#{seq}"
            db.execute("INSERT INTO chunks VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                       (cid, doc["doc_id"], doc["version"], chunker, seq, c["section"], c["start"],
                        c["end"], sha(body), body, EMBEDDER, json.dumps(hashvec(body))))
    db.execute("INSERT INTO events (kind, detail) VALUES (?,?)",
               ("ingest", json.dumps({"chunker": chunker, "documents": len(manifest)})))
    db.commit()


def chunk_rows(db):
    cur = db.execute("SELECT chunk_id, doc_id, version, seq, section, start, end, content_hash, chunker, "
                     "embedder, text FROM chunks ORDER BY doc_id, version, seq")
    keys = ["chunk_id", "doc_id", "version", "seq", "section", "start", "end", "content_hash", "chunker",
            "embedder", "text"]
    return [dict(zip(keys, r)) for r in cur.fetchall()]


# ------------------------------------------------------------- ranking

def bm25(query_terms, docs, k1=1.2, b=0.75):
    """docs: list of (chunk_id, term list). Returns {chunk_id: score} for score > 0."""
    n = len(docs)
    if n == 0:
        return {}
    avg = sum(len(t) for _, t in docs) / n
    df = {}
    for _, t in docs:
        for term in set(t):
            df[term] = df.get(term, 0) + 1
    out = {}
    for cid, t in docs:
        score = 0.0
        for q in set(query_terms):
            tf = t.count(q)
            if not tf:
                continue
            idf = math.log(1 + (n - df[q] + 0.5) / (df[q] + 0.5))
            score += idf * tf * (k1 + 1) / (tf + k1 * (1 - b + b * len(t) / avg))
        if score > 0:
            out[cid] = score
    return out


def rrf(rankings, k=60):
    fused = {}
    for ranking in rankings:
        for rank, cid in enumerate(ranking, start=1):
            fused[cid] = fused.get(cid, 0.0) + 1.0 / (k + rank)
    return fused


def rerank_score(query, text, section):
    """A deterministic stand-in for a cross-encoder: it reads query and chunk
    together. Coverage of distinct query terms, how close together they
    occur, and whether the section heading names them."""
    q = [t for t in dict.fromkeys(terms(query))]
    toks = terms(text)
    present = [t for t in q if t in toks]
    coverage = len(present) / len(q) if q else 0.0
    proximity = 0.0
    if len(present) >= 2:
        positions = {t: [i for i, x in enumerate(toks) if x == t] for t in present}
        best = None
        for anchor in positions[present[0]]:
            span_end = anchor
            span_start = anchor
            for t in present[1:]:
                nearest = min(positions[t], key=lambda p: abs(p - anchor))
                span_start, span_end = min(span_start, nearest), max(span_end, nearest)
            width = span_end - span_start + 1
            best = width if best is None else min(best, width)
        proximity = len(present) / best
    heading = sum(1 for t in q if t in terms(section)) / len(q) if q else 0.0
    return round(0.6 * coverage + 0.25 * proximity + 0.15 * heading, 4)


class Index:
    def __init__(self, db):
        self.db = db

    def candidates(self, principal, as_of, apply_access=True, apply_status=True, apply_dates=True,
                   date_rule="effective"):
        """Return the candidate chunk rows and a count after each filter.
        The filters are SQL predicates evaluated before any ranking."""
        stages = []
        base = ("SELECT c.chunk_id, c.doc_id, c.version, c.section, c.start, c.end, c.content_hash, "
                "c.text, c.vector, c.embedder, d.audience, d.status, d.effective_from, d.effective_to, "
                "d.ingested_at, d.source_hash FROM chunks c JOIN documents d "
                "ON c.doc_id = d.doc_id AND c.version = d.version")
        where, args = [], []
        total = self.db.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
        stages.append(("indexed chunks", total))
        if apply_access:
            where.append("d.audience IN (%s)" % ",".join("?" * len(principal["audiences"])))
            args += principal["audiences"]
            stages.append(("after access filter", self._count(base, where, args)))
        if apply_status:
            where.append("d.status = 'active'")
            stages.append(("after withdrawal filter", self._count(base, where, args)))
        if apply_dates and date_rule == "effective":
            where.append("d.effective_from <= ? AND (d.effective_to IS NULL OR d.effective_to >= ?)")
            args += [as_of, as_of]
            stages.append(("after effective-date filter", self._count(base, where, args)))
        elif apply_dates and date_rule == "latest-ingested":
            where.append("d.ingested_at = (SELECT MAX(d2.ingested_at) FROM documents d2 "
                         "WHERE d2.doc_id = d.doc_id AND d2.status = 'active')")
            stages.append(("after latest-ingested rule", self._count(base, where, args)))
        sql = base + (" WHERE " + " AND ".join(where) if where else "") + " ORDER BY c.chunk_id"
        keys = ["chunk_id", "doc_id", "version", "section", "start", "end", "content_hash", "text",
                "vector", "embedder", "audience", "status", "effective_from", "effective_to",
                "ingested_at", "source_hash"]
        rows = [dict(zip(keys, r)) for r in self.db.execute(sql, args).fetchall()]
        return rows, stages

    def _count(self, base, where, args):
        sql = "SELECT COUNT(*) FROM (" + base + " WHERE " + " AND ".join(where) + ")"
        return self.db.execute(sql, args).fetchone()[0]

    def search(self, query, principal, as_of, k=3, pool=8, query_embedder=EMBEDDER, **filters):
        rows, stages = self.candidates(principal, as_of, **filters)
        mismatched = sorted({r["embedder"] for r in rows} - {query_embedder})
        if mismatched:
            return {"status": "refused",
                    "reason": f"index vectors built with {mismatched[0]}; query encoder is "
                              f"{query_embedder}; re-index required",
                    "chunks_needing_reindex": len(rows), "stages": stages}
        by_id = {r["chunk_id"]: r for r in rows}
        qt = terms(query)
        kw = bm25(qt, [(r["chunk_id"], terms(r["text"])) for r in rows])
        kw_rank = sorted(kw, key=lambda c: (-kw[c], c))[:pool]
        qv = hashvec(query)
        dense = {r["chunk_id"]: cosine(qv, json.loads(r["vector"])) for r in rows}
        dense_rank = sorted(dense, key=lambda c: (-dense[c], c))[:pool]
        fused = rrf([kw_rank, dense_rank])
        fused_rank = sorted(fused, key=lambda c: (-fused[c], c))
        stages.append(("keyword candidates (BM25 > 0, top %d)" % pool, len(kw_rank)))
        stages.append(("vector candidates (top %d)" % pool, len(dense_rank)))
        stages.append(("fused candidates (union, RRF)", len(fused_rank)))
        rr = {c: rerank_score(query, by_id[c]["text"], by_id[c]["section"]) for c in fused_rank}
        reranked = sorted(fused_rank, key=lambda c: (-rr[c], -fused[c], c))
        top = reranked[:k]
        stages.append(("evidence after rerank (top %d)" % k, len(top)))
        table = []
        for c in reranked:
            table.append({"chunk_id": c, "section": by_id[c]["section"],
                          "bm25": round(kw.get(c, 0.0), 4),
                          "bm25_rank": kw_rank.index(c) + 1 if c in kw_rank else None,
                          "cosine": round(dense[c], 4),
                          "cosine_rank": dense_rank.index(c) + 1 if c in dense_rank else None,
                          "rrf": round(fused[c], 5), "fused_rank": fused_rank.index(c) + 1,
                          "rerank": rr[c], "final_rank": reranked.index(c) + 1})
        return {"status": "ok", "query": query, "principal": principal["id"], "as_of": as_of,
                "stages": stages, "ranking": table, "top": [by_id[c] for c in top],
                "keyword_only": kw_rank[:k], "vector_only": dense_rank[:k],
                "fused_only": fused_rank[:k],
                "orders": {"keyword": kw_rank, "vector": dense_rank, "hybrid (RRF)": fused_rank,
                           "hybrid + rerank": reranked}}


# ------------------------------------------------------------- evidence

def units(chunk_text):
    """Split chunk text into answerable units: sentences, and table rows that
    can be read because their header is present in the same chunk. A row
    whose header is missing is kept as an unreadable fragment."""
    out, fragments = [], []
    lines = chunk_text.splitlines()
    header = None
    for i, line in enumerate(lines):
        s = line.strip()
        if s.startswith("|"):
            cells = [c.strip() for c in s.strip("|").split("|")]
            if set("".join(cells)) <= set("-: "):
                continue
            nxt = lines[i + 1].strip() if i + 1 < len(lines) else ""
            if header is None and nxt.startswith("|") and set(nxt.replace("|", "")) <= set("-: "):
                header = cells
                continue
            if header and len(header) == len(cells):
                out.append({"kind": "row", "text": "; ".join(f"{h}: {c}" for h, c in zip(header, cells)),
                            "raw": s})
            else:
                fragments.append(s)
        else:
            header = None if not s else header
            for sent in re.split(r"(?<=[.])\s+", s):
                if sent and not sent.startswith("#"):
                    out.append({"kind": "sentence", "text": sent, "raw": sent})
    return out, fragments


def assemble(result):
    evidence = []
    for n, c in enumerate(result["top"], start=1):
        u, frags = units(c["text"])
        evidence.append({
            "cite": f"E{n}",
            "chunk_id": c["chunk_id"],
            "citation": f"{c['doc_id']} v{c['version']} | {c['section']} | chars {c['start']}-{c['end']} "
                        f"| sha {c['content_hash']}",
            "effective": f"{c['effective_from']} to {c['effective_to'] or 'open'}",
            "units": u, "unreadable_fragments": frags})
    return evidence


def verdict_of(unit_text):
    t = unit_text.lower()
    if "returnable: yes" in t or "may be returned" in t or "may approve" in t:
        return "yes"
    if "cannot" in t or "returnable: no" in t:
        return "no"
    return None


def answer(evidence, subject, qualifiers):
    """Extractive stand-in for the model. It picks the most specific unit that
    mentions the subject or a qualifier and states a verdict. A qualifier match
    beats a general rule. With no usable unit it declines."""
    best = None
    for e in evidence:
        for u in e["units"]:
            low = u["text"].lower()
            v = verdict_of(low)
            if v is None:
                continue
            q_hits = sum(1 for q in qualifiers if re.search(r"\b%s\b" % re.escape(q), low))
            s_hit = bool(re.search(r"\b%s\b" % re.escape(subject), low))
            if not (s_hit or q_hits):
                continue
            if qualifiers and not q_hits and u["kind"] == "row":
                continue
            score = (q_hits, s_hit, -int(e["cite"][1:]))
            if best is None or score > best[0]:
                best = (score, e, u, v)
    if best is None:
        return {"verdict": "insufficient", "text": "No authorized evidence settles this question; "
                                                    "escalate to a support agent.", "cites": []}
    _, e, u, v = best
    return {"verdict": v, "text": f"{'Yes' if v == 'yes' else 'No'}. {u['text']} [{e['cite']}]",
            "cites": [e["cite"]], "unit": u["text"]}


def covers(chunks, doc_id, version, span):
    for c in chunks:
        if c["doc_id"] == doc_id and c["version"] == version and c["start"] <= span[0] and c["end"] >= span[1]:
            return True
    return False


def needed_spans(fixture_dir, file, snippets):
    text = open(os.path.join(fixture_dir, file), encoding="utf-8").read()
    spans = []
    for s in snippets:
        i = text.index(s)
        spans.append((i, i + len(s)))
    return spans


def support_check(ans, evidence, expected):
    """Three outcomes. supported: the verdict matches gold and the unit it
    relies on appears verbatim in a cited chunk. declined: the answerer said the
    evidence does not settle the question. wrong: anything else."""
    cited = {e["cite"]: e for e in evidence}
    if ans["verdict"] == "insufficient":
        return {"outcome": "supported" if expected == "insufficient" else "declined",
                "reason": "no usable evidence unit"}
    unit_found = any(ans["unit"] == u["text"] for c in ans["cites"] for u in cited[c]["units"])
    ok = unit_found and ans["verdict"] == expected
    return {"outcome": "supported" if ok else "wrong",
            "reason": "verdict and cited unit agree with gold" if ok else
                      f"verdict {ans['verdict']} but gold is {expected}"}


def verify_lineage(fixture_dir, file, chunk):
    text = open(os.path.join(fixture_dir, file), encoding="utf-8").read()
    return sha(text[chunk["start"]:chunk["end"]]) == chunk["content_hash"], sha(text)
