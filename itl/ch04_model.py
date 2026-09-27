"""Chapter 4 helpers: a byte-pair tokenizer, list-based matrices, one causal
attention head, next-token sampling and KV-cache bookkeeping.

Pure Python, standard library only. Every random choice uses an explicit
random.Random(seed), so runs are repeatable on the same Python version.
Nothing here is a trained model: weights are seeded random numbers and
decoding logits are authored. The point is to make shapes, masking and
sampling inspectable, not to predict what a real model would do.
"""
import json
import math
import random
import re

# ---------------------------------------------------------------------------
# Byte-pair encoding
# ---------------------------------------------------------------------------

# Pre-tokenization: split text into chunks that merges may not cross.
# Letters (with one optional leading space), runs of up to three digits,
# runs of punctuation, and whitespace. Similar in spirit to the regex
# splitters of production byte-level BPE tokenizers, much simpler.
PRETOKEN = re.compile(r" ?[A-Za-z_]+| ?[0-9]{1,3}| ?[^\sA-Za-z0-9_]+|\s+")


def pretokenize(text):
    return PRETOKEN.findall(text)


class BPETokenizer:
    """Byte-level BPE: the base vocabulary is the 256 byte values; each
    learned merge adds one token made of two existing tokens."""

    def __init__(self):
        self.merges = []            # list of ((a, b), new_id) in training order
        self.vocab = {i: bytes([i]) for i in range(256)}

    @staticmethod
    def _pair_counts(words):
        counts = {}
        for seq, freq in words.items():
            for a, b in zip(seq, seq[1:]):
                counts[(a, b)] = counts.get((a, b), 0) + freq
        return counts

    @staticmethod
    def _merge(seq, pair, new_id):
        out, i = [], 0
        while i < len(seq):
            if i < len(seq) - 1 and (seq[i], seq[i + 1]) == pair:
                out.append(new_id)
                i += 2
            else:
                out.append(seq[i])
                i += 1
        return tuple(out)

    def train(self, text, num_merges):
        words = {}
        for chunk in pretokenize(text):
            key = tuple(chunk.encode("utf-8"))
            words[key] = words.get(key, 0) + 1
        for _ in range(num_merges):
            counts = self._pair_counts(words)
            if not counts:
                break
            # Most frequent pair; ties broken by the smaller pair of ids,
            # so training is deterministic.
            pair = min(counts, key=lambda p: (-counts[p], p))
            if counts[pair] < 2:
                break
            new_id = 256 + len(self.merges)
            self.merges.append((pair, new_id))
            self.vocab[new_id] = self.vocab[pair[0]] + self.vocab[pair[1]]
            words = {self._merge(s, pair, new_id): f for s, f in words.items()}
        return self

    def encode(self, text):
        ids = []
        rank = {pair: i for i, (pair, _) in enumerate(self.merges)}
        new_of = {pair: nid for pair, nid in self.merges}
        for chunk in pretokenize(text):
            seq = tuple(chunk.encode("utf-8"))
            while len(seq) > 1:
                pairs = [(rank[p], p) for p in zip(seq, seq[1:]) if p in rank]
                if not pairs:
                    break
                _, best = min(pairs)
                seq = self._merge(seq, best, new_of[best])
            ids.extend(seq)
        return ids

    def piece(self, tid):
        return self.vocab[tid].decode("utf-8", errors="replace")

    def pieces(self, ids):
        return [self.piece(t) for t in ids]

    def decode(self, ids):
        return b"".join(self.vocab[t] for t in ids).decode("utf-8", errors="replace")

    def to_json(self):
        return {"base_vocabulary": 256, "merges": [
            {"id": nid, "left": a, "right": b, "piece": self.piece(nid)}
            for (a, b), nid in self.merges]}


# ---------------------------------------------------------------------------
# Matrices as lists of rows
# ---------------------------------------------------------------------------

def shape(m):
    return [len(m), len(m[0]) if m else 0]


def matmul(a, b):
    assert len(a[0]) == len(b), f"cannot multiply {shape(a)} by {shape(b)}"
    bt = list(zip(*b))
    return [[sum(x * y for x, y in zip(row, col)) for col in bt] for row in a]


def transpose(m):
    return [list(r) for r in zip(*m)]


def add(a, b):
    return [[x + y for x, y in zip(r, s)] for r, s in zip(a, b)]


def rand_matrix(rng, rows, cols, scale=0.5):
    return [[round(rng.uniform(-scale, scale), 3) for _ in range(cols)] for _ in range(rows)]


def softmax(row):
    finite = [x for x in row if x != float("-inf")]
    top = max(finite)
    exps = [0.0 if x == float("-inf") else math.exp(x - top) for x in row]
    total = sum(exps)
    return [e / total for e in exps]


def rounded(m, nd=4):
    if isinstance(m, dict):
        return {k: rounded(v, nd) for k, v in m.items()}
    if isinstance(m, str):
        return m
    if isinstance(m, list):
        return [rounded(x, nd) for x in m]
    if m == float("-inf"):
        return "-inf"
    return round(m, nd)


def sinusoidal_positions(n, d):
    """Fixed position vectors from the original transformer paper."""
    out = []
    for pos in range(n):
        row = []
        for i in range(d):
            angle = pos / (10000 ** (2 * (i // 2) / d))
            row.append(math.sin(angle) if i % 2 == 0 else math.cos(angle))
        out.append(row)
    return out


# ---------------------------------------------------------------------------
# One attention head
# ---------------------------------------------------------------------------

def causal_mask(n):
    """0 where position j is visible from position i (j <= i), -inf otherwise."""
    return [[0.0 if j <= i else float("-inf") for j in range(n)] for i in range(n)]


def attention(Q, K, V, masked=True):
    d_k = len(K[0])
    scores = [[s / math.sqrt(d_k) for s in row] for row in matmul(Q, transpose(K))]
    n_q, n_k = len(Q), len(K)
    if masked:
        # Queries are the last n_q positions of an n_k-long sequence.
        offset = n_k - n_q
        mask = [[0.0 if j <= i + offset else float("-inf") for j in range(n_k)] for i in range(n_q)]
    else:
        mask = [[0.0] * n_k for _ in range(n_q)]
    masked_scores = add(scores, mask)
    weights = [softmax(r) for r in masked_scores]
    out = matmul(weights, V)
    return {"scores": scores, "mask": mask, "masked_scores": masked_scores,
            "weights": weights, "output": out}


class Head:
    def __init__(self, seed, d_model, d_head):
        rng = random.Random(seed)
        self.Wq = rand_matrix(rng, d_model, d_head)
        self.Wk = rand_matrix(rng, d_model, d_head)
        self.Wv = rand_matrix(rng, d_model, d_head)
        self.Wo = rand_matrix(rng, d_head, d_model)

    def forward(self, X, masked=True):
        Q, K, V = matmul(X, self.Wq), matmul(X, self.Wk), matmul(X, self.Wv)
        a = attention(Q, K, V, masked)
        a.update({"Q": Q, "K": K, "V": V, "projected": matmul(a["output"], self.Wo)})
        return a


class KVCache:
    """Keys and values for positions already processed by one head."""

    def __init__(self):
        self.K, self.V = [], []

    def __len__(self):
        return len(self.K)


def prefill(head, X, cache, counter):
    Q, K, V = matmul(X, head.Wq), matmul(X, head.Wk), matmul(X, head.Wv)
    counter["kv_rows_projected"] += len(X)
    cache.K.extend(K)
    cache.V.extend(V)
    a = attention(Q, cache.K, cache.V, masked=True)
    counter["score_entries"] += len(Q) * len(cache.K)
    return a["output"][-1]


def decode_step(head, x_row, cache, counter):
    X = [x_row]
    q, k, v = matmul(X, head.Wq), matmul(X, head.Wk), matmul(X, head.Wv)
    counter["kv_rows_projected"] += 1
    cache.K.extend(k)
    cache.V.extend(v)
    a = attention(q, cache.K, cache.V, masked=True)
    counter["score_entries"] += len(cache.K)
    return a["output"][-1]


def full_recompute_last(head, X, counter):
    """What a generate loop without a cache does: rerun the whole prefix."""
    a = head.forward(X, masked=True)
    counter["kv_rows_projected"] += len(X)
    counter["score_entries"] += len(X) * len(X)
    return a["output"][-1]


def kv_bytes(layers, tokens, kv_heads, head_dim, bytes_per_element):
    return 2 * layers * tokens * kv_heads * head_dim * bytes_per_element


# ---------------------------------------------------------------------------
# Decoding
# ---------------------------------------------------------------------------

def apply_temperature(logits, t):
    return [x / t for x in logits]


def top_k_filter(logits, k):
    kth = sorted(logits, reverse=True)[min(k, len(logits)) - 1]
    return [x if x >= kth else float("-inf") for x in logits]


def top_p_filter(logits, p):
    """Keep the smallest set of highest-probability candidates whose
    cumulative probability reaches p (nucleus sampling)."""
    probs = softmax(logits)
    order = sorted(range(len(logits)), key=lambda i: -probs[i])
    keep, total = set(), 0.0
    for i in order:
        keep.add(i)
        total += probs[i]
        if total >= p:
            break
    return [x if i in keep else float("-inf") for i, x in enumerate(logits)]


def choose(logits, setting, rng):
    """Return (index, final probabilities) for one decoding step."""
    if setting["mode"] == "greedy":
        probs = softmax(logits)
        best = max(range(len(logits)), key=lambda i: (logits[i], -i))
        return best, probs
    x = apply_temperature(logits, setting.get("temperature", 1.0))
    if "top_k" in setting:
        x = top_k_filter(x, setting["top_k"])
    if "top_p" in setting:
        x = top_p_filter(x, setting["top_p"])
    probs = softmax(x)
    r, acc = rng.random(), 0.0
    for i, pr in enumerate(probs):
        acc += pr
        if r < acc:
            return i, probs
    return max(i for i, pr in enumerate(probs) if pr > 0), probs


def dumps(obj):
    return json.dumps(obj, indent=2, ensure_ascii=False)
