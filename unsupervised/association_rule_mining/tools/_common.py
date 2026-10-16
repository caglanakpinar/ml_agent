"""Association-rule machinery the tool modules share, in pure Python so no mining library is needed:

  load_transactions   one table → a list of baskets, from long format, a delimited column, or one-hot columns
  apriori / eclat / fpgrowth   three frequent-itemset miners that must (and, in the tests, do) return identical itemsets
  closed / maximal    condensed representations of the frequent itemsets
  rules_from          every rule an itemset supports, with the standard interestingness measures
  prefixspan          frequent ordered subsequences

Itemsets are `frozenset`s of item names; supports are counts (divide by the number of baskets for a share)."""

from __future__ import annotations

from collections import defaultdict
from itertools import combinations
from typing import Any

import numpy as np
import pandas as pd

from ml_core.data import as_list, load_frame, require_columns


# ---- reading ------------------------------------------------------------------------------------
def load_transactions(data: Any, transaction_column: str | None = None, item_column: str | None = None, basket_column: str | None = None, separator: str = ",",
                      onehot_columns: Any = None, min_items: int = 1) -> tuple[list[frozenset[str]], list[Any]]:
    """Baskets as frozensets, and the matching transaction ids. Three input shapes: one row per (transaction, item) — `transaction_column` + `item_column`;
    one row per transaction with the items in one delimited cell — `basket_column`; one row per transaction with a 0/1 column per item — `onehot_columns`
    (every 0/1 column when the list is empty and neither of the others is given)."""
    frame = load_frame(data)
    if transaction_column and item_column:
        require_columns(frame, [transaction_column, item_column])
        groups = frame.dropna(subset=[item_column]).groupby(transaction_column, sort=False)[item_column].apply(lambda s: frozenset(str(v).strip() for v in s))
        ids, baskets = list(groups.index), list(groups.values)
    elif basket_column:
        require_columns(frame, [basket_column])
        cells = frame[basket_column].fillna("").astype(str)
        baskets = [frozenset(p.strip() for p in c.split(separator) if p.strip()) for c in cells]
        ids = list(frame.index)
    else:
        cols = as_list(onehot_columns) or [c for c in frame.columns if set(pd.unique(frame[c].dropna())) <= {0, 1, True, False}]
        if not cols:
            raise ValueError("no transaction layout recognised: give `transaction_column` + `item_column`, or `basket_column`, or `onehot_columns`")
        require_columns(frame, cols)
        baskets = [frozenset(c for c in cols if row[c]) for _, row in frame[cols].astype(bool).iterrows()]
        ids = list(frame.index)
    keep = [i for i, b in enumerate(baskets) if len(b) >= min_items]
    return [baskets[i] for i in keep], [ids[i] for i in keep]


def label(itemset: frozenset[str]) -> str:
    return " + ".join(sorted(itemset))


# ---- miners -------------------------------------------------------------------------------------
def item_counts(baskets: list[frozenset[str]]) -> dict[str, int]:
    c: dict[str, int] = defaultdict(int)
    for b in baskets:
        for i in b:
            c[i] += 1
    return c


def eclat(baskets: list[frozenset[str]], min_count: int, max_len: int | None = None) -> dict[frozenset[str], int]:
    """Vertical depth-first mining: each item has a bitset of the transactions containing it, and an itemset's support is the popcount of the AND."""
    tids: dict[str, int] = defaultdict(int)
    for t, b in enumerate(baskets):
        bit = 1 << t
        for i in b:
            tids[i] |= bit
    frequent = sorted(((i, m) for i, m in tids.items() if m.bit_count() >= min_count), key=lambda im: im[1].bit_count())
    out: dict[frozenset[str], int] = {}

    def grow(prefix: tuple[str, ...], items: list[tuple[str, int]]) -> None:
        for k, (item, mask) in enumerate(items):
            new = prefix + (item,)
            out[frozenset(new)] = mask.bit_count()
            if max_len and len(new) >= max_len:
                continue
            ext = [(o, mask & om) for o, om in items[k + 1 :] if (mask & om).bit_count() >= min_count]
            if ext:
                grow(new, ext)

    grow((), frequent)
    return out


def apriori(baskets: list[frozenset[str]], min_count: int, max_len: int | None = None) -> dict[frozenset[str], int]:
    """Level-wise mining: a (k+1)-itemset can only be frequent if all its k-subsets are, so candidates are joined from frequent k-itemsets and pruned before counting."""
    tids: dict[str, int] = defaultdict(int)
    for t, b in enumerate(baskets):
        for i in b:
            tids[i] |= 1 << t
    level = {(i,): m for i, m in sorted(tids.items()) if m.bit_count() >= min_count}
    out = {frozenset(k): m.bit_count() for k, m in level.items()}
    k = 1
    while level and (not max_len or k < max_len):
        keys = sorted(level)
        known = set(keys)
        nxt: dict[tuple[str, ...], int] = {}
        for a, b in combinations(keys, 2):
            if a[:-1] != b[:-1]:
                continue
            cand = a + (b[-1],)
            if all(cand[:j] + cand[j + 1 :] in known for j in range(len(cand) - 2)):  # Apriori pruning
                mask = level[a] & level[b]
                if mask.bit_count() >= min_count:
                    nxt[cand] = mask
        out.update({frozenset(c): m.bit_count() for c, m in nxt.items()})
        level, k = nxt, k + 1
    return out


class _Node:
    __slots__ = ("item", "count", "parent", "children", "link")

    def __init__(self, item: str | None, parent: "_Node | None"):
        self.item, self.count, self.parent, self.children, self.link = item, 0, parent, {}, None


def fpgrowth(baskets: list[frozenset[str]], min_count: int, max_len: int | None = None) -> dict[frozenset[str], int]:
    """FP-growth: compress the baskets into a prefix tree ordered by item frequency, then mine it recursively through conditional pattern bases — no candidate generation."""
    counts = item_counts(baskets)
    order = {i: (-c, i) for i, c in counts.items() if c >= min_count}

    def build(weighted: list[tuple[list[str], int]]) -> tuple[_Node, dict[str, _Node], dict[str, int]]:
        root, header, support = _Node(None, None), {}, defaultdict(int)
        for items, w in weighted:
            node = root
            for it in items:
                if it not in node.children:
                    child = _Node(it, node)
                    node.children[it] = child
                    child.link, header[it] = header.get(it), child
                node = node.children[it]
                node.count += w
                support[it] += w
        return root, header, support

    base = [(sorted((i for i in b if i in order), key=order.get), 1) for b in baskets]
    out: dict[frozenset[str], int] = {}

    def mine(weighted: list[tuple[list[str], int]], suffix: tuple[str, ...]) -> None:
        _, header, support = build(weighted)
        for item in sorted(support, key=lambda i: support[i]):
            if support[item] < min_count:
                continue
            new = suffix + (item,)
            out[frozenset(new)] = support[item]
            if max_len and len(new) >= max_len:
                continue
            cond = []
            node = header[item]
            while node is not None:
                path, p = [], node.parent
                while p is not None and p.item is not None:
                    path.append(p.item)
                    p = p.parent
                if path:
                    cond.append((path[::-1], node.count))
                node = node.link
            freq = defaultdict(int)
            for path, w in cond:
                for it in path:
                    freq[it] += w
            cond = [([i for i in path if freq[i] >= min_count], w) for path, w in cond]
            if any(p for p, _ in cond):
                mine(cond, new)

    mine(base, ())
    return out


def _by_size(freq: dict[frozenset[str], int]) -> dict[int, list[frozenset[str]]]:
    out: dict[int, list[frozenset[str]]] = defaultdict(list)
    for s in freq:
        out[len(s)].append(s)
    return out


def closed_itemsets(freq: dict[frozenset[str], int]) -> dict[frozenset[str], int]:
    """Frequent itemsets with no superset of equal support — they carry all the information of the full set, usually in far fewer rows."""
    sizes = _by_size(freq)
    return {s: c for s, c in freq.items() if not any(s < t and freq[t] == c for t in sizes.get(len(s) + 1, []))}


def maximal_itemsets(freq: dict[frozenset[str], int]) -> dict[frozenset[str], int]:
    """Frequent itemsets with no frequent superset — the boundary of the frequent region."""
    sizes = _by_size(freq)
    return {s: c for s, c in freq.items() if not any(s < t for t in sizes.get(len(s) + 1, []))}


MINERS = {"apriori": apriori, "eclat": eclat, "fpgrowth": fpgrowth}


def mine(baskets: list[frozenset[str]], min_support: float, algorithm: str = "fpgrowth", max_len: int | None = None) -> tuple[dict[frozenset[str], int], int]:
    min_count = max(2, int(np.ceil(min_support * len(baskets))))
    return MINERS[algorithm](baskets, min_count, max_len), min_count


# ---- rules --------------------------------------------------------------------------------------
def rules_from(freq: dict[frozenset[str], int], n: int, min_confidence: float = 0.0, min_lift: float = 0.0, max_antecedent: int | None = None, consequent: frozenset[str] | None = None,
               max_consequent: int = 1) -> pd.DataFrame:
    """Every rule A → B with A ∪ B frequent and A, B disjoint, with support, confidence, lift, leverage, conviction, Zhang's metric and the Kulczynski measure."""
    rows = []
    for s, c_ab in freq.items():
        if len(s) < 2:
            continue
        for k in range(1, min(len(s), max_consequent + 1)):
            for cons in combinations(sorted(s), k):
                cons = frozenset(cons)
                if consequent and cons != consequent:
                    continue
                ante = s - cons
                if max_antecedent and len(ante) > max_antecedent:
                    continue
                c_a, c_b = freq.get(ante), freq.get(cons)
                if not c_a or not c_b:
                    continue
                sup, conf = c_ab / n, c_ab / c_a
                lift = conf / (c_b / n)
                if conf < min_confidence or lift < min_lift:
                    continue
                lev = sup - (c_a / n) * (c_b / n)
                conv = (1 - c_b / n) / (1 - conf) if conf < 1 else np.inf
                zhang = lev / max(sup * (1 - c_a / n), (c_a / n) * (c_b / n - sup), 1e-12)
                kulc = 0.5 * (c_ab / c_a + c_ab / c_b)
                rows.append({"antecedent": label(ante), "consequent": label(cons), "antecedent_size": len(ante), "support_count": c_ab, "support": round(sup, 5), "antecedent_support": round(c_a / n, 5), "consequent_support": round(c_b / n, 5),
                             "confidence": round(conf, 5), "lift": round(lift, 4), "leverage": round(lev, 5), "conviction": round(float(conv), 4) if np.isfinite(conv) else float("inf"), "zhang": round(float(zhang), 4), "kulczynski": round(kulc, 4)})
    return pd.DataFrame(rows)


def freq_table(freq: dict[frozenset[str], int], n: int) -> pd.DataFrame:
    return pd.DataFrame([{"itemset": label(s), "size": len(s), "count": c, "support": round(c / n, 5)} for s, c in sorted(freq.items(), key=lambda kv: (-kv[1], label(kv[0])))])


def prefixspan(sequences: list[list[str]], min_count: int, max_len: int = 5) -> list[tuple[tuple[str, ...], int]]:
    """Frequent ordered subsequences (items need not be adjacent): project the database on each frequent item and recurse on what follows it."""
    out: list[tuple[tuple[str, ...], int]] = []

    def grow(prefix: tuple[str, ...], projected: list[list[str]]) -> None:
        counts: dict[str, int] = defaultdict(int)
        for seq in projected:
            for i in set(seq):
                counts[i] += 1
        for item, c in sorted(counts.items()):
            if c < min_count:
                continue
            new = prefix + (item,)
            out.append((new, c))
            if len(new) < max_len:
                grow(new, [seq[seq.index(item) + 1 :] for seq in projected if item in seq])

    grow((), sequences)
    return out
