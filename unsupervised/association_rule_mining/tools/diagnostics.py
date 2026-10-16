"""Read-only checks before mining: the shape of the baskets, how many patterns each support level would produce, and what dominates the co-occurrence."""

from __future__ import annotations

import numpy as np
import pandas as pd

from ml_core.tooling import tool
from unsupervised.association_rule_mining.tools._common import item_counts, load_transactions, mine


@tool
def profile_transactions(data: str, transaction_column: str | None = None, item_column: str | None = None, basket_column: str | None = None, separator: str = ",", onehot_columns: list[str] | None = None, top: int = 10) -> dict:
    """Profile a transaction table: number of baskets and distinct items, basket-size distribution, density (average basket size over item count), the most frequent items, items that occur once, duplicate baskets and single-item baskets (which contribute no co-occurrence).

    Args:
        transaction_column: Column identifying the transaction (long format, with `item_column`).
        item_column: Column holding one item per row (long format).
        basket_column: Alternatively, one column listing each transaction's items joined by `separator`.
        onehot_columns: Alternatively, 0/1 columns, one per item.
        top: Most frequent items listed.
    """
    baskets, _ = load_transactions(data, transaction_column, item_column, basket_column, separator, onehot_columns)
    counts = item_counts(baskets)
    sizes = np.array([len(b) for b in baskets])
    ser = pd.Series(counts).sort_values(ascending=False)
    return {"transactions": len(baskets), "distinct_items": len(counts), "basket_size": {"mean": round(float(sizes.mean()), 2), "median": float(np.median(sizes)), "p95": float(np.quantile(sizes, .95)), "max": int(sizes.max())},
            "density": round(float(sizes.mean() / max(len(counts), 1)), 4), "single_item_baskets": int((sizes == 1).sum()), "duplicate_baskets": int(len(baskets) - len(set(baskets))), "items_seen_once": int((ser == 1).sum()),
            "top_items": {i: {"count": int(c), "support": round(float(c / len(baskets)), 4)} for i, c in ser.head(top).items()},
            "advice": [a for a in ["very sparse data (density under 1%): use a low minimum support and expect most items to be too rare to rule on" if sizes.mean() / max(len(counts), 1) < 0.01 else None,
                                   "a few huge baskets can create thousands of spurious pairs: check check_basket_size_effects" if sizes.max() > 10 * max(np.median(sizes), 1) else None,
                                   "many single-item baskets carry no co-occurrence; they still count in the support denominator" if (sizes == 1).mean() > 0.3 else None] if a]}


@tool
def suggest_min_support(data: str, transaction_column: str | None = None, item_column: str | None = None, basket_column: str | None = None, separator: str = ",", onehot_columns: list[str] | None = None,
                        supports: list[float] | None = None, target_itemsets: int = 200, max_len: int = 4) -> dict:
    """Sweep the minimum support and report how many frequent itemsets (and of what size) each level produces, and which level gives roughly `target_itemsets` — the single most important setting in rule mining: too high finds only the obvious, too low drowns in noise and runs for hours.

    Args:
        supports: Support shares to try; defaults to a geometric ladder from 20% down to 0.1%.
        target_itemsets: How many itemsets a human can sensibly look at.
        max_len: Longest itemset considered.
    """
    baskets, _ = load_transactions(data, transaction_column, item_column, basket_column, separator, onehot_columns)
    rows = []
    for s in sorted(supports or [0.2, 0.1, 0.05, 0.03, 0.02, 0.01, 0.005, 0.002, 0.001], reverse=True):
        freq, mc = mine(baskets, s, "eclat", max_len)
        if len(freq) > 50000:
            rows.append({"min_support": s, "itemsets": f">{50000}", "note": "stopped: too many itemsets"})
            break
        sizes = pd.Series([len(i) for i in freq]).value_counts().sort_index()
        rows.append({"min_support": s, "min_count": mc, "itemsets": len(freq), "by_size": {int(k): int(v) for k, v in sizes.items()}})
    ok = [r for r in rows if isinstance(r["itemsets"], int)]
    pick = min(ok, key=lambda r: abs(r["itemsets"] - target_itemsets)) if ok else None
    return {"transactions": len(baskets), "sweep": rows, "suggested_min_support": pick["min_support"] if pick else None,
            "reading": "choose the level just before the count explodes; pairs and triples usually carry the actionable rules, so a sweep dominated by size-1 itemsets means support is too high"}


@tool
def check_item_frequency_distribution(data: str, transaction_column: str | None = None, item_column: str | None = None, basket_column: str | None = None, separator: str = ",", onehot_columns: list[str] | None = None, min_support: float = 0.01) -> dict:
    """How concentrated are the purchases? The share of all item occurrences taken by the top 1, 5 and 20 items, the share of items that clear the minimum support, and a Gini coefficient — a long tail means a global support threshold will discard most of the catalogue, and rules should be mined per segment or on categories.

    Args:
        min_support: Support share an item needs to be considered.
    """
    baskets, _ = load_transactions(data, transaction_column, item_column, basket_column, separator, onehot_columns)
    c = pd.Series(item_counts(baskets)).sort_values(ascending=False)
    share = c / c.sum()
    gini = float(1 - 2 * np.sum(np.cumsum(np.sort(share.to_numpy())) / len(share)) + 1 / len(share))
    kept = int((c / len(baskets) >= min_support).sum())
    return {"distinct_items": len(c), "top_1_share": round(float(share.head(1).sum()), 4), "top_5_share": round(float(share.head(5).sum()), 4), "top_20_share": round(float(share.head(20).sum()), 4), "gini": round(gini, 4),
            "items_clearing_min_support": kept, "share_of_catalogue_clearing_min_support": round(kept / len(c), 4),
            "advice": "a long tail: most items will never reach the support threshold — roll items up to categories (mine_rules_by_category) or lower the threshold for a targeted item" if kept / len(c) < 0.2 else "the catalogue is reasonably concentrated"}


@tool
def check_basket_size_effects(data: str, transaction_column: str | None = None, item_column: str | None = None, basket_column: str | None = None, separator: str = ",", onehot_columns: list[str] | None = None, top_share: float = 0.01) -> dict:
    """How much of the co-occurrence comes from a few huge baskets? Counts item pairs contributed by the largest baskets (n·(n−1)/2 per basket) — wholesale or bulk orders create thousands of 'bought together' pairs that say nothing about ordinary shoppers.

    Args:
        top_share: Share of baskets (the largest) treated as 'huge'.
    """
    baskets, _ = load_transactions(data, transaction_column, item_column, basket_column, separator, onehot_columns)
    sizes = np.array([len(b) for b in baskets])
    pairs = sizes * (sizes - 1) / 2
    k = max(1, int(len(baskets) * top_share))
    big = np.sort(pairs)[::-1][:k].sum()
    return {"baskets": len(baskets), "pairs_total": int(pairs.sum()), "share_of_pairs_from_largest_baskets": round(float(big / max(pairs.sum(), 1)), 4), "largest_basket": int(sizes.max()), "median_basket": float(np.median(sizes)),
            "advice": "a few huge baskets dominate the pairs: cap basket size (drop or split baskets above a limit) before mining" if big / max(pairs.sum(), 1) > 0.3 else "no basket-size distortion"}


@tool
def estimate_pattern_space(data: str, transaction_column: str | None = None, item_column: str | None = None, basket_column: str | None = None, separator: str = ",", onehot_columns: list[str] | None = None, min_support: float = 0.01) -> dict:
    """Before mining at a low support, estimate the cost: the number of frequent items and pairs (counted exactly) and the worst-case number of candidate triples — if the frequent pairs alone run into the hundreds of thousands, raise the support or restrict the items.

    Args:
        min_support: Support share to estimate at.
    """
    baskets, _ = load_transactions(data, transaction_column, item_column, basket_column, separator, onehot_columns)
    min_count = max(2, int(np.ceil(min_support * len(baskets))))
    c = item_counts(baskets)
    frequent = {i for i, v in c.items() if v >= min_count}
    pair: dict[tuple[str, str], int] = {}
    for b in baskets:
        items = sorted(i for i in b if i in frequent)
        for x in range(len(items)):
            for y in range(x + 1, len(items)):
                pair[(items[x], items[y])] = pair.get((items[x], items[y]), 0) + 1
    fp = sum(1 for v in pair.values() if v >= min_count)
    return {"min_count": min_count, "frequent_items": len(frequent), "frequent_pairs": fp, "candidate_triples_upper_bound": int(fp * max(len(frequent) - 2, 0) / 3),
            "verdict": "cheap" if fp < 20000 else "heavy: expect long runtimes and an unmanageable number of rules — raise the support or mine per category"}
