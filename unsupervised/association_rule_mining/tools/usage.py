"""Use the rules: recommend items for a basket, find bundles and cross-sell pairs, summarise how items relate, and export the rule set."""

from __future__ import annotations

import numpy as np
import pandas as pd

from ml_core.data import load_frame, require_columns, save_frame
from ml_core.tooling import tool
from unsupervised.association_rule_mining.tools._common import item_counts, load_transactions, mine


@tool
def recommend_items(rules_file: str, basket: list[str], top: int = 10, rank_by: str = "lift", min_confidence: float = 0.0) -> dict:
    """Recommend items for a basket from a rules file: every rule whose whole antecedent is already in the basket fires, and its consequent (if not already in the basket) is a candidate, scored by the chosen measure — the best rule per candidate wins. The classic 'customers who bought these also bought…'.

    Args:
        rules_file: CSV of rules written by `generate_rules`.
        basket: Items already in the basket.
        top: Recommendations returned.
        rank_by: lift, confidence, leverage, conviction, support, zhang or kulczynski.
        min_confidence: Smallest confidence for a rule to fire.
    """
    r = load_frame(rules_file)
    require_columns(r, ["antecedent", "consequent", rank_by, "confidence"])
    have = {str(b).strip() for b in basket}
    rows = []
    for _, rule in r.iterrows():
        if rule["confidence"] < min_confidence:
            continue
        A = set(str(rule["antecedent"]).split(" + "))
        C = set(str(rule["consequent"]).split(" + "))
        if A <= have and not C <= have:
            rows.append({"recommend": " + ".join(sorted(C - have)), "because": " + ".join(sorted(A)), rank_by: rule[rank_by], "confidence": rule["confidence"], "lift": rule.get("lift")})
    out = pd.DataFrame(rows)
    if out.empty:
        return {"basket": sorted(have), "recommendations": [], "note": "no rule fires for this basket — it contains no complete antecedent; lower the support when generating rules, or fall back to the most popular items"}
    out = out.sort_values(rank_by, ascending=False).drop_duplicates("recommend").head(top)
    return {"basket": sorted(have), "recommendations": out}


@tool
def find_bundles(data: str, min_support: float = 0.01, min_lift: float = 1.5, size: int = 2, transaction_column: str | None = None, item_column: str | None = None, basket_column: str | None = None, separator: str = ",",
                 onehot_columns: list[str] | None = None, top: int = 20, output_path: str | None = None) -> dict:
    """Candidate product bundles: itemsets of the chosen size whose members occur together more than their individual popularity predicts (lift of the whole itemset against independence, and the weakest pairwise lift inside it, so one strong pair cannot carry a weak third item). Ranked by the weakest link.

    Args:
        min_support: Share of transactions the bundle must appear in.
        min_lift: Smallest lift against independence.
        size: Items per bundle (2 or 3 is practical).
        transaction_column: Column identifying the transaction (long format, with `item_column`).
        item_column: Column holding one item per row (long format).
        basket_column: Alternatively, one column listing each transaction's items joined by `separator`.
        onehot_columns: Alternatively, 0/1 columns, one per item.
        top: Bundles listed.
        output_path: Where to write the bundles.
    """
    from itertools import combinations

    baskets, _ = load_transactions(data, transaction_column, item_column, basket_column, separator, onehot_columns)
    n = len(baskets)
    freq, _ = mine(baskets, min_support, "fpgrowth", size)
    single = {next(iter(s)): c for s, c in freq.items() if len(s) == 1}
    rows = []
    for s, c in freq.items():
        if len(s) != size:
            continue
        expected = np.prod([single[i] / n for i in s]) * n
        weakest = min((freq[frozenset(p)] / n) / ((single[a] / n) * (single[b] / n)) for p in combinations(sorted(s), 2) for a, b in [p] if frozenset(p) in freq) if size >= 2 and all(frozenset(p) in freq for p in combinations(sorted(s), 2)) else np.nan
        rows.append({"bundle": " + ".join(sorted(s)), "count": c, "support": round(c / n, 5), "lift": round(c / expected, 3), "weakest_pairwise_lift": round(float(weakest), 3) if weakest == weakest else None})
    table = pd.DataFrame(rows)
    table = table[table["lift"] >= min_lift].sort_values(["weakest_pairwise_lift", "lift"], ascending=False) if len(table) else table
    return {"transactions": n, "bundles": int(len(table)), "top": table.head(top), "written_to": save_frame(table, output_path, "bundles.csv") if output_path else None}


@tool
def analyze_item_affinity(data: str, item: str, transaction_column: str | None = None, item_column: str | None = None, basket_column: str | None = None, separator: str = ",", onehot_columns: list[str] | None = None, top: int = 15) -> dict:
    """Everything about one item's company: how often it is bought, and for every other item the number of joint baskets, P(other | item), P(item | other), lift and the Jaccard overlap — a one-item view of the whole rule set, for 'what goes with X' and 'what is X substituted by' (negative association: bought apart far more than chance).

    Args:
        item: The item to analyse.
        top: Items listed in each direction.
    """
    baskets, _ = load_transactions(data, transaction_column, item_column, basket_column, separator, onehot_columns)
    n = len(baskets)
    c = item_counts(baskets)
    if item not in c:
        raise ValueError(f"item {item!r} not found")
    with_item = [b for b in baskets if item in b]
    co: dict[str, int] = {}
    for b in with_item:
        for o in b - {item}:
            co[o] = co.get(o, 0) + 1
    rows = [{"item": o, "baskets_together": k, "p_other_given_item": round(k / len(with_item), 4), "p_item_given_other": round(k / c[o], 4), "lift": round((k / n) / ((len(with_item) / n) * (c[o] / n)), 3), "jaccard": round(k / (len(with_item) + c[o] - k), 4)} for o, k in co.items()]
    t = pd.DataFrame(rows)
    never = [o for o in c if o != item and o not in co]
    return {"item": item, "baskets_with_item": len(with_item), "support": round(len(with_item) / n, 5), "most_associated": t.sort_values("lift", ascending=False).head(top) if len(t) else [], "most_frequent_companions": t.sort_values("baskets_together", ascending=False).head(top) if len(t) else [],
            "possible_substitutes": t[(t["lift"] < 0.6) & (t["baskets_together"] >= 3)].sort_values("lift").head(top) if len(t) else [], "items_never_bought_together": len(never)}


@tool
def cross_sell_matrix(data: str, transaction_column: str | None = None, item_column: str | None = None, basket_column: str | None = None, separator: str = ",", onehot_columns: list[str] | None = None, top_items: int = 15, metric: str = "lift", output_path: str | None = None) -> dict:
    """A matrix of the chosen measure (lift, confidence or joint count) between the most frequent items — the overview picture of what goes with what, ready for a heat map; written as a file whose rows are the antecedent item and columns the consequent.

    Args:
        top_items: How many of the most frequent items to include.
        metric: lift, confidence or count.
        output_path: Where to write the matrix.
    """
    baskets, _ = load_transactions(data, transaction_column, item_column, basket_column, separator, onehot_columns)
    n = len(baskets)
    c = item_counts(baskets)
    items = [i for i, _ in sorted(c.items(), key=lambda kv: -kv[1])[:top_items]]
    M = pd.DataFrame(0.0, index=items, columns=items)
    for a in items:
        for b in items:
            if a == b:
                continue
            both = sum(1 for t in baskets if a in t and b in t)
            M.loc[a, b] = both if metric == "count" else (both / c[a] if metric == "confidence" else (both / n) / ((c[a] / n) * (c[b] / n)))
    flat = M.stack().sort_values(ascending=False)
    return {"items": items, "metric": metric, "strongest_pairs": [{"from": a, "to": b, metric: round(float(v), 4)} for (a, b), v in flat.head(10).items()], "written_to": save_frame(M.reset_index(names="item"), output_path, "cross_sell_matrix.csv")}


@tool
def summarise_rules_by_item(rules_file: str, top: int = 15) -> dict:
    """Which items appear most in the rule set, on which side, and with what average lift — the items that drive the structure (hubs that predict many others, sinks that many items lead to), found without reading thousands of rules.

    Args:
        rules_file: CSV of rules written by `generate_rules`.
        top: Items listed.
    """
    r = load_frame(rules_file)
    require_columns(r, ["antecedent", "consequent", "lift"])
    ante, cons = {}, {}
    for a, c, lift in zip(r["antecedent"], r["consequent"], r["lift"]):
        for i in str(a).split(" + "):
            ante.setdefault(i, []).append(lift)
        for i in str(c).split(" + "):
            cons.setdefault(i, []).append(lift)
    rows = [{"item": i, "as_antecedent": len(ante.get(i, [])), "as_consequent": len(cons.get(i, [])), "mean_lift": round(float(np.mean(ante.get(i, []) + cons.get(i, []))), 3)} for i in set(ante) | set(cons)]
    t = pd.DataFrame(rows)
    return {"rules": int(len(r)), "distinct_items": len(t), "hubs_predict_many": t.sort_values("as_antecedent", ascending=False).head(top), "sinks_predicted_by_many": t.sort_values("as_consequent", ascending=False).head(top)}
