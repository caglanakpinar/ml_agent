"""Frequent patterns: itemsets by Apriori, Eclat and FP-growth (identical results, different cost profiles), condensed closed/maximal sets, rare associations, high-value itemsets, sequential patterns, and rules for a chosen target or at category level."""

from __future__ import annotations


import numpy as np
import pandas as pd

from ml_core.data import load_frame, require_columns, save_frame
from ml_core.tooling import tool
from unsupervised.association_rule_mining.tools._common import (
    closed_itemsets,
    eclat,
    freq_table,
    label,
    load_transactions,
    maximal_itemsets,
    mine,
    prefixspan,
    rules_from,
)


def _itemsets(name, data, tc, ic, bc, sep, oh, min_support, max_len, output_path, top, algorithm):
    baskets, _ = load_transactions(data, tc, ic, bc, sep, oh)
    freq, mc = mine(baskets, min_support, algorithm, max_len)
    table = freq_table(freq, len(baskets))
    sizes = table["size"].value_counts().sort_index() if len(table) else pd.Series(dtype=int)
    return {"algorithm": name, "transactions": len(baskets), "min_support": min_support, "min_count": mc, "frequent_itemsets": int(len(table)), "by_size": {int(k): int(v) for k, v in sizes.items()},
            "top": table.head(top), "written_to": save_frame(table, output_path, f"itemsets_{name}.csv") if output_path else None}, freq, baskets


@tool
def mine_apriori(data: str, min_support: float = 0.02, transaction_column: str | None = None, item_column: str | None = None, basket_column: str | None = None, separator: str = ",", onehot_columns: list[str] | None = None,
                 max_len: int | None = 4, top: int = 15, output_path: str | None = None) -> dict:
    """Apriori: finds frequent itemsets level by level, pruning every candidate that has an infrequent subset. The classic, easiest to explain; slow when there are many frequent items, because it generates and tests many candidates.

    Args:
        min_support: Share of transactions an itemset must appear in.
        transaction_column: Column identifying the transaction (long format, with `item_column`).
        item_column: Column holding one item per row (long format).
        basket_column: Alternatively, one column listing each transaction's items joined by `separator`.
        onehot_columns: Alternatively, 0/1 columns, one per item.
        max_len: Longest itemset.
        top: Itemsets listed in the reply.
        output_path: Where to write all frequent itemsets.
    """
    return _itemsets("apriori", data, transaction_column, item_column, basket_column, separator, onehot_columns, min_support, max_len, output_path, top, "apriori")[0]


@tool
def mine_eclat(data: str, min_support: float = 0.02, transaction_column: str | None = None, item_column: str | None = None, basket_column: str | None = None, separator: str = ",", onehot_columns: list[str] | None = None,
               max_len: int | None = 4, top: int = 15, output_path: str | None = None) -> dict:
    """Eclat: depth-first mining over vertical transaction-id sets — an itemset's support is the size of the intersection of its items' sets. Fast on dense data with few items; memory grows with the number of transactions.

    Args:
        min_support: Share of transactions an itemset must appear in.
        transaction_column: Column identifying the transaction (long format, with `item_column`).
        item_column: Column holding one item per row (long format).
        basket_column: Alternatively, one column listing each transaction's items joined by `separator`.
        onehot_columns: Alternatively, 0/1 columns, one per item.
        max_len: Longest itemset.
        top: Itemsets listed in the reply.
        output_path: Where to write all frequent itemsets.
    """
    return _itemsets("eclat", data, transaction_column, item_column, basket_column, separator, onehot_columns, min_support, max_len, output_path, top, "eclat")[0]


@tool
def mine_fpgrowth(data: str, min_support: float = 0.02, transaction_column: str | None = None, item_column: str | None = None, basket_column: str | None = None, separator: str = ",", onehot_columns: list[str] | None = None,
                  max_len: int | None = 4, top: int = 15, output_path: str | None = None) -> dict:
    """FP-growth: compresses the transactions into a prefix tree and mines it recursively without generating candidates — the fastest of the three on large or dense data, and the default choice. Returns exactly the itemsets Apriori and Eclat would.

    Args:
        min_support: Share of transactions an itemset must appear in.
        transaction_column: Column identifying the transaction (long format, with `item_column`).
        item_column: Column holding one item per row (long format).
        basket_column: Alternatively, one column listing each transaction's items joined by `separator`.
        onehot_columns: Alternatively, 0/1 columns, one per item.
        max_len: Longest itemset.
        top: Itemsets listed in the reply.
        output_path: Where to write all frequent itemsets.
    """
    return _itemsets("fpgrowth", data, transaction_column, item_column, basket_column, separator, onehot_columns, min_support, max_len, output_path, top, "fpgrowth")[0]


@tool
def mine_closed_itemsets(data: str, min_support: float = 0.02, transaction_column: str | None = None, item_column: str | None = None, basket_column: str | None = None, separator: str = ",", onehot_columns: list[str] | None = None,
                         max_len: int | None = 5, top: int = 15, output_path: str | None = None) -> dict:
    """Closed frequent itemsets: those with no larger itemset of the *same* support. They are a lossless summary — every frequent itemset and its support can be recovered — typically many times smaller than the full list, and free of redundant sub-patterns.

    Args:
        min_support: Share of transactions an itemset must appear in.
        transaction_column: Column identifying the transaction (long format, with `item_column`).
        item_column: Column holding one item per row (long format).
        basket_column: Alternatively, one column listing each transaction's items joined by `separator`.
        onehot_columns: Alternatively, 0/1 columns, one per item.
        max_len: Longest itemset.
        top: Itemsets listed in the reply.
        output_path: Where to write the closed itemsets.
    """
    summary, freq, baskets = _itemsets("closed", data, transaction_column, item_column, basket_column, separator, onehot_columns, min_support, max_len, None, top, "fpgrowth")
    closed = closed_itemsets(freq)
    table = freq_table(closed, len(baskets))
    summary.update({"all_frequent_itemsets": len(freq), "closed_itemsets": len(closed), "compression": round(1 - len(closed) / max(len(freq), 1), 4), "top": table.head(top),
                    "written_to": save_frame(table, output_path, "itemsets_closed.csv") if output_path else None})
    return summary


@tool
def mine_maximal_itemsets(data: str, min_support: float = 0.02, transaction_column: str | None = None, item_column: str | None = None, basket_column: str | None = None, separator: str = ",", onehot_columns: list[str] | None = None,
                          max_len: int | None = 6, top: int = 15, output_path: str | None = None) -> dict:
    """Maximal frequent itemsets: the largest frequent combinations, with no frequent superset — the boundary of what customers buy together. The most compact summary (supports of subsets are lost), good for spotting product bundles and the longest recurring baskets.

    Args:
        min_support: Share of transactions an itemset must appear in.
        transaction_column: Column identifying the transaction (long format, with `item_column`).
        item_column: Column holding one item per row (long format).
        basket_column: Alternatively, one column listing each transaction's items joined by `separator`.
        onehot_columns: Alternatively, 0/1 columns, one per item.
        max_len: Longest itemset.
        top: Itemsets listed in the reply.
        output_path: Where to write the maximal itemsets.
    """
    summary, freq, baskets = _itemsets("maximal", data, transaction_column, item_column, basket_column, separator, onehot_columns, min_support, max_len, None, top, "fpgrowth")
    mx = maximal_itemsets(freq)
    table = freq_table(mx, len(baskets)).sort_values(["size", "count"], ascending=False)
    summary.update({"all_frequent_itemsets": len(freq), "maximal_itemsets": len(mx), "largest": int(table["size"].max()) if len(table) else 0, "top": table.head(top),
                    "written_to": save_frame(table, output_path, "itemsets_maximal.csv") if output_path else None})
    return summary


@tool
def mine_rare_associations(data: str, max_support: float = 0.02, min_count: int = 3, min_lift: float = 3.0, min_confidence: float = 0.5, transaction_column: str | None = None, item_column: str | None = None, basket_column: str | None = None,
                           separator: str = ",", onehot_columns: list[str] | None = None, top: int = 20, output_path: str | None = None) -> dict:
    """Rare but strong associations: pairs and triples that occur only a few times yet far more often together than chance predicts (high lift) and reliably (high confidence) — the niche affinities a support threshold throws away, e.g. specialist products, fraud rings, symptom pairs. Counts are small, so significance matters (see `test_rule_significance`).

    Args:
        max_support: Itemsets more frequent than this are left to the ordinary miners.
        min_count: Smallest number of occurrences to consider at all.
        min_lift: Smallest lift.
        min_confidence: Smallest confidence.
        transaction_column: Column identifying the transaction (long format, with `item_column`).
        item_column: Column holding one item per row (long format).
        basket_column: Alternatively, one column listing each transaction's items joined by `separator`.
        onehot_columns: Alternatively, 0/1 columns, one per item.
        top: Rules listed.
        output_path: Where to write all rare rules.
    """
    baskets, _ = load_transactions(data, transaction_column, item_column, basket_column, separator, onehot_columns)
    n = len(baskets)
    freq = eclat(baskets, max(min_count, 2), 3)
    rules = rules_from(freq, n, min_confidence, min_lift)
    rules = rules[(rules["support"] <= max_support) & (rules["support_count"] >= min_count)].sort_values(["lift", "support_count"], ascending=False)
    return {"transactions": n, "rare_rules": int(len(rules)), "top": rules.head(top), "written_to": save_frame(rules, output_path, "rare_rules.csv") if output_path else None,
            "caution": "with counts this small many strong-looking rules are chance: run test_rule_significance (with FDR correction) on them before acting"}


@tool
def mine_high_utility_itemsets(data: str, transaction_column: str, item_column: str, value_column: str, min_support: float = 0.005, min_total_value: float | None = None, max_len: int = 3, top: int = 20, output_path: str | None = None) -> dict:
    """Rank frequent itemsets by the *value* they carry, not their frequency: for every itemset the total of the value column (revenue, margin, quantity) over the transactions that contain it — a combination bought rarely but at high value outranks a common cheap one. Plain support mining would miss it.

    Args:
        transaction_column: Column identifying the transaction.
        item_column: Column holding one item per row.
        value_column: Numeric column holding each line's value.
        min_support: Share of transactions an itemset must appear in.
        min_total_value: Smallest total value to report; none when omitted.
        max_len: Longest itemset.
        top: Itemsets listed.
        output_path: Where to write the ranked itemsets.
    """
    frame = load_frame(data)
    require_columns(frame, [transaction_column, item_column, value_column])
    frame = frame.assign(**{item_column: frame[item_column].astype(str), value_column: pd.to_numeric(frame[value_column], errors="coerce").fillna(0)})
    baskets, ids = load_transactions(frame, transaction_column, item_column)
    freq, mc = mine(baskets, min_support, "fpgrowth", max_len)
    line = frame.groupby([transaction_column, item_column])[value_column].sum()
    by_tx = {t: line.xs(t, level=0).to_dict() for t in ids}
    rows = []
    for s, c in freq.items():
        if len(s) < 2:
            continue
        total = sum(sum(by_tx[t].get(i, 0.0) for i in s) for t, b in zip(ids, baskets) if s <= b)
        rows.append({"itemset": label(s), "size": len(s), "count": c, "support": round(c / len(baskets), 5), "total_value": round(float(total), 2), "value_per_transaction": round(float(total / c), 2)})
    table = pd.DataFrame(rows).sort_values("total_value", ascending=False) if rows else pd.DataFrame(columns=["itemset"])
    if min_total_value is not None and len(table):
        table = table[table["total_value"] >= min_total_value]
    return {"transactions": len(baskets), "itemsets_ranked": int(len(table)), "top": table.head(top), "written_to": save_frame(table, output_path, "high_utility_itemsets.csv") if output_path else None}


@tool
def mine_sequential_patterns(data: str, sequence_column: str, item_column: str, order_column: str, min_support: float = 0.05, max_len: int = 4, top: int = 20, output_path: str | None = None) -> dict:
    """Sequential patterns (PrefixSpan): ordered sequences of items that many customers or sessions follow — 'A, then later B, then later C' — not just what is bought together. For page-view paths, repeat-purchase journeys and event sequences; each sequence is the events of one customer, ordered by time.

    Args:
        sequence_column: Column identifying the sequence (customer, session).
        item_column: Column holding the event or item.
        order_column: Column that orders the events (timestamp or step number).
        min_support: Share of sequences a pattern must occur in.
        max_len: Longest pattern.
        top: Patterns listed.
        output_path: Where to write all patterns.
    """
    frame = load_frame(data)
    require_columns(frame, [sequence_column, item_column, order_column])
    seqs = [list(g.sort_values(order_column)[item_column].astype(str)) for _, g in frame.groupby(sequence_column)]
    mc = max(2, int(np.ceil(min_support * len(seqs))))
    found = prefixspan(seqs, mc, max_len)
    table = pd.DataFrame([{"pattern": " → ".join(p), "length": len(p), "sequences": c, "support": round(c / len(seqs), 5)} for p, c in found]).sort_values(["sequences", "length"], ascending=False)
    return {"sequences": len(seqs), "mean_length": round(float(np.mean([len(s) for s in seqs])), 2), "min_count": mc, "patterns": int(len(table)), "longest": int(table["length"].max()) if len(table) else 0, "top": table[table["length"] >= 2].head(top),
            "written_to": save_frame(table, output_path, "sequential_patterns.csv") if output_path else None}


@tool
def mine_rules_for_target(data: str, target_item: str, min_support: float = 0.01, min_confidence: float = 0.2, min_lift: float = 1.2, transaction_column: str | None = None, item_column: str | None = None, basket_column: str | None = None,
                          separator: str = ",", onehot_columns: list[str] | None = None, max_antecedent: int = 3, top: int = 20, output_path: str | None = None) -> dict:
    """Rules whose consequent is one chosen item — what predicts that someone buys / does / has X. The support threshold is applied to the *rule* (antecedent plus target), so a rare target can still be explained; the output is ranked by lift, the usual way to say 'which items most raise the chance of X'.

    Args:
        target_item: The item to explain.
        min_support: Share of transactions the antecedent together with the target must appear in.
        min_confidence: Smallest P(target | antecedent).
        min_lift: Smallest lift.
        max_antecedent: Most items on the left side.
        transaction_column: Column identifying the transaction (long format, with `item_column`).
        item_column: Column holding one item per row (long format).
        basket_column: Alternatively, one column listing each transaction's items joined by `separator`.
        onehot_columns: Alternatively, 0/1 columns, one per item.
        top: Rules listed.
        output_path: Where to write the rules.
    """
    baskets, _ = load_transactions(data, transaction_column, item_column, basket_column, separator, onehot_columns)
    if not any(target_item in b for b in baskets):
        raise ValueError(f"the item {target_item!r} never occurs in the transactions")
    freq, mc = mine(baskets, min_support, "fpgrowth", max_antecedent + 1)
    rules = rules_from(freq, len(baskets), min_confidence, min_lift, max_antecedent, frozenset([target_item]))
    rules = rules.sort_values(["lift", "confidence"], ascending=False) if len(rules) else rules
    base = sum(1 for b in baskets if target_item in b) / len(baskets)
    return {"target": target_item, "target_base_rate": round(base, 5), "rules_found": int(len(rules)), "top": rules.head(top), "written_to": save_frame(rules, output_path, f"rules_for_{target_item}.csv") if output_path else None,
            "reading": "lift is how many times more likely the target is, given the antecedent, than in a random basket"}


@tool
def mine_rules_by_category(data: str, transaction_column: str, item_column: str, category_column: str, min_support: float = 0.02, min_confidence: float = 0.3, min_lift: float = 1.2, max_len: int = 3, top: int = 20, output_path: str | None = None) -> dict:
    """Mine rules between *categories* instead of individual items: each item is replaced by its category (a department, brand, product family) before mining. With a long-tailed catalogue the individual items are too rare for any rule to clear the support threshold, while categories are not — the standard fix, and the right level for merchandising decisions.

    Args:
        transaction_column: Column identifying the transaction.
        item_column: Column holding one item per row (kept for the report).
        category_column: Column holding the item's category.
        min_support: Share of transactions an itemset must appear in.
        min_confidence: Smallest confidence.
        min_lift: Smallest lift.
        max_len: Longest itemset.
        top: Rules listed.
        output_path: Where to write the rules.
    """
    frame = load_frame(data)
    require_columns(frame, [transaction_column, item_column, category_column])
    items = frame[item_column].nunique()
    baskets, _ = load_transactions(frame, transaction_column, category_column)
    freq, mc = mine(baskets, min_support, "fpgrowth", max_len)
    rules = rules_from(freq, len(baskets), min_confidence, min_lift)
    rules = rules.sort_values("lift", ascending=False) if len(rules) else rules
    return {"items": int(items), "categories": int(frame[category_column].nunique()), "transactions": len(baskets), "frequent_itemsets": len(freq), "rules": int(len(rules)), "top": rules.head(top), "written_to": save_frame(rules, output_path, "category_rules.csv") if output_path else None}
