"""Turn frequent itemsets into rules and decide which to believe: all the interestingness measures, filtering, ranking, removing redundant rules, statistical significance with multiple-testing correction, and out-of-sample stability."""

from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd

from ml_core.data import load_frame, require_columns, save_frame
from ml_core.tooling import tool
from unsupervised.association_rule_mining.tools._common import load_transactions, mine, rules_from


def _rules(data, tc, ic, bc, sep, oh, min_support, min_conf, min_lift, max_len, algorithm="fpgrowth", max_antecedent=None, max_consequent=1):
    baskets, ids = load_transactions(data, tc, ic, bc, sep, oh)
    freq, mc = mine(baskets, min_support, algorithm, max_len)
    return rules_from(freq, len(baskets), min_conf, min_lift, max_antecedent, None, max_consequent), baskets, freq


@tool
def generate_rules(data: str, min_support: float = 0.02, min_confidence: float = 0.3, min_lift: float = 1.0, transaction_column: str | None = None, item_column: str | None = None, basket_column: str | None = None, separator: str = ",",
                   onehot_columns: list[str] | None = None, max_len: int = 4, max_consequent: int = 1, sort_by: Literal["lift", "confidence", "support", "leverage", "conviction", "zhang", "kulczynski"] = "lift", top: int = 20, output_path: str | None = None) -> dict:
    """The full pipeline in one call: mine frequent itemsets (FP-growth) and generate every association rule A → B with support, confidence, lift, leverage, conviction, Zhang's metric and the Kulczynski measure. Lift above 1 means A and B occur together more than chance; leverage is the excess in basket share; Kulczynski is robust to very unequal item frequencies.

    Args:
        min_support: Share of transactions the whole rule must appear in.
        min_confidence: Smallest P(B | A).
        min_lift: Smallest lift.
        transaction_column: Column identifying the transaction (long format, with `item_column`).
        item_column: Column holding one item per row (long format).
        basket_column: Alternatively, one column listing each transaction's items joined by `separator`.
        onehot_columns: Alternatively, 0/1 columns, one per item.
        max_len: Longest itemset (antecedent plus consequent).
        max_consequent: Most items on the right side.
        sort_by: Measure to rank by.
        top: Rules listed.
        output_path: Where to write all rules.
    """
    rules, baskets, freq = _rules(data, transaction_column, item_column, basket_column, separator, onehot_columns, min_support, min_confidence, min_lift, max_len, max_consequent=max_consequent)
    rules = rules.sort_values(sort_by, ascending=False) if len(rules) else rules
    return {"transactions": len(baskets), "frequent_itemsets": len(freq), "rules": int(len(rules)), "sorted_by": sort_by, "top": rules.head(top), "written_to": save_frame(rules, output_path, "association_rules.csv") if output_path else None,
            "reading": "support = how often the whole rule occurs; confidence = how often B follows A; lift = confidence relative to B's base rate; rules with high confidence but lift near 1 only restate that B is common"}


@tool
def filter_rules(rules_file: str, min_confidence: float = 0.0, min_lift: float = 1.0, min_support: float = 0.0, min_leverage: float = 0.0, min_conviction: float = 0.0, max_antecedent_size: int | None = None,
                 must_contain: str | None = None, exclude_item: str | None = None, output_path: str | None = None, top: int = 20) -> dict:
    """Filter a rules file (written by `generate_rules`) by any measure, by antecedent size, and by items that must or must not appear on either side — the interactive narrowing that turns thousands of rules into the dozens relevant to a question.

    Args:
        rules_file: CSV of rules with the columns written by `generate_rules`.
        min_leverage: Smallest leverage.
        min_conviction: Smallest conviction.
        max_antecedent_size: Largest number of items on the left.
        must_contain: Keep only rules mentioning this item.
        exclude_item: Drop rules mentioning this item.
        top: Rules listed.
        output_path: Where to write the filtered rules.
    """
    r = load_frame(rules_file)
    require_columns(r, ["antecedent", "consequent", "confidence", "lift", "support"])
    keep = (r["confidence"] >= min_confidence) & (r["lift"] >= min_lift) & (r["support"] >= min_support) & (r.get("leverage", 0) >= min_leverage) & (r.get("conviction", np.inf) >= min_conviction)
    if max_antecedent_size:
        keep &= r["antecedent_size"] <= max_antecedent_size
    text = r["antecedent"].astype(str) + " + " + r["consequent"].astype(str)
    if must_contain:
        keep &= text.str.contains(must_contain, regex=False)
    if exclude_item:
        keep &= ~text.str.contains(exclude_item, regex=False)
    out = r[keep].sort_values("lift", ascending=False)
    return {"rules_in": int(len(r)), "rules_kept": int(len(out)), "top": out.head(top), "written_to": save_frame(out, output_path, "filtered_rules.csv") if output_path else None}


@tool
def rank_rules(rules_file: str, weights: dict | None = None, top: int = 20, output_path: str | None = None) -> dict:
    """Rank rules by a weighted blend of measures (each converted to a percentile so scales are comparable) — defaults weigh lift and Kulczynski most, then confidence, leverage and support. A single sort key can mislead (confidence rewards popular consequents, lift rewards rare ones); a blend is steadier.

    Args:
        rules_file: CSV of rules written by `generate_rules`.
        weights: Measure name → weight, e.g. {"lift": 2, "confidence": 1, "support": 0.5}.
        top: Rules listed.
        output_path: Where to write the ranked rules.
    """
    r = load_frame(rules_file)
    w = weights or {"lift": 3, "kulczynski": 3, "confidence": 2, "leverage": 2, "support": 1}
    missing = [k for k in w if k not in r.columns]
    if missing:
        raise KeyError(f"measures {missing} are not columns of the rules file; columns are {list(r.columns)}")
    score = sum(float(v) * r[k].replace([np.inf], r[k][np.isfinite(r[k])].max()).rank(pct=True) for k, v in w.items()) / sum(w.values())
    out = r.assign(rank_score=score).sort_values("rank_score", ascending=False)
    return {"weights": w, "rules": int(len(out)), "top": out.head(top), "written_to": save_frame(out, output_path, "ranked_rules.csv") if output_path else None}


@tool
def prune_redundant_rules(rules_file: str, tolerance: float = 0.0, output_path: str | None = None, top: int = 20) -> dict:
    """Remove redundant rules: A,B → C is redundant when a simpler rule A → C (a sub-antecedent, same consequent) is at least as confident — the extra item adds nothing. Typically removes the majority of a rules file and leaves the rules that each say something new.

    Args:
        rules_file: CSV of rules written by `generate_rules`.
        tolerance: Confidence a longer rule may lose against its simpler version and still be called redundant.
        top: Rules listed.
        output_path: Where to write the pruned rules.
    """
    r = load_frame(rules_file)
    require_columns(r, ["antecedent", "consequent", "confidence"])
    best = {(frozenset(a.split(" + ")), c): conf for a, c, conf in zip(r["antecedent"], r["consequent"], r["confidence"])}
    drop = []
    for i, (a, c, conf) in enumerate(zip(r["antecedent"], r["consequent"], r["confidence"])):
        items = frozenset(a.split(" + "))
        from itertools import combinations

        simpler = [best.get((frozenset(sub), c)) for k in range(1, len(items)) for sub in combinations(sorted(items), k)]
        if any(v is not None and v >= conf - tolerance for v in simpler):
            drop.append(i)
    out = r.drop(index=r.index[drop]).sort_values("lift", ascending=False)
    return {"rules_in": int(len(r)), "redundant_removed": len(drop), "rules_kept": int(len(out)), "top": out.head(top), "written_to": save_frame(out, output_path, "pruned_rules.csv") if output_path else None}


@tool
def test_rule_significance(data: str, rules_file: str, transaction_column: str | None = None, item_column: str | None = None, basket_column: str | None = None, separator: str = ",", onehot_columns: list[str] | None = None,
                           alpha: float = 0.05, top: int = 20, output_path: str | None = None) -> dict:
    """Test every rule against chance: Fisher's exact test on the 2×2 table of (antecedent present, consequent present), then a Benjamini–Hochberg correction for the thousands of rules tested. A rule that survives is unlikely to be a coincidence of counting; with many rules and few transactions, many high-lift rules do not.

    Args:
        rules_file: CSV of rules written by `generate_rules` (the same transactions must be given).
        alpha: False-discovery rate to control.
        top: Rules listed.
        output_path: Where to write the rules with p-values.
    """
    from scipy.stats import fisher_exact

    baskets, _ = load_transactions(data, transaction_column, item_column, basket_column, separator, onehot_columns)
    n = len(baskets)
    r = load_frame(rules_file)
    rows = []
    for a, c in zip(r["antecedent"], r["consequent"]):
        A, C = frozenset(a.split(" + ")), frozenset(c.split(" + "))
        both = sum(1 for b in baskets if A <= b and C <= b)
        ante = sum(1 for b in baskets if A <= b)
        cons = sum(1 for b in baskets if C <= b)
        table = [[both, ante - both], [cons - both, n - ante - cons + both]]
        rows.append(fisher_exact(table, alternative="greater")[1])
    p = np.asarray(rows)
    order = np.argsort(p)
    adj = np.empty_like(p)
    adj[order] = np.minimum.accumulate((p[order] * len(p) / (np.arange(len(p)) + 1))[::-1])[::-1]
    out = r.assign(p_value=p, p_adjusted_bh=np.clip(adj, 0, 1))
    out["significant"] = out["p_adjusted_bh"] <= alpha
    kept = out[out["significant"]].sort_values("lift", ascending=False)
    return {"rules_tested": int(len(r)), "significant_after_correction": int(len(kept)), "false_discovery_rate": alpha, "share_surviving": round(float(len(kept) / max(len(r), 1)), 4), "top": kept.head(top),
            "written_to": save_frame(out, output_path, "rules_with_significance.csv") if output_path else None}


@tool
def validate_rules_out_of_sample(data: str, min_support: float = 0.02, min_confidence: float = 0.3, min_lift: float = 1.2, transaction_column: str | None = None, item_column: str | None = None, basket_column: str | None = None,
                                 separator: str = ",", onehot_columns: list[str] | None = None, test_fraction: float = 0.3, seed: int = 42, top: int = 15) -> dict:
    """Mine the rules on a random part of the transactions and re-measure each on the rest: how many survive the support / confidence / lift thresholds, how much the lift shrinks (it shrinks for chance rules — the 'winner's curse'), and the rank correlation between train and test lift. The honest answer to 'will these rules still hold next month?'.

    Args:
        min_support: Share of transactions the whole rule must appear in.
        min_confidence: Smallest P(B | A).
        min_lift: Smallest lift.
        transaction_column: Column identifying the transaction (long format, with `item_column`).
        item_column: Column holding one item per row (long format).
        basket_column: Alternatively, one column listing each transaction's items joined by `separator`.
        onehot_columns: Alternatively, 0/1 columns, one per item.
        test_fraction: Share of transactions held out.
        top: Rules listed.
    """
    from scipy.stats import spearmanr

    baskets, _ = load_transactions(data, transaction_column, item_column, basket_column, separator, onehot_columns)
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(baskets))
    cut = int(len(idx) * (1 - test_fraction))
    train, test = [baskets[i] for i in idx[:cut]], [baskets[i] for i in idx[cut:]]
    freq, _ = mine(train, min_support, "fpgrowth", 4)
    rules = rules_from(freq, len(train), min_confidence, min_lift)
    if rules.empty:
        raise ValueError("no rules at these thresholds on the training part; lower them")
    nt = len(test)
    rows = []
    for _, r in rules.iterrows():
        A, C = frozenset(r["antecedent"].split(" + ")), frozenset(r["consequent"].split(" + "))
        both = sum(1 for b in test if A <= b and C <= b)
        a = sum(1 for b in test if A <= b)
        c = sum(1 for b in test if C <= b)
        conf = both / a if a else np.nan
        rows.append({"test_support": both / nt, "test_confidence": conf, "test_lift": conf / (c / nt) if a and c else np.nan})
    res = pd.concat([rules.reset_index(drop=True), pd.DataFrame(rows)], axis=1)
    held = res[(res["test_support"] >= min_support / 2) & (res["test_confidence"] >= min_confidence) & (res["test_lift"] >= min_lift)]
    ok = res.dropna(subset=["test_lift"])
    return {"train_transactions": len(train), "test_transactions": nt, "rules_mined": int(len(res)), "rules_holding_up": int(len(held)), "share_holding_up": round(float(len(held) / len(res)), 4),
            "mean_lift_train": round(float(ok["lift"].mean()), 3), "mean_lift_test": round(float(ok["test_lift"].mean()), 3), "lift_shrinkage": round(float(1 - ok["test_lift"].mean() / ok["lift"].mean()), 4),
            "rank_correlation_train_vs_test_lift": round(float(spearmanr(ok["lift"], ok["test_lift"])[0]), 4), "most_stable": held.sort_values("test_lift", ascending=False).head(top)[["antecedent", "consequent", "lift", "test_lift", "confidence", "test_confidence"]],
            "reading": "a share holding up below ~50%, or strong shrinkage, means many of the rules are noise: raise the support, or keep only rules that pass test_rule_significance"}
