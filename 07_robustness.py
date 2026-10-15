"""Robustness analyses from the appendix: bootstrap CIs, MDS sensitivity and label stability.

Inputs:  results/final_scores/<model>_eval_ood_subgroup_canonical_scores.csv (from 04_score.py),
         data/singapore_wvs_analysis_rich.json, data/codebook.json
Outputs: results/robustness/bootstrap_ci_main_results.csv        paired bootstrap CIs, base -> SFT
         results/robustness/mds_sensitivity_by_category.csv      MDS (two normalisers), ordinal Wasserstein
         results/robustness/mds_vs_ordinal_metric_rankcorr.json  rank correlations and label stability
         results/robustness/label_stability_summary.csv          mode margin per (question, subgroup)
         outputs/tables/{bootstrap_ci,bootstrap_ci_oe,mds_ordinal_comparison}.{csv,tex}
Paper:   tab:bootstrap_ci, tab:bootstrap_ci_oe, tab:mds_ordinal_comparison and the label-stability
         result (rho = -0.007, p = 0.32 over 22,837 pairs).
Runtime: about 15 seconds.

As published, the MDS-sensitivity and label-stability analyses use every question in
singapore_wvs_analysis_rich.json: 234 questions, including Q7-Q26, which are excluded from the
dataset and from the main-text MDS table (214 questions). Their MDS values therefore differ
slightly from the main text, and the 22,837 pairs span all ten strata.
"""

import json
import math
from collections import Counter

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, wasserstein_distance

from svalues import config, metrics
from svalues.tables import fmt, save_table

SPLIT = "eval_ood_subgroup"
N_BOOT = 2000
ALPHA = 0.05
SEED = 42

# Short category names used in the appendix table.
SHORT_CATEGORY = {
    "Religious Values": "Religious Values",
    "Perceptions about Science and Technology": r"Percep.\ Sci/Tech",
    "Political Culture and Political Regimes": "Political Culture",
    "Economic Values": "Economic Values",
    "Perceptions of Migration": r"Percep.\ Migration",
    "Perceptions of Corruption": r"Percep.\ Corruption",
    "Perceptions of Security": r"Percep.\ Security",
    "Social Values, Norms, Stereotypes": "Social Values/Norms",
    "Political Interest and Political Participation": "Political Interest",
    "Happiness and Wellbeing": "Happiness/Wellbeing",
    "Ethical Values": "Ethical Values",
    "Social Capital, Trust and Organizational Membership": "Social Capital/Trust",
}


# --- Bootstrap CIs -------------------------------------------------------------


def item_scores(df: pd.DataFrame) -> dict:
    """Per-item arrays needed by the metrics: correctness, refusal flag, normalised error, WRs."""
    y_true = df["preferred_answer"].astype(int).to_numpy()
    lo = df["choice_min"].astype(int).to_numpy()
    hi = df["choice_max"].astype(int).to_numpy()
    y_pred = df["numerical_response"].apply(metrics.parse_numerical_response).to_numpy()
    refused = (y_pred == -1) | (y_pred < lo) | (y_pred > hi)  # out-of-range answers count as refusals
    y_pred = np.where(refused, -1, y_pred)
    scale = (hi - lo).astype(float)
    scale[scale <= 0] = 1.0
    scores = {"correct": y_true == y_pred, "refused": refused, "error": np.abs(y_true - y_pred) / scale}
    for key in ("persona", "value", "overall"):
        scores[f"{key}_wr"] = df[f"canonical_{key}_score"].to_numpy()
    return scores


def numerical_metrics(s: dict, idx: np.ndarray) -> dict:
    """Accuracy, NMAE and refusal rate; refusals are wrong and get the maximum error of 1."""
    correct, refused, error = s["correct"][idx], s["refused"][idx], s["error"][idx]
    n_refused = refused.sum()
    # Valid errors first, then the refusal penalties: this summation order reproduces the
    # published CSV bit for bit.
    errors = np.concatenate([error[~refused], np.ones(n_refused)])
    return {"accuracy": float(correct.sum() / len(idx)), "nmae": float(np.mean(errors)),
            "refusal_rate": float(n_refused / len(idx))}


def open_ended_metrics(s: dict, idx: np.ndarray) -> dict:
    """Open-ended win rates vs GPT-4.1 (mean canonical judge score)."""
    return {key: float(np.mean(s[key][idx])) for key in ("overall_wr", "persona_wr", "value_wr")}


def bootstrap_pair(base_df: pd.DataFrame, sft_df: pd.DataFrame) -> list[dict]:
    """Paired bootstrap of SFT - base deltas, resampling (question, subgroup) items."""
    base, sft = base_df.set_index("sample_id"), sft_df.set_index("sample_id")
    common = base.index.intersection(sft.index)
    b, t = item_scores(base.loc[common]), item_scores(sft.loc[common])
    n = len(common)

    tasks = {"numerical": numerical_metrics, "open-ended": open_ended_metrics}
    full = np.arange(n)
    points = {task: (fn(b, full), fn(t, full)) for task, fn in tasks.items()}
    deltas = {task: {k: [] for k in points[task][0]} for task in tasks}

    # A fresh generator with the same seed for every model pair (and both tasks), so all
    # pairs are evaluated on the same 2,000 resamples, exactly as in the published run.
    rng = np.random.default_rng(SEED)
    for _ in range(N_BOOT):
        idx = rng.choice(full, size=n, replace=True)
        for task, fn in tasks.items():
            mb, mt = fn(b, idx), fn(t, idx)
            for k in mb:
                deltas[task][k].append(mt[k] - mb[k])

    rows = []
    for task in tasks:
        base_point, sft_point = points[task]
        for metric, values in deltas[task].items():
            arr = np.array(values, dtype=float)
            arr = arr[np.isfinite(arr)]
            rows.append({"task": task, "metric": metric, "base": base_point[metric],
                         "tuned": sft_point[metric], "delta": sft_point[metric] - base_point[metric],
                         "ci_lo": float(np.quantile(arr, ALPHA / 2)),
                         "ci_hi": float(np.quantile(arr, 1 - ALPHA / 2)),
                         "n_samples": n, "n_boot": N_BOOT})
    return rows


def run_bootstrap() -> pd.DataFrame:
    """Bootstrap CIs for all seven base/SFT pairs on the OOD split."""
    rows = []
    # Rows are ordered by results file name, as in the original output.
    for base_model, sft_model in sorted(config.BASE_SFT_PAIRS, key=lambda pair: pair[0].safe_name):
        print(f"  bootstrapping {base_model.label}")
        base_df = pd.read_csv(config.scores_file(base_model.safe_name, SPLIT))
        sft_df = pd.read_csv(config.scores_file(sft_model.safe_name, SPLIT))
        for row in bootstrap_pair(base_df, sft_df):
            rows.append({"model": base_model.safe_name, "split": SPLIT, **row})
    return pd.DataFrame(rows)


# --- MDS sensitivity, ordinal divergence and label stability -------------------


def modal_diversity(modal_answers: list[str], num_choices: int, denominator: str) -> float:
    """Normalised entropy of subgroup modal answers ("min_sc": paper, "observed_m": alternative)."""
    s = len(modal_answers)
    p = np.array([c / s for c in Counter(modal_answers).values()], dtype=float)
    entropy = float(-(p * np.log2(p)).sum())
    if denominator == "min_sc":
        denom = math.log2(max(1, min(s, max(1, num_choices))))
    else:
        denom = math.log2(max(1, len(p)))
    return entropy / denom if denom > 0 else 0.0


def label_to_number(codebook: dict) -> dict:
    """q_id -> {answer label: numeric scale point}."""
    out = {}
    for q_id, info in codebook.items():
        out[q_id] = {}
        for key, label in info.get("choices", {}).items():
            try:
                out[q_id][str(label)] = int(key)
            except ValueError:
                continue
    return out


def to_numeric(dist: dict, label_map: dict) -> tuple[np.ndarray, np.ndarray]:
    """Answer distribution -> (scale points, normalised weights)."""
    xs, ws = [], []
    for label, prob in dist.items():
        if label in label_map:
            xs.append(label_map[label])
            ws.append(float(prob))
        elif isinstance(label, str) and label.isdigit():
            xs.append(int(label))
            ws.append(float(prob))
    w = np.array(ws, dtype=float)
    if w.sum() > 0:
        w = w / w.sum()
    return np.array(xs, dtype=float), w


def mean_pairwise_wasserstein(dists: dict, label_map: dict, scale_min: int, scale_max: int) -> float:
    """Mean Wasserstein distance between all pairs of subgroup distributions, / scale range."""
    if len(dists) < 2:
        return 0.0
    scale = max(1, scale_max - scale_min)
    numeric = [to_numeric(d, label_map) for d in dists.values()]
    numeric = [(x, w) for x, w in numeric if len(x) > 0]
    if len(numeric) < 2:
        return float("nan")
    distances = [wasserstein_distance(xa, xb, u_weights=wa, v_weights=wb) / scale
                 for i, (xa, wa) in enumerate(numeric) for xb, wb in numeric[i + 1:]]
    return float(np.mean(distances))


def analyse_question(q_id: str, info: dict, label_map: dict) -> tuple[dict | None, list[dict]]:
    """Per-question conflict scores (averaged over strata) and per-subgroup stability rows."""
    stability, mds_min_sc, mds_observed_m, ordinal = [], [], [], []
    for stratum, stratum_info in info.get("strata_analysis", {}).items():
        subgroups = stratum_info.get("subgroups", {})
        if not subgroups:
            continue

        for name, sg in subgroups.items():
            dist, n = sg.get("answer_distribution"), sg.get("sample_count")
            if isinstance(dist, dict) and dist and isinstance(n, (int, float)):
                probs = sorted((float(v) for v in dist.values()), reverse=True)
                top2 = probs[1] if len(probs) > 1 else 0.0
                stability.append({"q_id": q_id, "category": info.get("category"), "stratum": stratum,
                                  "subgroup": name, "n": int(n), "top1_prob": probs[0],
                                  "top2_prob": top2, "mode_margin": probs[0] - top2})

        modes = [sg.get("most_frequent_answer") for sg in subgroups.values()]
        modes = [m for m in modes if isinstance(m, str) and m]
        if modes:
            # As published, |C| is the number of answers observed in the first subgroup's
            # distribution rather than the codebook's number of options.
            first = next((sg["answer_distribution"] for sg in subgroups.values()
                          if isinstance(sg.get("answer_distribution"), dict) and sg["answer_distribution"]), None)
            num_choices = len(first) if first else 1
            mds_min_sc.append(modal_diversity(modes, num_choices, "min_sc"))
            mds_observed_m.append(modal_diversity(modes, num_choices, "observed_m"))

        dists = {name: sg["answer_distribution"] for name, sg in subgroups.items()
                 if isinstance(sg.get("answer_distribution"), dict) and sg["answer_distribution"]}
        lo, hi = info.get("answer_scale_min"), info.get("answer_scale_max")
        if dists and isinstance(lo, (int, float)) and isinstance(hi, (int, float)):
            score = mean_pairwise_wasserstein(dists, label_map, int(lo), int(hi))
            if np.isfinite(score):
                ordinal.append(score)

    if not mds_min_sc:
        return None, stability
    scores = {"q_id": q_id, "category": info.get("category"), "mds_min_sc": float(np.mean(mds_min_sc)),
              "mds_observed_m": float(np.mean(mds_observed_m)),
              "ordinal_pairwise_wd": float(np.mean(ordinal)) if ordinal else float("nan")}
    return scores, stability


def spearman(x: pd.Series, y: pd.Series) -> tuple[float, float]:
    """Spearman rank correlation as plain floats."""
    rho, p = spearmanr(x, y)
    return float(rho), float(p)


def run_conflict_and_stability() -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Category-level MDS sensitivity, rank correlations and the label-stability table."""
    analysis = json.loads(config.WVS_ANALYSIS.read_text(encoding="utf-8"))
    label_map = label_to_number(json.loads(config.CODEBOOK.read_text(encoding="utf-8")))

    questions, stability = [], []
    for q_id, info in analysis.items():
        scores, rows = analyse_question(q_id, info, label_map.get(q_id, {}))
        stability.extend(rows)
        if scores:
            questions.append(scores)
    stability = pd.DataFrame(stability)

    categories = (pd.DataFrame(questions).groupby("category", dropna=False)
                  .agg(n_questions=("q_id", "count"), mds_min_sc=("mds_min_sc", "mean"),
                       mds_observed_m=("mds_observed_m", "mean"),
                       ordinal_pairwise_wd=("ordinal_pairwise_wd", "mean"))
                  .reset_index().sort_values("mds_min_sc", ascending=False))

    corr = {}
    for key, other in (("spearman_mds_vs_ordinal_wd", "ordinal_pairwise_wd"),
                       ("spearman_mds_denom_sensitivity", "mds_observed_m")):
        clean = categories.dropna(subset=["mds_min_sc", other])
        rho, p = spearman(clean["mds_min_sc"], clean[other])
        corr[key] = {"rho": rho, "p": p, "n_categories": len(clean)}
    return categories, stability, corr


def label_stability(path) -> dict:
    """Spearman correlation between subgroup size N and mode margin, over all pairs.

    Computed from the saved CSV, as in the original analysis: pandas' default float parser
    shifts some margins by one unit in the last place, which changes a few ties. The
    in-memory values give rho = -0.006558 (p = 0.3217); both round to the reported
    rho = -0.007 (p = 0.32). Rounded to 6 decimals, as in the published file.
    """
    saved = pd.read_csv(path)
    rho, p = spearman(saved["n"], saved["mode_margin"])
    return {"rho": round(rho, 6), "p": round(p, 6), "n_pairs": len(saved)}


# --- Paper tables --------------------------------------------------------------


def signed(x: float) -> str:
    """'+.012' / '\\textminus.009'; values that round to zero print as '+.000', as in the paper."""
    return fmt(round(x, 3) + 0.0, signed=True).replace("-", r"\textminus")


def bootstrap_tables(ci: pd.DataFrame) -> None:
    """tab:bootstrap_ci (accuracy) and tab:bootstrap_ci_oe (Value / Overall WR)."""
    label = {base.safe_name: base.label for base, _ in config.BASE_SFT_PAIRS}
    ci = ci.assign(model=ci["model"].map(label))
    acc = ci[ci["metric"] == "accuracy"].sort_values("delta")  # paper orders rows by accuracy gain
    order = list(acc["model"])

    acc_table = acc[["model", "base", "tuned", "delta", "ci_lo", "ci_hi"]].rename(columns={"tuned": "sft"})
    body = [f"{r.model} & {fmt(r.base)} & {fmt(r.sft)} & {fmt(r.delta, signed=True)} & "
            f"[{fmt(r.ci_lo)}, {fmt(r.ci_hi)}] \\\\" for r in acc_table.itertuples()]
    latex = "\n".join([r"\begin{tabular}{@{} l c c c c @{}}", r"\toprule",
                       r"\textbf{Model} & \textbf{Base} & \textbf{SFT} & $\boldsymbol{\Delta}$ & \textbf{95\% CI} \\",
                       r"\midrule", *body, r"\bottomrule", r"\end{tabular}"])
    save_table("bootstrap_ci", acc_table, latex)

    oe = (ci[ci["metric"].isin(["value_wr", "overall_wr"])]
          .pivot(index="model", columns="metric", values=["delta", "ci_lo", "ci_hi"]).loc[order])
    oe.columns = [f"{metric}_{stat}" for stat, metric in oe.columns]
    oe = oe.reset_index()[["model", "value_wr_delta", "value_wr_ci_lo", "value_wr_ci_hi",
                           "overall_wr_delta", "overall_wr_ci_lo", "overall_wr_ci_hi"]]
    body = [f"{r.model} & {signed(r.value_wr_delta)} & [{signed(r.value_wr_ci_lo)}, {signed(r.value_wr_ci_hi)}] & "
            f"{signed(r.overall_wr_delta)} & [{signed(r.overall_wr_ci_lo)}, {signed(r.overall_wr_ci_hi)}] \\\\"
            for r in oe.itertuples()]
    latex = "\n".join([r"\begin{tabular}{@{} l l l @{\hskip 4pt} l l @{}}", r"\toprule",
                       r" & \multicolumn{2}{c}{\textbf{Value WR}} & \multicolumn{2}{c}{\textbf{Overall WR}} \\",
                       r"\cmidrule(lr){2-3} \cmidrule(lr){4-5}",
                       r"\textbf{Model} & \textbf{$\Delta$} & \textbf{95\% CI} & \textbf{$\Delta$} & \textbf{95\% CI} \\",
                       r"\midrule", *body, r"\bottomrule", r"\end{tabular}"])
    save_table("bootstrap_ci_oe", oe, latex)


def mds_table(categories: pd.DataFrame, corr: dict) -> None:
    """tab:mds_ordinal_comparison."""
    def p_text(p: float) -> str:
        return "p < 0.001" if p < 0.001 else f"p = {p:.2f}"

    denom, ordinal = corr["spearman_mds_denom_sensitivity"], corr["spearman_mds_vs_ordinal_wd"]
    body = [f"{SHORT_CATEGORY.get(r.category, r.category)} & {fmt(r.mds_min_sc)} & "
            f"{fmt(r.mds_observed_m)} & {fmt(r.ordinal_pairwise_wd)} \\\\" for r in categories.itertuples()]
    latex = "\n".join([
        r"\begin{tabular}{@{} p{3.4cm} c c c @{}}", r"\toprule",
        r"\textbf{Value Category} & \textbf{MDS} & \textbf{MDS} & \textbf{Ordinal} \\",
        r"& $\min(|S|,|C|)$ & $|M|$ & \textbf{Wass.} \\", r"\midrule", *body, r"\midrule",
        rf"\multicolumn{{4}}{{@{{}}l}}{{\footnotesize Spearman $\rho = {denom['rho']:.3f}$ (${p_text(denom['p'])}$) between the two MDS}} \\",
        r"\multicolumn{4}{@{}l}{\footnotesize denominators $\min(|S|,|C|)$ and $|M|$;} \\",
        rf"\multicolumn{{4}}{{@{{}}l}}{{\footnotesize $\rho = {ordinal['rho']:.3f}$ (${p_text(ordinal['p'])}$) between MDS and ordinal Wass.}} \\",
        r"\bottomrule", r"\end{tabular}"])
    save_table("mds_ordinal_comparison", categories[["category", "mds_min_sc", "mds_observed_m", "ordinal_pairwise_wd"]], latex)


def main():
    config.ROBUSTNESS_DIR.mkdir(parents=True, exist_ok=True)

    print(f"[1/2] Paired bootstrap CIs ({N_BOOT} resamples, {SPLIT})")
    ci = run_bootstrap()
    ci.to_csv(config.ROBUSTNESS_DIR / "bootstrap_ci_main_results.csv", index=False)
    bootstrap_tables(ci)

    print("[2/2] MDS sensitivity, ordinal divergence and label stability")
    categories, stability, corr = run_conflict_and_stability()
    stability_csv = config.ROBUSTNESS_DIR / "label_stability_summary.csv"
    stability.to_csv(stability_csv, index=False)
    categories.to_csv(config.ROBUSTNESS_DIR / "mds_sensitivity_by_category.csv", index=False)
    corr["spearman_n_vs_mode_margin"] = stab = label_stability(stability_csv)
    # No trailing newline, matching the published file byte for byte.
    (config.ROBUSTNESS_DIR / "mds_vs_ordinal_metric_rankcorr.json").write_text(json.dumps(corr, indent=2), encoding="utf-8")
    mds_table(categories, corr)

    print(f"  label stability: Spearman rho(N, mode margin) = {stab['rho']:.3f} "
          f"(p = {stab['p']:.2f}) over {stab['n_pairs']:,} (question, subgroup) pairs")
    print(f"Done. Outputs in {config.ROBUSTNESS_DIR}")


if __name__ == "__main__":
    main()
