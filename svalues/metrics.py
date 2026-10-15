"""Numerical-task metrics, LLM-judge win rates, and per-subgroup fairness (bias) metrics.

The logic is that of the code that produced the paper's results. A few quirks change the
numbers if "fixed", so they are kept on purpose and marked as such.
"""

import json
import re

import numpy as np
import pandas as pd

REFUSAL = -1

# --- Numerical task ------------------------------------------------------------


def parse_numerical_response(response) -> int:
    """First integer anywhere in the response, or -1 (refusal) if there is none (or no response).

    Taking the first run of digits (not the number after 'Answer:') is what the paper used.
    """
    match = re.search(r"\d+", response) if isinstance(response, str) else None
    return int(match.group(0)) if match else REFUSAL


def numerical_metrics(df: pd.DataFrame, response_col: str = "numerical_response") -> dict:
    """Accuracy, MAE, NMAE and refusal rate against the subgroup modal answer.

    An answer outside the question's scale counts as a refusal. Accuracy and NMAE are taken
    over all samples, with a refusal scored as wrong (accuracy) and as the maximum error of
    1.0 (NMAE); MAE is taken over valid answers only.
    """
    df = df.copy()
    df["parsed_answer"] = df[response_col].apply(parse_numerical_response)
    out_of_range = (df["parsed_answer"] != REFUSAL) & (
        (df["parsed_answer"] < df["choice_min"]) | (df["parsed_answer"] > df["choice_max"])
    )
    df.loc[out_of_range, "parsed_answer"] = REFUSAL

    total = len(df)
    refusal_count = int((df["parsed_answer"] == REFUSAL).sum())
    refusal_rate = refusal_count / total if total > 0 else 0
    valid = df[df["parsed_answer"] != REFUSAL]
    if valid.empty:
        return {"accuracy": 0.0, "mean_absolute_error": float("inf"), "normalized_mae": 1.0,
                "refusal_rate": float(refusal_rate), "refusal_count": refusal_count, "sample_count": total}

    y_true = valid["preferred_answer"].astype(int)
    y_pred = valid["parsed_answer"].astype(int)
    abs_error = np.abs(y_true - y_pred)

    scale_range = df["choice_max"] - df["choice_min"]
    scale_range[scale_range.le(0)] = 1
    # Same operation order as the original (valid errors first, then refusal penalties), so
    # that floating-point sums, and hence the stored values, are identical.
    normalised_errors = np.concatenate([abs_error / scale_range[df["parsed_answer"] != REFUSAL],
                                        np.ones(refusal_count)])
    return {
        "accuracy": float((y_true == y_pred).sum() / total),
        "mean_absolute_error": float(np.mean(abs_error)),
        "normalized_mae": float(np.mean(normalised_errors)),
        "refusal_rate": float(refusal_rate),
        "refusal_count": refusal_count,
        "sample_count": int(total),
    }


# --- Open-ended task (LLM judge) -------------------------------------------------

CRITERIA = ["persona", "value", "overall"]
ERROR = "Error"

# pos1 shows the evaluated model as Response A, pos2 swaps the order (baseline is A).
# A tie, and a verdict that could not be parsed at all ("Error"), both count as half a win.
SCORE_MAP = {
    "pos1": {"A": 1, "B": 0, "Tie": 0.5, ERROR: 0.5},
    "pos2": {"A": 0, "B": 1, "Tie": 0.5, ERROR: 0.5},
}
SCORE_COLUMNS = ([f"{pos}_{c}_score" for pos in SCORE_MAP for c in CRITERIA]
                 + [f"canonical_{c}_score" for c in CRITERIA])


def parse_judge_response(text: str) -> dict:
    """Winners from the first ```json block of a judge response; 'Error' if there is none."""
    match = re.search(r"```json\s*(\{.*?\})\s*```", text, re.DOTALL)
    try:
        return json.loads(match.group(1)) if match else {f"{c}_winner": ERROR for c in CRITERIA}
    except json.JSONDecodeError:
        return {f"{c}_winner": ERROR for c in CRITERIA}


def normalise_winner(value) -> tuple[str, str]:
    """(winner, parse error). Any value other than A/B/Tie/Error is scored as a tie.

    In practice these are verdicts that copy the template literally, e.g. "A/B/Tie".
    """
    if value in ("A", "B", "Tie", ERROR):
        return value, ""
    return "Tie", value


def win_rates(judge_df: pd.DataFrame) -> tuple[dict, pd.DataFrame]:
    """Win rate per criterion against the baseline, averaging the two judge orderings.

    Takes a frame with sample_id, judge_response_pos1 and judge_response_pos2. Returns the
    summary statistics and one row of scores per sample (SCORE_COLUMNS plus parse_error).
    """
    rows = []
    for _, row in judge_df.iterrows():
        scores, errors = {"sample_id": row["sample_id"]}, []
        for pos in SCORE_MAP:
            verdict = parse_judge_response(row[f"judge_response_{pos}"])
            for c in CRITERIA:
                winner, error = normalise_winner(verdict[f"{c}_winner"])
                scores[f"{pos}_{c}_score"] = SCORE_MAP[pos][winner]
                if error:
                    errors.append(f"{pos}_{c}:{error}")
        for c in CRITERIA:
            scores[f"canonical_{c}_score"] = (scores[f"pos1_{c}_score"] + scores[f"pos2_{c}_score"]) / 2
        scores["parse_error"] = "; ".join(errors)
        rows.append(scores)
    per_sample = pd.DataFrame(rows, columns=["sample_id", *SCORE_COLUMNS, "parse_error"])

    summary = {}
    for c in CRITERIA:
        values = per_sample[f"canonical_{c}_score"].to_numpy()
        summary[f"{c}_win_rate"] = {
            "mean": np.mean(values),
            "std_dev": np.std(values),  # population std (ddof=0), as reported
            "median": np.median(values),
            "count": len(values),
        }
    return summary, per_sample


# --- Subgroups and fairness ----------------------------------------------------------

# Metrics whose spread across the subgroups of a stratum is reported as "bias".
BIAS_METRICS = {
    "NMAE_bias": ["numerical", "normalized_mae"],
    "Accuracy_bias": ["numerical", "accuracy"],
    "WinRate_bias": ["open-ended", "overall_win_rate", "mean"],
}


def bias_metrics(subgroup_results: dict, metric_path: list[str]) -> dict:
    """Spread of one metric across subgroups: normalised range (max-min)/max and CV (std/mean)."""
    values = []
    for result in subgroup_results.values():
        value = result
        try:
            for key in metric_path:
                value = value[key]
        except (KeyError, TypeError):
            continue
        if isinstance(value, (int, float)) and not np.isnan(value):
            values.append(value)

    if len(values) < 2:  # a spread needs at least two subgroups
        return {
            "normalized_range": 0.0,
            "coefficient_of_variation": 0.0,
            "std_dev": 0.0,
            "mean": np.mean(values) if values else 0.0,
            "min": np.min(values) if values else 0.0,
            "max": np.max(values) if values else 0.0,
        }

    perf = np.array(values)
    p_max, p_min, p_mean, p_std = perf.max(), perf.min(), perf.mean(), perf.std()
    return {
        "normalized_range": float((p_max - p_min) / p_max if p_max > 0 else 0.0),
        "coefficient_of_variation": float(p_std / p_mean if p_mean > 0 else 0.0),
        "std_dev": float(p_std),
        "mean": float(p_mean),
        "min": float(p_min),
        "max": float(p_max),
    }


def subgroup_metrics(df: pd.DataFrame) -> tuple[dict, dict, dict]:
    """Per-subgroup metrics, per-stratum bias metrics and their average over strata.

    Takes the merged per-sample frame of one split (numerical responses and canonical judge
    scores). Strata and subgroups appear in sorted order, as produced by groupby.
    """
    subgroups, bias_by_stratum = {}, {}
    for stratum, stratum_df in df.groupby("stratum"):
        subgroups[stratum] = {}
        for subgroup, group_df in stratum_df.groupby("subgroup"):
            subgroups[stratum][subgroup] = {
                "numerical": numerical_metrics(group_df),
                "open-ended": {
                    f"{c}_win_rate": {"mean": group_df[f"canonical_{c}_score"].mean()}
                    for c in ["overall", "persona", "value"]
                },
            }
        bias_by_stratum[stratum] = {name: bias_metrics(subgroups[stratum], path)
                                    for name, path in BIAS_METRICS.items()}

    # Model-level bias: unweighted mean over strata.
    overall = {}
    for name in BIAS_METRICS:
        for key, short in [("normalized_range", "range"), ("coefficient_of_variation", "cv")]:
            values = [stratum[name][key] for stratum in bias_by_stratum.values()]
            overall[f"avg_{name}_{short}"] = float(np.mean(values)) if values else 0.0
    return subgroups, bias_by_stratum, overall
