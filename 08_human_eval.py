"""Human validation of the LLM judge: agreement with the judge and between annotators.

Inputs:  data/human_eval/annotations.json   300 anonymised annotations (100 items x annotators A1-A3)
         results/raw_inference/<judge>__judge/  the judge's verdicts on the same comparisons
Outputs: outputs/tables/human_eval_combined.{csv,tex}
Paper:   tab:human_eval_combined (H-AI and H-H accuracy, linear-weighted Cohen's kappa, Likert).
Runtime: a few seconds.

Protocol: annotators first compare responses A and B on each criterion (A / B / Tie), then see
the judge's reasoning and verdict and rate their agreement (1-5 Likert). H-AI averages each
annotator's agreement with the judge; H-H averages the three annotator pairs. Kappa uses linear
weights on the ordered scale A < Tie < B (1 for a match, 0.5 for A vs Tie, 0 for A vs B). The
Likert column averages the annotators' mean ratings.

Sample: 100 open-ended comparisons against the GPT-4.1 baseline, 12 per model for eight models
(6 train, 6 OOD), topped up with 4 random comparisons (seed 42). Each item was shown in one of
the judge's two orders ('shown_order': pos1 = evaluated model as Response A), chosen at random,
and annotators are compared with the judge's verdict for that order.
"""

import json
from itertools import combinations

import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score

from svalues import config, metrics
from svalues.tables import fmt, save_table

ANNOTATIONS = config.HUMAN_EVAL_DIR / "annotations.json"

CRITERIA = ["persona", "value", "overall"]
LABELS = ["A", "Tie", "B"]  # ordinal order used by the linear kappa weights
HUMAN_LABEL = {"Response A": "A", "Response B": "B", "Tie": "Tie"}


def judge_verdicts(items: pd.DataFrame) -> pd.DataFrame:
    """The judge's winners for each item, in the order shown to the annotators."""
    rows = []
    for (model, split, order), group in items.groupby(["model", "split", "shown_order"]):
        path = config.judge_output_file(model, split, order)
        responses = {r["custom_id"]: r["response"] for r in json.loads(path.read_text(encoding="utf-8"))}
        for item in group.itertuples():
            verdict = metrics.parse_judge_response(responses[f"{item.sample_id}_{order}"])
            rows.append({"human_eval_id": item.human_eval_id,
                         **{f"judge_{c}": verdict.get(f"{c}_winner", "Error") for c in CRITERIA}})
    return pd.DataFrame(rows)


def load_annotations() -> pd.DataFrame:
    """One row per (item, annotator) with human labels, judge labels and Likert ratings."""
    df = pd.DataFrame(json.loads(ANNOTATIONS.read_text(encoding="utf-8")))
    items = df.drop_duplicates("human_eval_id")
    df = df.merge(judge_verdicts(items), on="human_eval_id", validate="many_to_one")
    for c in CRITERIA:
        df[f"human_{c}"] = df[f"human_{c}_winner"].map(HUMAN_LABEL)
        df[f"likert_{c}"] = df[f"{c}_agreement_rating"].str.split(":").str[0].astype(int)  # "4: Agree" -> 4
    return df


def agreement(y1: pd.Series, y2: pd.Series) -> tuple[float, float]:
    """Exact-match accuracy and linear-weighted Cohen's kappa of two label sequences."""
    y1, y2 = y1.to_numpy(), y2.to_numpy()
    return float((y1 == y2).mean()), cohen_kappa_score(y1, y2, labels=LABELS, weights="linear")


def agreement_table(df: pd.DataFrame) -> pd.DataFrame:
    """H-AI and H-H accuracy / kappa and the mean Likert rating for each criterion."""
    annotators = sorted(df["annotator"].unique())
    rows = []
    for c in CRITERIA:
        h_ai = [agreement(g[f"human_{c}"], g[f"judge_{c}"]) for _, g in df.groupby("annotator")]
        wide = df.pivot(index="human_eval_id", columns="annotator", values=f"human_{c}")
        h_h = [agreement(wide[a], wide[b]) for a, b in combinations(annotators, 2)]
        (acc_ai, kappa_ai), (acc_hh, kappa_hh) = np.mean(h_ai, axis=0), np.mean(h_h, axis=0)
        rows.append({"criterion": c.capitalize(), "accuracy_h_ai": acc_ai, "accuracy_h_h": acc_hh,
                     "kappa_h_ai": kappa_ai, "kappa_h_h": kappa_hh,
                     "likert": df.groupby("annotator")[f"likert_{c}"].mean().mean()})
    return pd.DataFrame(rows)


def save_agreement_table(table: pd.DataFrame) -> None:
    """Writes tab:human_eval_combined."""
    body = [f"{r.criterion} & {fmt(r.accuracy_h_ai)} & {fmt(r.accuracy_h_h)} & {fmt(r.kappa_h_ai)} & "
            f"{fmt(r.kappa_h_h)} & {r.likert:.2f} \\\\" for r in table.itertuples()]
    latex = "\n".join([
        r"\begin{tabular}{l|cc|cc|c}", r"\toprule",
        r"\multirow{2}{*}{\textbf{Criterion}} & \multicolumn{4}{c|}{\textbf{Avg. Pairwise Agreement}} & \textbf{Indiv. Agreement}\\",
        r"\cmidrule(lr){2-6}",
        r"& \multicolumn{2}{c|}{\textbf{Accuracy ($\uparrow$)}} & \multicolumn{2}{c|}{\textbf{w-Kappa ($\uparrow$)}} & \textbf{Likert Rating ($\uparrow$)} \\",
        r"& \textbf{H-AI} & \textbf{H-H} & \textbf{H-AI} & \textbf{H-H} & \textbf{(1-5 Scale)} \\",
        r"\midrule", *body, r"\bottomrule", r"\end{tabular}"])
    save_table("human_eval_combined", table, latex)


def main() -> None:
    print("Human vs judge agreement")
    table = agreement_table(load_annotations())
    print(table.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    save_agreement_table(table)


if __name__ == "__main__":
    main()
