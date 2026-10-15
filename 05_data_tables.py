"""Builds the paper's data tables (value conflict, category divisiveness, dataset and subgroup counts).

Inputs:  data/all_questions_ranked_by_modal_diversity.json, data/subgroup_counts.json
         (from 01_process_wvs.py), data/wvs_SG_preference_dataset.csv (from 02_build_dataset.py),
         data/codebook.json
Outputs: outputs/tables/{conflict_examples,category_divisiveness,dataset_composition,
         subgroup_counts_full}.{csv,tex}
Paper:   tab:conflict_examples, tab:category_divisiveness, tab:dataset_composition,
         tab:subgroup_counts_full.
"""

import json

import pandas as pd

from svalues import config
from svalues.tables import save_table

# --- Paper wording -------------------------------------------------------------
# The no-conflict examples were picked by hand from the 49 questions with a score of 0.
NO_CONFLICT_EXAMPLES = ["Q127", "Q137", "Q144"]
# Questions the paper paraphrases rather than shortens.
QUESTION_PARAPHRASES = {
    "Q127": "On a scale of 0 to 2... does immigration offer people from poor countries a better living?",
    "Q137": "On a scale of 1 to 4... how often do street violence and fights occur in your neighbourhood?",
}
ANSWER_ABBREVIATIONS = {
    "An essential characteristic of democracy": "Essential",
    "Only on special holy days": "Holy days only",
    "Never, practically never": "Never",
}
CATEGORY_ABBREVIATIONS = {
    "Perceptions about Science and Technology": "Perceptions about Science/Tech",
    "Political Culture and Political Regimes": "Political Culture & Regimes",
    "Political Interest and Political Participation": "Political Interest & Participation",
    "Social Capital, Trust and Organizational Membership": "Social Capital, Trust, Membership",
}
# Subgroup labels: single-axis strata (and sex_x_ethnicity) use the longer forms, the other
# pairwise strata the compact ones, as in the paper.
LONG_LABELS = {
    "South Asian Indian, Pakistan, etc.": "South Asian",
    "Caucasian white": "Caucasian",
    "Do not belong to a denomination": "No Denomination",
}
COMPACT_LABELS = {
    "South Asian Indian, Pakistan, etc.": "Indian",
    "Caucasian white": "Caucasian",
    "Do not belong to a denomination": "No Rel.",
    "Roman Catholic": "Roman Cath.",
    "65 and over": "65+",
}
LONG_LABEL_STRATA = {"sex", "age_group", "ethnicity", "religion", "sex_x_ethnicity"}
# Strata listed in age order; all others are sorted by respondent count.
AGE_ORDER_STRATA = {"sex", "age_group", "sex_x_age"}
# For the widest strata the paper lists only a few of the excluded (N < 30) subgroups.
EXCLUDED_SHOWN = {"age_x_ethnicity": 0, "ethnicity_x_religion": 4, "age_x_religion": 0}
SUBGROUP_TABLE_ORDER = ["sex", "age_group", "ethnicity", "religion", "sex_x_age", "sex_x_ethnicity",
                        "sex_x_religon", "age_x_ethnicity", "ethnicity_x_religion", "age_x_religion"]
# The stratum key keeps the original typo; the paper prints it correctly.
STRATUM_NAMES = {"sex_x_religon": "sex_x_religion"}
SPLIT_NAMES = {"train": "Train", "eval_ood_subgroup": "OOD Eval"}


def tex(text: str) -> str:
    """Escapes LaTeX special characters."""
    for char in "&%_#$":
        text = text.replace(char, "\\" + char)
    return text


def stratum_name(stratum: str) -> str:
    """Stratum name as printed in the paper."""
    return STRATUM_NAMES.get(stratum, stratum)


# --- tab:conflict_examples ------------------------------------------------------
QUOTE_END = "', "  # end of the "... meaning '<label>', " part of a question


def short_question(q_id: str, question: str) -> str:
    """'On a scale of 1 to 4, 1 meaning ... and 4 meaning '...', how ...' -> 'On a scale of 1 to 4... how ...'."""
    if q_id in QUESTION_PARAPHRASES:
        return QUESTION_PARAPHRASES[q_id]
    scale, _, rest = question.partition(", ")
    return f"{scale}... {rest.split(QUOTE_END, 1)[-1]}"


def answer_text(label: str, choices: dict) -> str:
    """'Once a week' -> '2: Once a week'; numeric labels such as '7' stay as they are."""
    code = next(k for k, v in choices.items() if v == label and int(k) >= 0)
    return code if label == code else f"{code}: {ANSWER_ABBREVIATIONS.get(label, label)}"


def short_subgroup(label: str) -> str:
    """'Female_25-34 years' -> 'F(25-34)', 'Male_65 and over' -> 'M(65+)'."""
    sex, age = label.split("_")
    return f"{sex[0]}({age.replace(' years', '').replace(' and over', '+')})"


def modal_answers(entry: dict, choices: dict) -> list[tuple[str, str]]:
    """(answer, subgroups) pairs of the sex_x_age stratum, ordered by answer code."""
    buckets = entry["strata_analysis"]["sex_x_age"]["answer_buckets_with_counts"]
    pairs = [(answer_text(label, choices), ", ".join(short_subgroup(s["subgroup"]) for s in members))
             for label, members in buckets.items()]
    return sorted(pairs, key=lambda pair: int(pair[0].split(":")[0]))


def conflict_examples(ranking: dict, codebook: dict) -> None:
    """Top-3 most divisive questions with their sex_x_age modal answers, plus 3 consensus questions."""
    rows, lines = [], [r"\begin{tabularx}{\linewidth}{@{}l X X r@{}}", r"\toprule",
                       r"\textbf{ID} & \textbf{Question} & \textbf{Modal answers (\texttt{sex\_x\_age})} & \textbf{Score} \\"]
    for kind, q_ids, title in [("high", list(ranking)[:3], "High-conflict questions (high Modal Diversity)"),
                               ("none", NO_CONFLICT_EXAMPLES, "No-conflict questions (zero Modal Diversity)")]:
        lines += [r"\midrule", rf"\multicolumn{{4}}{{@{{}}l}}{{\textbf{{\textit{{{title}}}}}}} \\", r"\midrule"]
        for q_id in q_ids:
            entry = ranking[q_id]
            answers = modal_answers(entry, codebook[q_id]["choices"])
            question, score = short_question(q_id, entry["question"]), entry["average_modal_diversity"]
            rows.append({"conflict": kind, "q_id": q_id, "category": entry["category"], "question": entry["question"],
                         "question_short": question, "modal_answers_sex_x_age": "; ".join(f'"{a}": {s}' for a, s in answers),
                         "modal_diversity": score})
            # Consensus questions show only the single (unanimous) answer.
            cell = (r" \newline ".join(rf'\textbf{{"{tex(a)}"}}: {s}.' for a, s in answers) if kind == "high"
                    else rf'\textbf{{"{tex(answers[0][0])}"}}')
            lines.append(rf"{q_id} & {tex(question)} & {cell} & {score:.3f} \\")
    lines += [r"\bottomrule", r"\end{tabularx}"]
    save_table("conflict_examples", pd.DataFrame(rows), "\n".join(lines))


# --- tab:category_divisiveness --------------------------------------------------
def category_divisiveness(ranking: dict) -> pd.DataFrame:
    """Questions, unanimous questions (score 0) and mean Modal Diversity per value category."""
    scores = pd.DataFrame([{"category": e["category"], "score": e["average_modal_diversity"]} for e in ranking.values()])
    table = (scores.groupby("category")
             .agg(total_questions=("score", "size"), unanimous_questions=("score", lambda s: int((s == 0).sum())),
                  avg_modal_diversity=("score", "mean"))
             .sort_values("avg_modal_diversity", ascending=False).reset_index())
    table.insert(1, "label", table["category"].map(lambda c: CATEGORY_ABBREVIATIONS.get(c, c)))
    # The overall score is the mean over questions, not over categories.
    overall = {"category": "Overall Summary", "label": "Overall Summary", "total_questions": len(scores),
               "unanimous_questions": int((scores["score"] == 0).sum()), "avg_modal_diversity": scores["score"].mean()}
    table = pd.concat([table, pd.DataFrame([overall])], ignore_index=True)

    body = [rf"{tex(r.label)} & {r.total_questions} & {r.unanimous_questions} & {r.avg_modal_diversity:.3f} \\"
            for r in table.iloc[:-1].itertuples()]
    total = table.iloc[-1]
    latex = "\n".join([
        r"\begin{tabularx}{\columnwidth}{@{} X c c c @{}}", r"\toprule",
        r"\textbf{Value Category} & \textbf{Total} & \textbf{Unan.} & \textbf{Avg. Div.} \\",
        r"& \textbf{Qns.} & \textbf{Qns.} & \textbf{(↑)} \\", r"\midrule", *body, r"\midrule",
        rf"\textbf{{Overall Summary}} & \textbf{{{total.total_questions}}} & \textbf{{{total.unanimous_questions}}} "
        rf"& \textbf{{{total.avg_modal_diversity:.3f}}} \\",
        r"\bottomrule", r"\end{tabularx}"])
    save_table("category_divisiveness", table, latex)
    return table


# --- tab:dataset_composition ----------------------------------------------------
def dataset_composition(samples: pd.DataFrame) -> pd.DataFrame:
    """Subgroups and samples per stratum, with split and grand totals."""
    rows, lines = [], [r"\begin{tabularx}{\columnwidth}{@{} X r r @{}}", r"\toprule",
                       r"\textbf{Stratum Definition} & \textbf{Subgroups} & \textbf{Samples} \\"]
    for split, title in [("train", "Train Set"), ("eval_ood_subgroup", "Eval Set")]:
        part = samples[samples["split"] == split]
        # Alphabetical first, then a stable sort by size, so equal sizes stay alphabetical.
        counts = (part.groupby("stratum").agg(subgroups=("subgroup", "nunique"), samples=("sample_id", "size"))
                  .sort_values("samples", ascending=False, kind="stable"))
        heading = "Train Set Strata" if split == "train" else "Evaluation (OOD) Set Strata"
        lines += [r"\midrule", rf"\multicolumn{{3}}{{@{{}}l}}{{\textit{{\textbf{{{heading}}}}}}} \\"]
        for stratum, row in counts.iterrows():
            rows.append({"split": split, "stratum": stratum_name(stratum), "subgroups": row.subgroups, "samples": row.samples})
            lines.append(rf"\quad \texttt{{{tex(stratum_name(stratum))}}} & {row.subgroups} & {row.samples:,} \\")
        rows.append({"split": split, "stratum": "Total", "subgroups": counts.subgroups.sum(), "samples": counts.samples.sum()})
        lines += [r"\cmidrule(lr){2-3}",
                  rf"\textbf{{{title} Total}} & \textbf{{{counts.subgroups.sum()}}} & \textbf{{{counts.samples.sum():,}}} \\"]
    table = pd.DataFrame(rows)
    totals = table[table["stratum"] == "Total"]
    grand = {"split": "all", "stratum": "Grand Total", "subgroups": totals.subgroups.sum(), "samples": totals.samples.sum()}
    table = pd.concat([table, pd.DataFrame([grand])], ignore_index=True)
    lines += [r"\midrule", rf"\textbf{{Grand Total}} & \textbf{{{grand['subgroups']}}} & \textbf{{{grand['samples']:,}}} \\",
              r"\bottomrule", r"\end{tabularx}"]
    save_table("dataset_composition", table, "\n".join(lines))
    return table


# --- tab:subgroup_counts_full ---------------------------------------------------
def paper_label(subgroup: str, stratum: str) -> str:
    """Subgroup label as printed in the paper, e.g. '65 and over_SG: Chinese' -> '65+_Chinese'."""
    long_form = stratum in LONG_LABEL_STRATA
    names = LONG_LABELS if long_form else COMPACT_LABELS
    parts = []
    for part in subgroup.split("_"):
        part = part.removeprefix(f"{config.COUNTRY_CODE}: ")  # ethnicity labels, e.g. "SG: Malay"
        part = names.get(part, part)
        parts.append(part if long_form else part.replace(" years", ""))
    return "_".join(parts)


def subgroup_counts_full(counts: dict, samples: pd.DataFrame) -> pd.DataFrame:
    """Respondents (N) and dataset samples (S) per subgroup, including excluded subgroups."""
    sample_counts = samples.groupby(["stratum", "subgroup"]).size()
    rows, lines = [], [r"\begin{longtable}{@{} l r r @{}}", r"\toprule",
                       r"\textbf{Subgroup} & \textbf{N} & \textbf{S} \\"]
    for stratum in SUBGROUP_TABLE_ORDER:
        info = counts[stratum]
        split = "train" if stratum in config.TRAIN_STRATA else "eval_ood_subgroup"
        sizes = pd.Series(info["subgroups"])  # sorted by label
        if stratum not in AGE_ORDER_STRATA:
            sizes = sizes.sort_values(ascending=False, kind="stable")
        for subgroup, n in sizes.items():
            included = n >= config.MIN_SUBGROUP_SIZE
            rows.append({"stratum": stratum_name(stratum), "split": split, "subgroup": subgroup,
                         "label": paper_label(subgroup, stratum), "respondents": n,
                         "samples": sample_counts[(stratum, subgroup)] if included else pd.NA,
                         "included": included, "stratum_coverage": info["coverage"]})

        block = pd.DataFrame(rows[-len(sizes):])
        excluded = block[~block["included"]]
        shown = EXCLUDED_SHOWN.get(stratum, len(excluded))
        lines += [r"\midrule", rf"\multicolumn{{3}}{{@{{}}l}}{{\textbf{{Stratum: {tex(stratum_name(stratum))}}} "
                               rf"({SPLIT_NAMES[split]})}} \\", r"\midrule"]
        lines += [rf"{tex(r.label)} & {r.respondents} & {int(r.samples)} \\" for r in block[block["included"]].itertuples()]
        lines += [rf"\textcolor{{gray}}{{{tex(r.label)}}} & \textcolor{{gray}}{{{r.respondents}}} & \textcolor{{gray}}{{-}} \\"
                  for r in excluded.head(shown).itertuples()]
        if len(excluded) > shown:
            lines.append(rf"\textcolor{{gray}}{{+{len(excluded) - shown} more dropped}} & & \\")
        coverage = f"{info['coverage'] * 100:.1f}"
        lines.append(rf"\rowcolor{{gray!10}} \multicolumn{{3}}{{@{{}}l}}{{\textit{{Coverage: {coverage}\%}}}} \\")
    lines += [r"\bottomrule", r"\end{longtable}"]
    table = pd.DataFrame(rows).astype({"samples": "Int64"})
    save_table("subgroup_counts_full", table, "\n".join(lines))
    return table


def main() -> None:
    ranking = json.loads(config.QUESTION_RANKING.read_text(encoding="utf-8"))
    counts = json.loads(config.SUBGROUP_COUNTS.read_text(encoding="utf-8"))
    codebook = json.loads(config.CODEBOOK.read_text(encoding="utf-8"))
    samples = pd.read_csv(config.PREFERENCE_DATASET)

    print("Building data tables ...")
    conflict_examples(ranking, codebook)
    categories = category_divisiveness(ranking)
    composition = dataset_composition(samples)
    subgroups = subgroup_counts_full(counts, samples)

    overall = categories.iloc[-1]
    grand = composition.iloc[-1]
    print(f"Questions: {overall.total_questions} ({overall.unanimous_questions} unanimous), "
          f"mean Modal Diversity {overall.avg_modal_diversity:.3f}; most divisive: {categories.label[0]}")
    print(f"Dataset: {grand.samples:,} samples from {grand.subgroups} subgroups "
          f"({int(subgroups.included.sum())} of {len(subgroups)} subgroups have N >= {config.MIN_SUBGROUP_SIZE})")


if __name__ == "__main__":
    main()
