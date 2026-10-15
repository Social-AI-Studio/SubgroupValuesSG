"""Builds the paper's result tables from the evaluation summaries and scored responses.

Inputs:  results/final_scores/<model>_evaluation_summary.json   (only "results_summary" is read)
         results/final_scores/<model>_<split>_canonical_scores.csv  (refusals)
         data/wvs_SG_preference_dataset.csv  (subgroup N), data/question_metadata.json
Outputs: outputs/tables/<paper label>.{csv,tex}   CSV at full precision, LaTeX as printed
Paper:   results_ood_simplified, disparity_ood, subgroup_impact, results_train_split,
         disparity_train_split, refused_questions, refusal_rates, sft_impact_train_*,
         sft_impact_ood_age_religion.
Runtime: about 10 seconds.

Rounding follows the paper. In the main results, disparity and main-text subgroup impact
tables a change is the difference of the two printed (3-decimal) values. In the appendix
subgroup impact tables it is rounded from the raw difference. Ranks are taken on the printed
changes so that equal printed values share a rank. Bold marks the best printed value(s) per
column within a model group; averages are underlined.

The refusal tables are corrected: the original notebook also counted each base model's
SFT responses under the base model, which changed the Llama-3.1-8B counts and halved the
question-level rates. The README lists the published values.
"""

import json

import pandas as pd

from svalues import config
from svalues.tables import fmt, save_table

# Metric -> (path inside results_summary[split], higher is better).
PERFORMANCE = {
    "accuracy": (("numerical", "accuracy"), True),
    "nmae": (("numerical", "normalized_mae"), False),
    "persona_wr": (("open-ended", "persona_win_rate", "mean"), True),
    "value_wr": (("open-ended", "value_win_rate", "mean"), True),
    "overall_wr": (("open-ended", "overall_win_rate", "mean"), True),
}
DISPARITY = {
    "acc_range": (("overall_model_bias", "avg_Accuracy_bias_range"), False),
    "acc_cv": (("overall_model_bias", "avg_Accuracy_bias_cv"), False),
    "nmae_range": (("overall_model_bias", "avg_NMAE_bias_range"), False),
    "nmae_cv": (("overall_model_bias", "avg_NMAE_bias_cv"), False),
}

# Only the main OOD table cites each model.
CITATIONS = {
    "Llama-3.1-8B": "IntroducingLlama312024",
    "Llama-3.2-3B": "Llama32Revolutionizing2024a",
    "SEA-LION-v3-8B": "ngSEALIONSoutheastAsian2025",
    "Phi-4-mini": "microsoftPhi4miniTechnicalReport2025",
    "Qwen2.5-7B": "qwenQwen25TechnicalReport2025",
    "Sailor2-8B": "douSailor2SailingSoutheast2025",
    "SeaLLMs-v3-7B": "zhangSeaLLMs3Open2025",
    "GPT-4.1": "IntroducingGPT41API2024",
    "GPT-4o": "HelloGPT4o2024",
    "GPT-4.1-mini": "IntroducingGPT41API2024",
    "GPT-4o-mini": "HelloGPT4o2024",
}
# The single-column tables (disparity_ood, refusal_rates) use shorter names.
SHORT_NAMES = {"SEA-LION-v3-8B": "SEA-LION-v3", "SeaLLMs-v3-7B": "SeaLLMs-v3"}

PERFORMANCE_HEADER = [
    r"\begin{tabular}{l|cc|cc|cc|cc|cc}",
    r"\toprule",
    r"\multirow{3}{*}{\textbf{Model}} & \multicolumn{4}{c|}{\textbf{Structured Numerical Eval}} & "
    r"\multicolumn{6}{c}{\textbf{Open-Ended Eval Win Rate (WR) \% vs. GPT-4.1 (↑)}} \\",
    r"\cmidrule(lr){2-5} \cmidrule(lr){6-11}",
    r"& \multicolumn{2}{c|}{\textbf{Accuracy (↑)}} & \multicolumn{2}{c|}{\textbf{NMAE (↓)}} & "
    r"\multicolumn{2}{c|}{\textbf{Persona WR}} & \multicolumn{2}{c|}{\textbf{Value WR}} & "
    r"\multicolumn{2}{c}{\textbf{Overall WR}} \\",
    r"& " + " & ".join([r"\textbf{Base} & \textbf{SFT}"] * 5) + r" \\",
    r"\midrule",
]
DISPARITY_HEADER = [
    r"\begin{tabular}{l|cc|cc|cc|cc}",
    r"\toprule",
    r"\multirow{3}{*}{\textbf{Model}} & \multicolumn{4}{c|}{\textbf{Accuracy Disparity (↓)}} & "
    r"\multicolumn{4}{c}{\textbf{NMAE Disparity (↓)}} \\",
    r"\cmidrule(lr){2-5} \cmidrule(lr){6-9}",
    r"& \multicolumn{2}{c|}{\textbf{Norm. Range}} & \multicolumn{2}{c|}{\textbf{CV}} & "
    r"\multicolumn{2}{c|}{\textbf{Norm. Range}} & \multicolumn{2}{c}{\textbf{CV}} \\",
    r"& " + " & ".join([r"\textbf{Base} & \textbf{SFT}"] * 4) + r" \\",
    r"\midrule",
]

# Subgroup label abbreviations differ between the tables in the paper:
# style -> (replacements per label part, separator between parts).
ETHNICITY = {"SG: Chinese": "Chinese", "SG: Malay": "Malay"}
LABEL_STYLES = {
    "single": ({**ETHNICITY, "SG: South Asian Indian, Pakistan, etc.": "South Asian",
                "Do not belong to a denomination": "No Rel."}, ""),
    "pair": ({**ETHNICITY, "SG: South Asian Indian, Pakistan, etc.": "Indian", "Buddhist": "Buddh.",
              "Protestant": "Protes.", "Roman Catholic": "Cath.",
              "Do not belong to a denomination": "No Rel."}, r"\_"),
    "main": ({"SG: Chinese": "Chin.", "SG: Malay": "Malay", "SG: South Asian Indian, Pakistan, etc.": "Indian",
              "Buddhist": "Buddh.", "Protestant": "Protes.", "Roman Catholic": "Catholic",
              "Do not belong to a denomination": "No Rel."}, " "),
}

# (paper label, split, strata, label style, first column spec, Δ = difference of printed values)
IMPACT_TABLES = [
    ("subgroup_impact", "eval_ood_subgroup", ["age_x_ethnicity", "ethnicity_x_religion"], "main", "p{2.6cm}", True),
    ("sft_impact_train_sex", "train", ["sex"], "single", "l", False),
    ("sft_impact_train_age", "train", ["age_group"], "single", "l", False),
    ("sft_impact_train_ethnicity", "train", ["ethnicity"], "single", "l", False),
    ("sft_impact_train_religion", "train", ["religion"], "single", "p{2.8cm}", False),
    ("sft_impact_train_sex_age", "train", ["sex_x_age"], "pair", "p{2.7cm}", False),
    ("sft_impact_train_sex_ethnicity", "train", ["sex_x_ethnicity"], "pair", "p{2.8cm}", False),
    ("sft_impact_train_sex_religion", "train", ["sex_x_religon"], "pair", "p{2.7cm}", False),
    ("sft_impact_ood_age_religion", "eval_ood_subgroup", ["age_x_religion"], "pair", "p{2.65cm}", False),
]

# Abbreviated question texts of the refused_questions table, in the order of the published
# table. Rows with equal refusal rates keep this order (Q36 is new in the corrected table).
QUESTION_SHORT = {
    "Q204": "How often do you use... Mobile phone?",
    "Q206": "How often do you use... Internet?",
    "Q182": "...how justifiable is homosexuality?",
    "Q189": "...how justifiable is it for a man to beat his wife?",
    "Q190": "...how justifiable is it for parents to beat their children?",
    "Q183": "...how justifiable is prostitution?",
    "Q184": "...how justifiable is abortion?",
    "Q138": "...how often do... Sexual harassment occur?",
    "Q192": "...how justifiable is terrorism...?",
    "Q186": "...how justifiable is sex before marriage?",
    "Q179": "...how justifiable is it to steal property?",
    "Q205": "How often do you use... Email?",
    "Q111": "Protecting environment vs. Economic growth...",
    "Q180": "...how justifiable is it to cheat on taxes...?",
    "Q207": "How often do you use... Social media...?",
    "Q188": "...how justifiable is euthanasia?",
    "Q36": "...agree: Homosexual couples are as good parents as other couples?",
}
REFUSED_QUESTIONS_SHOWN = 17  # keeps Q111, which the text discusses


# --- Formatting helpers -------------------------------------------------------


def bold(text: str, on: bool = True) -> str:
    return rf"\textbf{{{text}}}" if on else text


def underline(text: str) -> str:
    return rf"\underline{{{text}}}"


def latex_row(cells: list[str]) -> str:
    return " & ".join(cells) + r" \\"


def section_row(title: str, n_cols: int) -> list[str]:
    return [rf"\multicolumn{{{n_cols}}}{{l}}{{\textit{{{title}}}}} \\", r"\midrule"]


def best(values: pd.Series, higher_is_better: bool) -> pd.Series:
    """Marks the best value(s) as printed (3 decimals), so printed ties are all bold."""
    shown = values.round(3)
    return shown == (shown.max() if higher_is_better else shown.min())


def best_marks(frame: pd.DataFrame, higher: dict, on: bool = True) -> pd.DataFrame:
    """best() for every column of a models x metrics frame; all False when bolding is off."""
    return pd.DataFrame({col: best(frame[col], higher[col]) & on for col in frame.columns})


def display_name(label: str, cite: bool) -> str:
    return rf"{label} \citeyearpar{{{CITATIONS[label]}}}" if cite else label


def printed_delta(pre, post):
    """Change as printed in the main tables: the difference of the two 3-decimal values."""
    return (post.round(3) - pre.round(3)).round(3)


def change(pre, post, printed: bool):
    """Δ shown in a table: the difference of the printed values, or the raw difference."""
    return printed_delta(pre, post) if printed else post - pre


def percent(count: int, total: int, digits: int = 2) -> str:
    return rf"{count / total:.{digits}%}".replace("%", r"\%")


# --- Data loading -------------------------------------------------------------


def load_summaries() -> dict:
    """results_summary of every model, keyed by the model's file name."""
    summaries = {}
    for model in config.ALL_MODELS:
        path = config.summary_file(model.safe_name)
        summaries[model.safe_name] = json.loads(path.read_text(encoding="utf-8"))["results_summary"]
    return summaries


def scores(summaries: dict, models: list, split: str, metrics: dict) -> pd.DataFrame:
    """One row per model (indexed by paper label) with the requested metrics for a split."""
    rows = {}
    for model in models:
        rows[model.label] = {}
        for name, (path, _) in metrics.items():
            value = summaries[model.safe_name][split]
            for key in path:
                value = value[key]
            rows[model.label][name] = value
    return pd.DataFrame.from_dict(rows, orient="index")


def subgroup_sizes() -> pd.Series:
    """N per subgroup as printed in the paper: sample_n of the subgroup's first sample (Q1).

    sample_n is the number of respondents who answered that question, so it can differ by a
    few from the subgroup size in the subgroup_counts_full table.
    """
    df = pd.read_csv(config.PREFERENCE_DATASET, usecols=["subgroup", "sample_n"])
    return df.groupby("subgroup", sort=False)["sample_n"].first()


def load_refusals() -> pd.DataFrame:
    """One row per scored sample of every model: model label, SFT flag, split, question, refused.

    A refusal is a response with no parsable number (parsed_numerical_answer == -1), the
    definition used for the published tables. Each model is read from its own files only.
    """
    frames = []
    for model in config.ALL_MODELS:
        for split in config.SPLITS:
            df = pd.read_csv(config.scores_file(model.safe_name, split), usecols=["q_id", "parsed_numerical_answer"], encoding="utf-8-sig")
            frames.append(pd.DataFrame({
                "model": model.label, "is_sft": model.is_sft, "is_closed": model in config.CLOSED_MODELS,
                "split": split, "q_id": df["q_id"], "refused": df["parsed_numerical_answer"] == -1,
            }))
    return pd.concat(frames, ignore_index=True)


# --- Main results and disparity tables ------------------------------------------


def comparison_tables(summaries: dict, split: str, metrics: dict):
    """Base, SFT and closed-source scores (models x metrics) for one split."""
    return (scores(summaries, config.BASE_MODELS, split, metrics),
            scores(summaries, config.SFT_MODELS, split, metrics),
            scores(summaries, config.CLOSED_MODELS, split, metrics))


def comparison_csv(base: pd.DataFrame, sft: pd.DataFrame, closed: pd.DataFrame) -> pd.DataFrame:
    """Full-precision version of a Base/SFT table, averages included."""
    open_source = pd.concat([base.add_suffix("_base"), sft.add_suffix("_sft"),
                             (sft - base).add_suffix("_delta")], axis=1)
    open_source.loc["Average"] = open_source.mean()
    closed = closed.add_suffix("_base")
    closed.loc["Average"] = closed.mean()
    return pd.concat([open_source.assign(group="open-source"), closed.assign(group="closed-source")]) \
        .rename_axis("model").reset_index()


def comparison_latex(base, sft, closed, metrics: dict, header: list[str], closed_title: str,
                     with_bold: bool, cite: bool) -> str:
    """Base vs SFT (change) table with an open-source and a closed-source block."""
    higher = {name: hib for name, (_, hib) in metrics.items()}
    delta = printed_delta(base, sft)
    marks = {key: best_marks(frame, higher, with_bold) for key, frame in
             (("base", base), ("sft", sft), ("delta", delta), ("closed", closed))}
    n_cols = 1 + 2 * len(metrics)

    lines = header + section_row("Open-Source Models", n_cols)
    for label in base.index:
        cells = [display_name(label, cite)]
        for m in metrics:
            sft_text = bold(fmt(sft.at[label, m]), marks["sft"].at[label, m])
            delta_text = bold(fmt(delta.at[label, m], signed=True), marks["delta"].at[label, m])
            cells += [bold(fmt(base.at[label, m]), marks["base"].at[label, m]), f"{sft_text} ({delta_text})"]
        lines.append(latex_row(cells))

    # The average change is again the difference of the printed averages.
    avg_base, avg_sft = base.mean(), sft.mean()
    avg_delta = printed_delta(avg_base, avg_sft)
    cells = [r"\textbf{Average}"]
    for m in metrics:
        cells += [underline(fmt(avg_base[m])),
                  f"{underline(fmt(avg_sft[m]))} ({underline(fmt(avg_delta[m], signed=True))})"]
    lines += [r"\midrule", latex_row(cells), r"\midrule"] + section_row(closed_title, n_cols)

    for label in closed.index:
        cells = [display_name(label, cite)]
        for m in metrics:
            cells += [bold(fmt(closed.at[label, m]), marks["closed"].at[label, m]), "--"]
        lines.append(latex_row(cells))
    avg_closed = closed.mean()
    cells = [r"\textbf{Average}"] + [cell for m in metrics for cell in (underline(fmt(avg_closed[m])), "--")]
    lines += [r"\midrule", latex_row(cells), r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines)


def disparity_ood_latex(base, sft, closed) -> str:
    """The compact OOD disparity table: one block per open-source model, then closed-source models."""
    lower = {metric: False for metric in DISPARITY}
    marks = {key: best_marks(frame, lower) for key, frame in (("base", base), ("sft", sft), ("closed", closed))}

    def pair(label, metric, avg=False):
        if avg:
            return rf"{underline(fmt(base[metric].mean()))} $\rightarrow$ {underline(fmt(sft[metric].mean()))}"
        return (rf"{bold(fmt(base.at[label, metric]), marks['base'].at[label, metric])} $\rightarrow$ "
                rf"{bold(fmt(sft.at[label, metric]), marks['sft'].at[label, metric])}")

    lines = [
        r"\begin{tabular}{@{}l l | c | c@{}}",
        r"\toprule",
        r"\multirow{2}{*}{\textbf{Model}} & \multirow{2}{*}{\textbf{Metric}} & "
        r"\multicolumn{2}{c}{\textbf{Perf. Disparity (Base $\rightarrow$ SFT)}} \\",
        r"\cmidrule(lr){3-4}",
        r"& & \textbf{Norm. Range (↓)} & \textbf{CV (↓)} \\",
        r"\midrule",
    ]
    for label in list(base.index) + ["Average"]:
        avg = label == "Average"
        lines += [rf"\multirow{{2}}{{*}}{{{bold('Average') if avg else SHORT_NAMES.get(label, label)}}}",
                  latex_row(["", "Acc.", pair(label, "acc_range", avg), pair(label, "acc_cv", avg)]),
                  latex_row(["", "NMAE", pair(label, "nmae_range", avg), pair(label, "nmae_cv", avg)]),
                  r"\midrule" if not avg else r"\bottomrule"]
    lines += [
        r"\end{tabular}",
        "",
        r"\begin{tabular}{l c c c c}",
        r"\toprule",
        r"\multirow{2}{*}{\textbf{Closed-Source}} & \multicolumn{2}{c}{\textbf{Acc. Disparity (↓)}} & "
        r"\multicolumn{2}{c}{\textbf{NMAE Disparity (↓)}} \\",
        r"\cmidrule(lr){2-3} \cmidrule(lr){4-5}",
        r"& \textbf{Norm. Range} & \textbf{CV} & \textbf{Norm. Range} & \textbf{CV} \\",
        r"\midrule",
    ]
    for label in closed.index:
        lines.append(latex_row([label] + [bold(fmt(closed.at[label, m]), marks["closed"].at[label, m])
                                          for m in DISPARITY]))
    lines += [r"\midrule",
              latex_row([r"\textbf{Average}"] + [underline(fmt(closed[m].mean())) for m in DISPARITY]),
              r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines)


def write_main_tables(summaries: dict) -> dict:
    """Writes the results and disparity tables of both splits; returns the OOD frames."""
    ood = {}
    for split in ("eval_ood_subgroup", "train"):
        is_ood = split == "eval_ood_subgroup"
        perf = comparison_tables(summaries, split, PERFORMANCE)
        disp = comparison_tables(summaries, split, DISPARITY)
        # As in the paper, only the OOD results table has bold best values and citations, and
        # the train table's header omits "(WR)".
        header = PERFORMANCE_HEADER if is_ood else [line.replace(" (WR)", "") for line in PERFORMANCE_HEADER]
        save_table("results_ood_simplified" if is_ood else "results_train_split", comparison_csv(*perf),
                   comparison_latex(*perf, PERFORMANCE, header,
                                    "Closed-Source Models" if is_ood else "Closed-Source Models (Reference)",
                                    with_bold=is_ood, cite=is_ood))
        if is_ood:
            save_table("disparity_ood", comparison_csv(*disp), disparity_ood_latex(*disp))
            ood = {"performance": perf, "disparity": disp}
        else:
            save_table("disparity_train_split", comparison_csv(*disp),
                       comparison_latex(*disp, DISPARITY, DISPARITY_HEADER, "Closed-Source Models (Reference)",
                                        with_bold=False, cite=False))
    return ood


# --- Subgroup impact tables -----------------------------------------------------


def subgroup_means(summaries: dict, models: list, split: str, stratum: str) -> pd.DataFrame:
    """Accuracy and NMAE per subgroup, averaged over the given models."""
    rows = []
    for model in models:
        for subgroup, metrics in summaries[model.safe_name][split]["subgroups"][stratum].items():
            rows.append({"subgroup": subgroup, "acc": metrics["numerical"]["accuracy"],
                         "nmae": metrics["numerical"]["normalized_mae"]})
    return pd.DataFrame(rows).groupby("subgroup").mean()


def impact_frame(summaries: dict, sizes: pd.Series, split: str, stratum: str, printed: bool) -> pd.DataFrame:
    """Pre- vs post-SFT scores per subgroup, averaged over the open-source models."""
    pre = subgroup_means(summaries, config.BASE_MODELS, split, stratum)
    post = subgroup_means(summaries, config.SFT_MODELS, split, stratum)
    df = pd.DataFrame({"stratum": stratum, "subgroup": pre.index, "n": sizes[pre.index].to_numpy(),
                       "acc_pre": pre["acc"], "acc_post": post["acc"],
                       "nmae_pre": pre["nmae"], "nmae_post": post["nmae"]})
    # Δ rule as in the paper: the main-text table (printed=True) shows the difference of the
    # two printed values, like the main results tables; the appendix tables show the raw
    # difference, rounded when printed.
    df["acc_delta"] = change(df["acc_pre"], df["acc_post"], printed)
    df["nmae_delta"] = change(df["nmae_pre"], df["nmae_post"], printed)
    # Rank 1 = largest improvement. Ranking the printed (3-decimal) changes gives equal
    # printed values the same rank, as in the paper.
    df["acc_rank"] = df["acc_delta"].round(3).rank(method="min", ascending=False).astype(int)
    df["nmae_rank"] = df["nmae_delta"].round(3).rank(method="min").astype(int)
    # Sorted by pre-SFT accuracy so that the initially most disadvantaged subgroups come first.
    # Accuracies are multiples of 1/214, so exact ties occur; rounding away floating-point noise
    # breaks them alphabetically (in the notebook the noise decided, see README).
    order = df["acc_pre"].round(12).sort_values(kind="stable").index
    return df.loc[order].reset_index(drop=True)


def short_label(subgroup: str, style: str) -> str:
    """Paper label of a subgroup, e.g. '35-44 years_SG: Malay' -> '35-44 Malay'."""
    replacements, separator = LABEL_STYLES[style]
    parts = [part.replace(" years", "").replace("65 and over", "65+") for part in subgroup.split("_")]
    return separator.join(replacements.get(part, part) for part in parts)


def impact_rows(df: pd.DataFrame, style: str, printed: bool) -> list[str]:
    """Table rows of one stratum: one line per subgroup plus the underlined average."""
    higher = {"acc_pre": True, "acc_post": True, "acc_delta": True,
              "nmae_pre": False, "nmae_post": False, "nmae_delta": False}
    marks = {col: best(df[col], hib) for col, hib in higher.items()}

    def cell(i, col):
        return bold(fmt(df.at[i, col], signed=col.endswith("delta")), marks[col][i])

    lines = []
    for i in df.index:
        lines.append(latex_row([
            f"{short_label(df.at[i, 'subgroup'], style)} ({df.at[i, 'n']})",
            cell(i, "acc_pre"), f"{cell(i, 'acc_post')} ({cell(i, 'acc_delta')})$^{{{df.at[i, 'acc_rank']}}}$",
            cell(i, "nmae_pre"), f"{cell(i, 'nmae_post')} ({cell(i, 'nmae_delta')})$^{{{df.at[i, 'nmae_rank']}}}$",
        ]))
    avg = df[["acc_pre", "acc_post", "nmae_pre", "nmae_post"]].mean()
    acc_change = change(avg["acc_pre"], avg["acc_post"], printed)
    nmae_change = change(avg["nmae_pre"], avg["nmae_post"], printed)
    lines += [r"\midrule", latex_row([
        underline("Average"), underline(fmt(avg["acc_pre"])),
        underline(f"{fmt(avg['acc_post'])} ({fmt(acc_change, signed=True)})"),
        underline(fmt(avg["nmae_pre"])),
        underline(f"{fmt(avg['nmae_post'])} ({fmt(nmae_change, signed=True)})"),
    ])]
    return lines


def write_impact_table(summaries: dict, sizes: pd.Series, name: str, split: str, strata: list[str],
                       style: str, first_column: str, printed: bool) -> None:
    """Writes one subgroup impact table (one block per stratum)."""
    frames = [impact_frame(summaries, sizes, split, stratum, printed) for stratum in strata]
    lines = [
        rf"\begin{{tabular}}{{@{{}} {first_column} | c c | c c @{{}}}}",
        r"\toprule",
        r"\multirow{2}{*}{\textbf{Subgroup (\textbf{\textit{N}})}} & \multicolumn{2}{c|}{\textbf{Accuracy (↑)}} & "
        r"\multicolumn{2}{c}{\textbf{NMAE (↓)}} \\",
        r"\cmidrule(lr){2-3} \cmidrule(lr){4-5}",
        r"& \textbf{Pre} & \textbf{Post ($\Delta$)}$^{\text{Rank}}$ & "
        r"\textbf{Pre} & \textbf{Post ($\Delta$)}$^{\text{Rank}}$ \\",
        r"\midrule",
    ]
    for k, (stratum, df) in enumerate(zip(strata, frames)):
        if len(strata) > 1:
            if k > 0:
                lines.append(r"\midrule")
            escaped = stratum.replace("_", r"\_")
            lines += section_row(f"Eval OOD Stratum: {escaped}", 5)
        lines += impact_rows(df, style, printed)
    lines += [r"\bottomrule", r"\end{tabular}"]
    save_table(name, pd.concat(frames, ignore_index=True), "\n".join(lines))


# --- Refusal tables (corrected) ---------------------------------------------------


def write_refusal_rates(refusals: pd.DataFrame) -> pd.DataFrame:
    """Refusal rate per model and split, base -> SFT for open-source models."""
    counts = (refusals.groupby(["model", "is_sft", "is_closed", "split"], sort=False)["refused"]
              .agg(refused="sum", total="size").reset_index())
    counts["rate"] = counts["refused"] / counts["total"]
    get = counts.set_index(["model", "is_sft", "split"])

    lines = [
        r"\begin{tabular}{l|cc}",
        r"\toprule",
        r"\multirow{2}{*}{\textbf{Model}} & \multicolumn{2}{c}{\textbf{Refusal Rate (Count / Total)}} \\",
        r"\cmidrule(lr){2-3}",
        r"& \textbf{Train Split} & \textbf{OOD Split} \\",
        r"\midrule",
    ] + section_row(r"Open-Source Models (Base $\rightarrow$ SFT)", 3)
    for model in config.BASE_MODELS:
        rates, fractions = [], []
        for split in config.SPLITS:
            pre, post = get.loc[(model.label, False, split)], get.loc[(model.label, True, split)]
            rates.append(rf"{percent(pre.refused, pre.total)} $\rightarrow$ {percent(post.refused, post.total)}")
            fractions.append(rf"\scriptsize{{({pre.refused}/{pre.total}) $\rightarrow$ ({post.refused}/{post.total})}}")
        lines += [latex_row([SHORT_NAMES.get(model.label, model.label)] + rates),
                  latex_row([""] + fractions), r"\midrule"]
    lines += section_row("Closed-Source Models (Base Performance)", 3)
    for model in config.CLOSED_MODELS:
        cells = [model.label]
        for split in config.SPLITS:
            c = get.loc[(model.label, False, split)]
            cells.append(rf"{percent(c.refused, c.total)} \scriptsize{{({c.refused}/{c.total})}}")
        lines.append(latex_row(cells))
    lines += [r"\bottomrule", r"\end{tabular}"]
    save_table("refusal_rates", counts, "\n".join(lines))
    return counts


def write_refused_questions(refusals: pd.DataFrame) -> None:
    """Questions most often refused by the base open-source models (both splits pooled)."""
    base = refusals[~refusals["is_sft"] & ~refusals["is_closed"]]
    counts = (base.groupby(["q_id", "model"], sort=False)["refused"]
              .agg(refused="sum", total="size").reset_index())
    counts = counts[counts["refused"] > 0]
    counts["rate"] = counts["refused"] / counts["total"]
    # Columns: the base models that refused at least once, in paper order.
    models = [m.label for m in config.BASE_MODELS if m.label in set(counts["model"])]

    # One row per question, ordered by its highest refusal rate; ties keep the published order.
    paper_order = {q_id: i for i, q_id in enumerate(QUESTION_SHORT)}
    top = (counts.groupby("q_id")["rate"].max().rename("max_rate").reset_index()
           .assign(tie_break=lambda d: d["q_id"].map(paper_order).fillna(len(paper_order)))
           .sort_values(["max_rate", "tie_break"], ascending=[False, True])
           .head(REFUSED_QUESTIONS_SHOWN))
    shown = counts[counts["q_id"].isin(top["q_id"])]
    metadata = json.loads(config.QUESTION_METADATA.read_text(encoding="utf-8"))

    lines = [
        r"\begin{tabular}{@{} l p{7.0cm} " + "c " * len(models) + "@{}}",
        r"\toprule",
        latex_row([r"\textbf{Q ID}", r"\textbf{Question Text (Abbreviated)}"] + [bold(m) for m in models]),
        r"\midrule",
    ]
    for q_id in top["q_id"]:
        cells = [q_id, QUESTION_SHORT.get(q_id, metadata[q_id]["question"])]
        for model in models:
            hit = shown[(shown["q_id"] == q_id) & (shown["model"] == model)]
            if hit.empty:
                cells.append("-")
            else:
                r = hit.iloc[0]
                cells.append(rf"{percent(r.refused, r.total, 1)} \newline\scriptsize{{({r.refused}/{r.total})}}")
        lines.append(latex_row(cells))
    lines += [r"\bottomrule", r"\end{tabular}"]
    order = {q: i for i, q in enumerate(top["q_id"])}
    csv = shown.assign(order=shown["q_id"].map(order)).sort_values(["order", "model"]).drop(columns="order")
    save_table("refused_questions", csv, "\n".join(lines))


# --- Summary ------------------------------------------------------------------------


def print_headlines(ood: dict, refusal_counts: pd.DataFrame) -> None:
    """Prints the headline numbers quoted in the text of the paper."""
    base, sft, closed = ood["performance"]
    d_base, d_sft, _ = ood["disparity"]
    pre, post = base.mean(), sft.mean()

    def pp(metric):
        """Change of the printed averages in percentage points, as quoted in the text."""
        return f"{100 * (round(post[metric], 3) - round(pre[metric], 3)):+.1f} pp"

    best_sft = sft["accuracy"].idxmax()
    base_refusals = refusal_counts[~refusal_counts["is_sft"] & ~refusal_counts["is_closed"]]
    worst = base_refusals.nlargest(1, "rate").iloc[0]
    print("\nHeadline numbers (OOD split, averages over the open-source models):")
    print(f"  Accuracy {fmt(pre['accuracy'])} -> {fmt(post['accuracy'])} ({pp('accuracy')}); "
          f"NMAE {fmt(pre['nmae'])} -> {fmt(post['nmae'])} ({fmt(post['nmae'] - pre['nmae'], signed=True)})")
    print(f"  Win rate vs GPT-4.1 after SFT: Value {pp('value_wr')}, Overall {pp('overall_wr')}, "
          f"Persona {pp('persona_wr')}")
    print(f"  GPT-4.1 accuracy {closed.at['GPT-4.1', 'accuracy']:.1%}; best SFT model {best_sft} "
          f"{sft.at[best_sft, 'accuracy']:.1%}")
    print(f"  Norm. Range: Accuracy {fmt(d_base['acc_range'].mean())} -> {fmt(d_sft['acc_range'].mean())}, "
          f"NMAE {fmt(d_base['nmae_range'].mean())} -> {fmt(d_sft['nmae_range'].mean())} "
          f"({fmt(d_sft['nmae_range'].mean() - d_base['nmae_range'].mean(), signed=True)})")
    print(f"  Highest base-model refusal rate: {worst['model']} {worst['rate']:.2%} ({worst['split']})")


def main() -> None:
    summaries = load_summaries()
    print("Main results and disparity tables ...")
    ood = write_main_tables(summaries)
    print("Subgroup impact tables ...")
    sizes = subgroup_sizes()
    for spec in IMPACT_TABLES:
        write_impact_table(summaries, sizes, *spec)
    print("Refusal tables (corrected) ...")
    refusals = load_refusals()
    refusal_counts = write_refusal_rates(refusals)
    write_refused_questions(refusals)
    print_headlines(ood, refusal_counts)


if __name__ == "__main__":
    main()
