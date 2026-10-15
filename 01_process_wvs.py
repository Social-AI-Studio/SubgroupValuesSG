"""Maps the value landscape: subgroup modal answers and Modal Diversity Scores from the raw WVS file.

Inputs:  data/raw/F00013217-WVS_Wave_7_Singapore_Excel_v5.1.xlsx (not redistributed; see README)
         data/codebook.json, data/question_metadata.json
Outputs: data/singapore_wvs_analysis_rich.json             per question and stratum: subgroup sizes,
                                                           answer distributions, modal answers, scores
         data/all_questions_ranked_by_modal_diversity.json  the 214 questions ranked by Modal Diversity
         data/subgroup_counts.json                          respondents per subgroup (incl. N < 30)
Paper:   Modal Diversity Score (Section 3.2); the data behind the conflict_examples,
         category_divisiveness and subgroup_counts_full tables (built by 05_data_tables.py).
Runtime: about 1 minute.

codebook.json is the WorldValuesBench codebook with the answer labels of Q7-Q26 harmonised from
'Important'/'Mentioned'/'Not mentioned' to 'Yes'/'No', matching the question wording.
"""

import hashlib
import json
import sys
import warnings
from functools import reduce
from itertools import combinations

import numpy as np
import pandas as pd
import scipy.stats
from scipy.stats import wasserstein_distance
from sklearn.manifold import MDS

from svalues import config

# The original run listed "religon" here instead of "religion", so the single-axis religion
# stratum is missing from the ranking file. Kept for byte-identical output.
RANKING_STRATA = [s for s in config.STRATA if s != "religion"]


def check_raw_file() -> None:
    """Stops if the raw survey file is missing; warns if it is not the expected release."""
    if not config.WVS_EXCEL.exists():
        sys.exit(f"Missing {config.WVS_EXCEL.relative_to(config.ROOT)}. Download the WVS Wave 7 "
                 "Singapore file (v5.1) as described in the README (step 2 of 'Reproducing the paper').")
    if hashlib.sha256(config.WVS_EXCEL.read_bytes()).hexdigest() != config.WVS_EXCEL_SHA256:
        print("Warning: the WVS file differs from v5.1; the published files may not be reproduced.")


def load_survey(question_metadata: dict) -> pd.DataFrame:
    """Reads the survey, names columns by variable code and turns negative codes into NaN."""
    # openpyxl warns about the file's missing default style; harmless.
    warnings.filterwarnings("ignore", message="Workbook contains no default style")
    df = pd.read_excel(config.WVS_EXCEL, sheet_name="Data")
    # Columns look like "Q1: Important in life: Family"; the code is the part before the colon.
    df.columns = [col.split(":")[0] for col in df.columns]
    # Negative codes mean "Don't know", "No answer", etc. This also cleans the demographic
    # variables, which are listed in the metadata too.
    for code in question_metadata:
        if code in df.columns and pd.api.types.is_numeric_dtype(df[code]):
            df[code] = df[code].mask(df[code] < 0, np.nan)
    return df


def answer_label(codebook: dict, code: str, value: float) -> str:
    """Human-readable label of a numeric answer code."""
    if code not in codebook or "choices" not in codebook[code]:
        return str(value)
    return codebook[code]["choices"].get(str(int(value)), str(value))


def subgroup_labels(df: pd.DataFrame, codes: list[str], codebook: dict) -> pd.Series:
    """Subgroup label per respondent, e.g. 'Female_SG: Chinese' (rows must have no NaN)."""
    parts = [df[code].map(lambda v, code=code: answer_label(codebook, code, v)) for code in codes]
    return reduce(lambda a, b: a + "_" + b, parts)


def modal_diversity(modal_answers: list, num_choices: int) -> float:
    """Normalised Shannon entropy of the subgroups' modal answers (0 = consensus)."""
    if len(modal_answers) < 2:
        return 0.0
    counts = pd.Series(modal_answers).value_counts()
    entropy = scipy.stats.entropy(counts / counts.sum(), base=2)
    max_entropy = min(len(modal_answers), num_choices)
    if max_entropy <= 1:
        return 0.0
    return entropy / np.log2(max_entropy)


def emd_matrix(distributions: dict, scale_min, scale_max) -> pd.DataFrame:
    """Pairwise Wasserstein distances between subgroup distributions, normalised by scale range."""
    labels = list(distributions)
    # Built as in the original run (empty frame, then the diagonal set to 0): the memory layout
    # decides the summation order of the row means, which shows up in the last float digit.
    matrix = pd.DataFrame(index=labels, columns=labels, dtype=float)
    for label in labels:
        matrix.loc[label, label] = 0.0
    for a, b in combinations(labels, 2):
        p, q = distributions[a], distributions[b]
        emd = wasserstein_distance(p.index, q.index, p.values, q.values)
        valid_scale = scale_min is not None and scale_max is not None and scale_max > scale_min
        matrix.loc[a, b] = matrix.loc[b, a] = emd / (scale_max - scale_min) if valid_scale else 0
    return matrix


def analyse_stratum(df: pd.DataFrame, q_id: str, q_info: dict, codes: list[str], codebook: dict) -> dict | None:
    """Subgroup statistics for one question and stratum, or None if < 2 subgroups have N >= 30."""
    data = df[codes + [q_id]].dropna().copy()
    if data.empty:
        return None
    data["subgroup"] = subgroup_labels(data, codes, codebook)
    sizes = data["subgroup"].value_counts()
    significant = sizes[sizes >= config.MIN_SUBGROUP_SIZE].index
    if len(significant) < 2:
        return None
    data = data[data["subgroup"].isin(significant)]

    answers = sorted(data[q_id].unique())
    grouped = data.groupby("subgroup")[q_id]  # sorted by label: fixes key order in the output
    distributions = {}
    for name, group in grouped:
        counts = group.value_counts().reindex(answers, fill_value=0)
        distributions[name] = counts / counts.sum()

    distances = emd_matrix(distributions, q_info.get("answer_scale_min"), q_info.get("answer_scale_max"))
    labels = list(distributions)
    divergence = distances.values[np.triu_indices(len(labels), k=1)].mean()
    centrality = distances.mean(axis=1).to_dict()
    mds = MDS(n_components=2, dissimilarity="precomputed", random_state=42, normalized_stress=False)
    coords = mds.fit_transform(distances)

    # Series.mode() returns values sorted ascending, so ties go to the lowest answer code.
    modes = {name: group.mode().iloc[0] for name, group in grouped}
    num_choices = len([k for k in codebook[q_id]["choices"] if int(k) >= 0])

    buckets = {}
    for name, mode in modes.items():
        buckets.setdefault(answer_label(codebook, q_id, mode), []).append(name)

    return {
        "total_valid_samples": len(data),
        "subgroup_divergence_score": divergence,
        "modal_diversity_score": modal_diversity(list(modes.values()), num_choices),
        "mds_coordinates": {label: list(coords[i]) for i, label in enumerate(labels)},
        "subgroups": {
            name: {
                "sample_count": len(group),
                "subgroup_centrality_score": centrality.get(name),
                "most_frequent_answer": answer_label(codebook, q_id, modes[name]),
                "answer_distribution": {answer_label(codebook, q_id, k): v
                                        for k, v in distributions[name].to_dict().items() if v > 0},
            }
            for name, group in grouped
        },
        "answer_buckets": buckets,
    }


def analyse_questions(df: pd.DataFrame, question_metadata: dict, codebook: dict) -> dict:
    """Runs analyse_stratum for every value question in the survey and averages the scores."""
    results = {}
    for q_id, q_info in question_metadata.items():
        if q_info.get("use_case") != "value" or q_id not in df.columns:
            continue
        entry = q_info.copy()
        entry["strata_analysis"] = {}
        for stratum, codes in config.STRATA.items():
            stats = analyse_stratum(df, q_id, q_info, codes, codebook)
            if stats is not None:
                entry["strata_analysis"][stratum] = stats
        strata = entry["strata_analysis"].values()
        entry["average_divergence_score"] = np.mean([s["subgroup_divergence_score"] for s in strata]) if strata else 0.0
        entry["average_modal_diversity"] = np.mean([s["modal_diversity_score"] for s in strata]) if strata else 0.0
        results[q_id] = entry
    return results


def rank_questions() -> dict:
    """Questions (without Q7-Q26) ranked by average Modal Diversity, with modal-answer buckets."""
    # Re-read with pandas' default float parser, which rounds some scores in the last digit;
    # this is how the released ranking file was produced.
    data = pd.read_json(config.WVS_ANALYSIS).T
    data = data[~data.index.isin(config.EXCLUDED_QUESTIONS)]
    data["average_modal_diversity"] = pd.to_numeric(data["average_modal_diversity"], errors="coerce").fillna(0)
    # Default (quicksort) ordering decides the order of tied questions, e.g. the 49 zero scores.
    ranked = data.sort_values("average_modal_diversity", ascending=False)

    ranking = {}
    for q_id, row in ranked.iterrows():
        entry = {
            "question": row["question"],
            "category": row.get("category"),
            "average_modal_diversity": row["average_modal_diversity"],
            "average_divergence_score (EMD)": row.get("average_divergence_score"),
            "strata_analysis": {},
        }
        for stratum in RANKING_STRATA:
            stats = row["strata_analysis"].get(stratum)
            if stats is None or "answer_buckets" not in stats:
                continue
            entry["strata_analysis"][stratum] = {
                "modal_diversity_score": stats.get("modal_diversity_score"),
                "answer_buckets_with_counts": {
                    answer: [{"subgroup": name, "sample_count": stats["subgroups"][name]["sample_count"]}
                             for name in names]
                    for answer, names in stats["answer_buckets"].items()
                },
            }
        ranking[q_id] = entry
    return ranking


def count_subgroups(df: pd.DataFrame, codebook: dict) -> dict:
    """Respondents per subgroup (all subgroups, incl. N < 30) and coverage of each stratum."""
    counts = {}
    for stratum, codes in config.STRATA.items():
        data = df[codes].dropna()
        sizes = subgroup_labels(data, codes, codebook).value_counts().sort_index()
        covered = int(sizes[sizes >= config.MIN_SUBGROUP_SIZE].sum())
        counts[stratum] = {
            "respondents": int(sizes.sum()),
            "respondents_in_included_subgroups": covered,
            "coverage": covered / int(sizes.sum()),
            "subgroups": {label: int(n) for label, n in sizes.items()},
        }
    return counts


def write_json(path, data, indent: int) -> None:
    """Writes JSON exactly as the original notebooks did (no trailing newline)."""
    path.write_text(json.dumps(data, indent=indent, ensure_ascii=False), encoding="utf-8")
    print(f"  wrote {path.relative_to(config.ROOT)}")


def main() -> None:
    check_raw_file()
    codebook = json.loads(config.CODEBOOK.read_text(encoding="utf-8"))
    question_metadata = json.loads(config.QUESTION_METADATA.read_text(encoding="utf-8"))

    print("Loading the WVS survey ...")
    df = load_survey(question_metadata)
    print(f"  {len(df)} respondents")

    print("Analysing subgroup answers per question and stratum ...")
    write_json(config.WVS_ANALYSIS, analyse_questions(df, question_metadata, codebook), indent=4)
    write_json(config.QUESTION_RANKING, rank_questions(), indent=4)
    write_json(config.SUBGROUP_COUNTS, count_subgroups(df, codebook), indent=2)


if __name__ == "__main__":
    main()
