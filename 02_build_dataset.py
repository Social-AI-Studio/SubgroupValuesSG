"""Builds the persona-prompted preference dataset and the SFT files from the subgroup analysis.

Inputs:  data/singapore_wvs_analysis_rich.json (from 01_process_wvs.py),
         data/codebook.json, data/question_metadata.json
Outputs: data/wvs_SG_preference_dataset.csv   20,877 (question, subgroup) samples: prompts, modal answer
         data/sft_SG_train_dataset.json       10,700 training conversations (train strata)
         data/sft_SG_eval_dataset.json        500 random OOD conversations (validation loss during SFT)
Paper:   Dataset Construction (Section 3.4) and tab:dataset_composition (built by 05_data_tables.py);
         the evaluation scripts read the CSV directly.
"""

import json
import random

import pandas as pd

from svalues import config
from svalues.prompts import (generate_system_prompt, generate_user_prompt_numerical, generate_user_prompt_open_ended,
                             valid_choices)

EVAL_SUBSET_SIZE = 500


def build_samples(analysis: dict, question_metadata: dict, codebook: dict) -> pd.DataFrame:
    """One row per (question, subgroup) of the train and OOD-eval strata."""
    split_of = {s: "train" for s in config.TRAIN_STRATA} | {s: "eval_ood_subgroup" for s in config.EVAL_STRATA}
    records = []
    for q_id, q_analysis in analysis.items():
        meta = question_metadata.get(q_id)
        if q_id in config.EXCLUDED_QUESTIONS or not meta or meta.get("use_case") != "value":
            continue
        choices = valid_choices(codebook, q_id)
        if not choices:
            continue
        user_prompt_numerical = generate_user_prompt_numerical(q_id, question_metadata, codebook)
        user_prompt_open_ended = generate_user_prompt_open_ended(q_id, question_metadata, codebook)

        for stratum, stratum_data in q_analysis["strata_analysis"].items():
            if stratum not in split_of:
                continue
            for subgroup, info in stratum_data["subgroups"].items():
                # The modal answer is stored as a label; map it back to its first matching key.
                preferred = next((k for k, v in choices.items() if v == info["most_frequent_answer"]), None)
                if preferred is None:
                    continue
                records.append({
                    # Spaces become hyphens, e.g. "Q1_sex_x_religon_Female_Do-not-belong-to-a-denomination".
                    "sample_id": f"{q_id}_{stratum}_{subgroup}".replace(" ", "-"),
                    "q_id": q_id,
                    "split": split_of[stratum],
                    "stratum": stratum,
                    "subgroup": subgroup,
                    "question_text": meta.get("question"),
                    "category": meta.get("category"),
                    "choices": json.dumps(choices),
                    # Scale bounds come from the metadata, which differs from the codebook for a
                    # few questions (e.g. Q241: 1-10 vs 0-10); kept as in the original.
                    "choice_min": meta.get("answer_scale_min"),
                    "choice_max": meta.get("answer_scale_max"),
                    "preferred_answer": preferred,
                    "dispreferred_answers": json.dumps([k for k in choices if k != preferred]),
                    "system_prompt": generate_system_prompt(subgroup, config.COUNTRY_NAME, config.COUNTRY_CODE),
                    "user_prompt_numerical": user_prompt_numerical,
                    "user_prompt_open_ended": user_prompt_open_ended,
                    "sample_n": info["sample_count"],
                    "country": config.COUNTRY_NAME,
                    "country_code": config.COUNTRY_CODE,
                })
    return pd.DataFrame(records)


def to_conversations(samples: pd.DataFrame) -> list[dict]:
    """ShareGPT-style conversations (persona system prompt, question, 'Answer: <key>')."""
    return [
        {
            "messages": [
                {"role": "system", "content": row.system_prompt},
                {"role": "user", "content": row.user_prompt_numerical},
                {"role": "assistant", "content": f"Answer: {row.preferred_answer}"},
            ],
            "sample_id": row.sample_id,
        }
        for row in samples.itertuples()
    ]


def write_json(path, data) -> None:
    """Writes a JSON list as the original notebooks did (indent 2, no trailing newline)."""
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"  wrote {path.relative_to(config.ROOT)} ({len(data)} samples)")


def main() -> None:
    analysis = json.loads(config.WVS_ANALYSIS.read_text(encoding="utf-8"))
    codebook = json.loads(config.CODEBOOK.read_text(encoding="utf-8"))
    question_metadata = json.loads(config.QUESTION_METADATA.read_text(encoding="utf-8"))

    print("Building the preference dataset ...")
    samples = build_samples(analysis, question_metadata, codebook)
    samples.to_csv(config.PREFERENCE_DATASET, index=False)
    print(f"  wrote {config.PREFERENCE_DATASET.relative_to(config.ROOT)} ({len(samples)} samples)")

    write_json(config.SFT_TRAIN, to_conversations(samples[samples["split"] == "train"]))
    # Fixed-seed subset of the OOD split, used only as LLaMA-Factory's validation set.
    random.seed(42)
    eval_conversations = to_conversations(samples[samples["split"] == "eval_ood_subgroup"])
    write_json(config.SFT_EVAL, random.sample(eval_conversations, EVAL_SUBSET_SIZE))


if __name__ == "__main__":
    main()
