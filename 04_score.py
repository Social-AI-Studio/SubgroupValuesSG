"""Scores every evaluated model from its cached raw outputs.

Inputs:  results/raw_inference/<model>/  numerical and open-ended responses (03_generate.py,
         or downloaded from Hugging Face), the GPT-4.1 baseline responses and the
         LLM-judge verdicts in results/raw_inference/<judge>__judge/, and
         data/wvs_SG_preference_dataset.csv.
Outputs: results/final_scores/ per model and split:
         <model>_<split>_canonical_scores.csv            per-sample answers and scores
         judge_<model>_vs_<baseline>_<split>_responses.csv  raw judge verdicts (both orders)
         <model>_evaluation_summary.json                 split, subgroup and bias metrics
Paper:   every results table and figure (accuracy, NMAE, win rates, refusals, fairness).
Runtime: CPU only, about 5 s per model (under 2 min for all 18).

Usage:   python 04_score.py [--models <safe name or paper label> ...]
"""

import argparse
import json
from pathlib import Path

import pandas as pd

from svalues import config, metrics

PROMPT_COLUMNS = ["system_prompt", "user_prompt_numerical", "user_prompt_open_ended"]


def load_raw(path: Path, column: str) -> pd.DataFrame:
    """Raw output file -> frame with columns [sample_id, <column>]."""
    if not path.exists():
        raise FileNotFoundError(f"{path} is missing: download the model outputs (README, step 3) or run 03_generate.py.")
    records = json.loads(path.read_text(encoding="utf-8"))
    return pd.DataFrame(records).rename(columns={"custom_id": "sample_id", "response": column})


def load_judge(safe: str, split: str) -> pd.DataFrame:
    """Judge verdicts for both answer orders, one row per sample."""
    orders = []
    for position in ["pos1", "pos2"]:
        df = load_raw(config.judge_output_file(safe, split, position), f"judge_response_{position}")
        df["sample_id"] = df["sample_id"].str.removesuffix(f"_{position}")
        orders.append(df)
    return orders[0].merge(orders[1], on="sample_id")


def write_csv(df: pd.DataFrame, path: Path) -> None:
    """CSV in the released format.

    utf-8-sig (with BOM) and '\\n' line endings, as in the original files. Carriage returns
    inside model responses are written as they are, so do not normalise line endings.
    """
    df.to_csv(path, index=False, encoding="utf-8-sig", lineterminator="\n")


def score_split(model: config.Model, split: str, dataset: pd.DataFrame) -> dict:
    """Scores one split of one model, writes its two CSVs and returns its summary block."""
    safe, baseline = model.safe_name, config.safe_name(config.BASELINE_MODEL)
    df = dataset[dataset["split"] == split].copy()

    numerical = load_raw(config.raw_output_file(safe, split, "numerical"), "numerical_response")
    # Raw parse before the range check: out-of-range answers keep their value in this
    # column, whereas the metrics count them as refusals.
    numerical["parsed_numerical_answer"] = numerical["numerical_response"].apply(metrics.parse_numerical_response)
    df = df.merge(numerical, on="sample_id", how="left")
    summary = {"numerical": metrics.numerical_metrics(df)}

    judge = load_judge(safe, split)
    write_csv(judge, config.judge_scores_file(safe, split))
    summary["open-ended"], scores = metrics.win_rates(judge)
    df = df.merge(scores[["sample_id", *metrics.SCORE_COLUMNS]], on="sample_id", how="left")

    df = df.merge(load_raw(config.raw_output_file(safe, split, "open-ended"), "open_ended_response"), on="sample_id", how="left")
    baseline_responses = load_raw(config.raw_output_file(baseline, split, "open-ended"), "baseline_open_ended_response")
    df = df.merge(baseline_responses, on="sample_id", how="left")

    summary["subgroups"], summary["bias_metrics_by_stratum"], summary["overall_model_bias"] = metrics.subgroup_metrics(df)
    write_csv(df.drop(columns=PROMPT_COLUMNS), config.scores_file(safe, split))
    return summary


def evaluation_parameters(model: config.Model) -> dict:
    """Settings that produced the scores, stored alongside them."""
    return {
        "model_name": model.model_id,
        "provider": model.provider,
        "lora_repo_id": model.adapter_id,
        "eval_splits": config.SPLITS,
        "country_name": config.COUNTRY_NAME,
        "country_code": config.COUNTRY_CODE,
        "baseline_model": config.BASELINE_MODEL,
        "judge_model": config.JUDGE_MODEL,
        "judge_provider": "vllm",
    }


def score_model(model: config.Model, dataset: pd.DataFrame) -> None:
    """Scores both splits of a model and writes its evaluation summary."""
    results = {split: score_split(model, split, dataset) for split in config.SPLITS}
    summary = {"evaluation_parameters": evaluation_parameters(model), "results_summary": results}
    # No trailing newline, as in the original files.
    config.summary_file(model.safe_name).write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")


def select_models(names: list[str] | None) -> list[config.Model]:
    """Models matching the given safe names or paper labels (a label selects base and SFT)."""
    if not names:
        return config.ALL_MODELS
    wanted = {name.lower() for name in names}
    selected = [m for m in config.ALL_MODELS if m.safe_name.lower() in wanted or m.label.lower() in wanted]
    if not selected:
        raise SystemExit(f"No model matches {names}; see config.ALL_MODELS.")
    return selected


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--models", nargs="+", help="safe names or paper labels (default: all 18 models)")
    args = parser.parse_args()

    dataset = pd.read_csv(config.PREFERENCE_DATASET)
    config.FINAL_SCORES_DIR.mkdir(parents=True, exist_ok=True)
    for model in select_models(args.models):
        score_model(model, dataset)
        print(f"  scored {model.safe_name}")
    print(f"Wrote scores to {config.FINAL_SCORES_DIR}")


if __name__ == "__main__":
    main()
