"""Paths, experimental constants and the registry of evaluated models."""

from dataclasses import dataclass
from pathlib import Path

# --- Paths -----------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
HUMAN_EVAL_DIR = DATA_DIR / "human_eval"

RESULTS_DIR = ROOT / "results"
RAW_INFERENCE_DIR = RESULTS_DIR / "raw_inference"
FINAL_SCORES_DIR = RESULTS_DIR / "final_scores"
ROBUSTNESS_DIR = RESULTS_DIR / "robustness"

TABLES_DIR = ROOT / "outputs" / "tables"

# Raw survey file (not redistributed; see README for the download link). The checksum
# identifies the exact version (v5.1) used in the paper.
WVS_EXCEL = RAW_DIR / "F00013217-WVS_Wave_7_Singapore_Excel_v5.1.xlsx"
WVS_EXCEL_SHA256 = "65456cd2542a447890926f6ba6edbfe0f5f6f139d2dfb933eb43f4f339543e1d"

# Question text, answer options and categories, adopted from WorldValuesBench.
CODEBOOK = DATA_DIR / "codebook.json"
QUESTION_METADATA = DATA_DIR / "question_metadata.json"

# Derived data written by 01_process_wvs.py and 02_build_dataset.py.
WVS_ANALYSIS = DATA_DIR / "singapore_wvs_analysis_rich.json"
QUESTION_RANKING = DATA_DIR / "all_questions_ranked_by_modal_diversity.json"
SUBGROUP_COUNTS = DATA_DIR / "subgroup_counts.json"
PREFERENCE_DATASET = DATA_DIR / "wvs_SG_preference_dataset.csv"
SFT_TRAIN = DATA_DIR / "sft_SG_train_dataset.json"
SFT_EVAL = DATA_DIR / "sft_SG_eval_dataset.json"

# --- Experimental constants -------------------------------------------------
COUNTRY_NAME = "Singaporean"
COUNTRY_CODE = "SG"

# A subgroup's modal answer is only used if at least this many respondents answered.
MIN_SUBGROUP_SIZE = 30

# Demographic strata -> WVS variable codes (Q260 sex, X003R age group, Q290 ethnicity,
# Q289 religion). "sex_x_religon" is a typo from the original run; it is kept because it
# is part of every sample_id and of all released files.
STRATA = {
    "sex": ["Q260"],
    "age_group": ["X003R"],
    "ethnicity": ["Q290"],
    "religion": ["Q289"],
    "sex_x_age": ["Q260", "X003R"],
    "sex_x_ethnicity": ["Q260", "Q290"],
    "sex_x_religon": ["Q260", "Q289"],
    "ethnicity_x_religion": ["Q290", "Q289"],
    "age_x_religion": ["X003R", "Q289"],
    "age_x_ethnicity": ["X003R", "Q290"],
}
TRAIN_STRATA = ["sex", "age_group", "ethnicity", "religion", "sex_x_age", "sex_x_ethnicity", "sex_x_religon"]
EVAL_STRATA = ["ethnicity_x_religion", "age_x_religion", "age_x_ethnicity"]

SPLITS = ["train", "eval_ood_subgroup"]

# Q7-Q26 use a non-standard "mentioned / not mentioned" format and are excluded from the
# dataset (Section 3.2 of the paper). The robustness analyses in the appendix were run on
# all questions, including these, and are reproduced as published.
EXCLUDED_QUESTIONS = [f"Q{i}" for i in range(7, 27)]

# --- Models ------------------------------------------------------------------
BASELINE_MODEL = "gpt-4.1-2025-04-14"  # reference answers for the open-ended win rate
JUDGE_MODEL = "RedHatAI/Mistral-Small-3.1-24B-Instruct-2503-quantized.w8a8"


def safe_name(model_id: str, adapter_id: str | None = None) -> str:
    """File-system-safe name used for every results file of a model."""
    name = model_id.replace("/", "_")
    if adapter_id:
        name += "__lora__" + adapter_id.replace("/", "_")
    return name


@dataclass(frozen=True)
class Model:
    label: str                   # name used in the paper
    model_id: str                # Hugging Face base model or OpenAI model id
    adapter_id: str | None = None  # LoRA adapter on Hugging Face (SFT models only)
    provider: str = "vllm"       # "vllm" (local GPU) or "openai"

    @property
    def safe_name(self) -> str:
        return safe_name(self.model_id, self.adapter_id)

    @property
    def is_sft(self) -> bool:
        return self.adapter_id is not None


# Open-source models in paper order: (label, base model, SFT adapter).
OPEN_SOURCE = [
    ("Llama-3.1-8B", "meta-llama/Llama-3.1-8B-Instruct", "Incomple/Llama-3.1-8B-Instruct_sft_sg_values"),
    ("Llama-3.2-3B", "meta-llama/Llama-3.2-3B-Instruct", "Incomple/Llama-3.2-3B-Instruct_sft_sg_values"),
    ("SEA-LION-v3-8B", "aisingapore/Llama-SEA-LION-v3-8B-IT", "Incomple/Llama-SEA-LION-v3-8B-IT_sft_sg_values"),
    ("Phi-4-mini", "microsoft/Phi-4-mini-instruct", "Incomple/Phi-4-mini-instruct_sft_sg_values"),
    ("Qwen2.5-7B", "Qwen/Qwen2.5-7B-Instruct", "Incomple/Qwen2.5-7B_sft_sg_values"),
    ("Sailor2-8B", "sail/Sailor2-8B-Chat", "Incomple/Sailor2-8B-Chat_sft_sg_values"),
    ("SeaLLMs-v3-7B", "SeaLLMs/SeaLLMs-v3-7B-Chat", "Incomple/SeaLLMs-v3-7B-Chat_sft_sg_values"),
]

# Closed-source models in paper order: (label, OpenAI model id).
CLOSED_SOURCE = [
    ("GPT-4.1", "gpt-4.1-2025-04-14"),
    ("GPT-4o", "gpt-4o-2024-08-06"),
    ("GPT-4.1-mini", "gpt-4.1-mini-2025-04-14"),
    ("GPT-4o-mini", "gpt-4o-mini-2024-07-18"),
]

BASE_MODELS = [Model(label, model_id) for label, model_id, _ in OPEN_SOURCE]
SFT_MODELS = [Model(label, model_id, adapter_id) for label, model_id, adapter_id in OPEN_SOURCE]
CLOSED_MODELS = [Model(label, model_id, provider="openai") for label, model_id in CLOSED_SOURCE]

# All 18 evaluated models, and (base, SFT) pairs for before/after comparisons.
ALL_MODELS = BASE_MODELS + SFT_MODELS + CLOSED_MODELS
BASE_SFT_PAIRS = list(zip(BASE_MODELS, SFT_MODELS))

# --- Results files (names as in the original runs) ------------------------------


def raw_output_file(safe: str, split: str, task: str) -> Path:
    """Raw answers of a model for one split and task ('numerical' or 'open-ended')."""
    return RAW_INFERENCE_DIR / safe / f"{safe}_{split}_{task}_raw.json"


def judge_output_file(safe: str, split: str, position: str) -> Path:
    """Raw judge verdicts on a model's open-ended answers vs the baseline, in one order ('pos1'/'pos2')."""
    judge, baseline = safe_name(JUDGE_MODEL), safe_name(BASELINE_MODEL)
    return RAW_INFERENCE_DIR / f"{judge}__judge" / f"{judge}_judge_{safe}_vs_{baseline}_{split}_{position}_raw.json"


def scores_file(safe: str, split: str) -> Path:
    """Per-sample answers and scores of a model (written by 04_score.py)."""
    return FINAL_SCORES_DIR / f"{safe}_{split}_canonical_scores.csv"


def judge_scores_file(safe: str, split: str) -> Path:
    """Judge verdicts in both orders, one row per sample (written by 04_score.py)."""
    return FINAL_SCORES_DIR / f"judge_{safe}_vs_{safe_name(BASELINE_MODEL)}_{split}_responses.csv"


def summary_file(safe: str) -> Path:
    """Split-, subgroup- and bias-level metrics of a model (written by 04_score.py)."""
    return FINAL_SCORES_DIR / f"{safe}_evaluation_summary.json"
