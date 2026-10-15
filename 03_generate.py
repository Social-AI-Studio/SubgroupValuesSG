"""Generates the raw answers of one evaluated model and the LLM-judge verdicts on them.

Inputs:  data/wvs_SG_preference_dataset.csv (persona system prompts, numerical and open-ended
         question prompts) and data/codebook.json (answer labels for the judge's ground truth).
Outputs: results/raw_inference/<model>/<model>_<split>_{numerical,open-ended}_raw.json
         results/raw_inference/gpt-4.1-2025-04-14/... (baseline answers, shared by all models)
         results/raw_inference/<judge>__judge/<judge>_judge_<model>_vs_<baseline>_<split>_pos{1,2}_raw.json
Paper:   the raw outputs scored by 04_score.py (also released on Hugging Face).
Runtime: open models and the judge need a GPU with vLLM 0.8.5.post1 (the judge needs a 48 GB GPU);
         GPT models and the GPT-4.1 baseline need OPENAI_API_KEY. Existing raw files are
         never regenerated, so an interrupted run can simply be restarted. On a single GPU,
         run '--stage generate' and '--stage judge' as separate processes, so that the
         evaluated model's memory is released before the judge is loaded.

Usage:   python 03_generate.py --model <safe name or paper label> [--sft] [--stage generate|judge|all] [--dry-run]
"""

import argparse
import asyncio
import gc
import json
from pathlib import Path

import pandas as pd

from svalues import config, prompts

# Greedy decoding with 2048 new tokens for every task, as in the paper's appendix. The
# original script meant to cap numerical answers at 64 tokens, but a task-naming slip meant
# every task used the open-ended limit. Runs made before 7 July 2025 (Llama-3.1-8B base and
# SFT; Llama-3.2-3B, Sailor2-8B and SeaLLMs-v3-7B base; all GPT models, including the
# baseline) used 1024 tokens and slightly different vLLM engine settings.
MAX_TOKENS = 2048
TASKS = {"numerical": "user_prompt_numerical", "open-ended": "user_prompt_open_ended"}
OPENAI_CONCURRENCY = 200
OPENAI_RETRIES = 5


# --- Raw files ---------------------------------------------------------------------------


def read_raw(path: Path) -> list[dict]:
    """Raw output file as a list of {custom_id, response}."""
    return json.loads(path.read_text(encoding="utf-8"))


def write_raw(path: Path, results: list[dict]) -> None:
    """Writes [{custom_id, response}, ...] in the format of the released raw files."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"  wrote {path.name}")


# --- Requests ---------------------------------------------------------------------------


def answer_requests(dataset: pd.DataFrame, split: str, task: str) -> list[dict]:
    """One chat request per sample: persona system prompt plus the task's question prompt."""
    rows = dataset[dataset["split"] == split]
    return [
        {"custom_id": row["sample_id"],
         "messages": [{"role": "system", "content": row["system_prompt"]},
                      {"role": "user", "content": row[TASKS[task]]}]}
        for _, row in rows.iterrows()
    ]


def judge_requests(dataset: pd.DataFrame, codebook: dict, split: str,
                   answers: dict, baseline_answers: dict) -> dict[str, list[dict]]:
    """Judge prompts in both answer orders: pos1 shows the model as Response A, pos2 as B.

    The judge sees the full open-ended prompt as the question, and the label of the
    subgroup's modal answer as the ground-truth stance. Samples without both answers are skipped.
    """
    requests = {"pos1": [], "pos2": []}
    for _, row in dataset[dataset["split"] == split].iterrows():
        sid = row["sample_id"]
        if sid not in answers or sid not in baseline_answers:
            continue
        stance = codebook[row["q_id"]]["choices"].get(str(row["preferred_answer"]), "Unknown")
        pairs = {"pos1": (answers[sid], baseline_answers[sid]), "pos2": (baseline_answers[sid], answers[sid])}
        for position, (response_a, response_b) in pairs.items():
            prompt = prompts.generate_judge_prompt(row["user_prompt_open_ended"], row["subgroup"], response_a,
                                                   response_b, stance, config.COUNTRY_NAME, config.COUNTRY_CODE)
            requests[position].append({"custom_id": f"{sid}_{position}",
                                       "messages": [{"role": "user", "content": prompt}]})
    return requests


# --- Inference back-ends (imported lazily so that --dry-run works without them) -------------


def run_vllm(model_id: str, adapter_id: str | None, jobs: list[tuple[list[dict], Path]]) -> None:
    """Runs each (requests, output path) job on one vLLM engine, loaded once."""
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest

    lora = None
    if adapter_id:
        from huggingface_hub import snapshot_download
        lora = LoRARequest("eval_adapter", 1, snapshot_download(repo_id=adapter_id))
    # Engine settings listed in the paper's appendix.
    llm = LLM(model=model_id, max_model_len=4096, enforce_eager=False, gpu_memory_utilization=0.9,
              enable_lora=adapter_id is not None, max_num_batched_tokens=8192)
    sampling = SamplingParams(temperature=0, max_tokens=MAX_TOKENS)
    try:
        for requests, path in jobs:
            # Each file is one llm.chat call, as in the original runs; the model's own chat
            # template is applied.
            outputs = llm.chat([r["messages"] for r in requests], sampling, use_tqdm=True, lora_request=lora)
            write_raw(path, [{"custom_id": r["custom_id"], "response": o.outputs[0].text}
                             for r, o in zip(requests, outputs)])
    finally:
        # Frees the engine before the judge is loaded in the same process ('--stage all').
        del llm
        gc.collect()


async def _openai_complete(client, semaphore, model_id: str, request: dict) -> dict:
    """One chat completion with retries.

    A request that still fails is stored as 'Error: ...' text, as in the original runs (two
    timed-out GPT-4.1 numerical answers, which 04_score.py counts as refusals).
    """
    error = None
    async with semaphore:
        for attempt in range(OPENAI_RETRIES):
            try:
                response = await client.chat.completions.create(
                    model=model_id, messages=request["messages"], temperature=0, max_tokens=MAX_TOKENS)
                return {"custom_id": request["custom_id"], "response": response.choices[0].message.content}
            except Exception as exc:  # any API failure is retried, then recorded
                error = exc
                await asyncio.sleep(2 ** attempt)
    return {"custom_id": request["custom_id"], "response": f"Error: {error}"}


async def _openai_batch(model_id: str, requests: list[dict]) -> list[dict]:
    from openai import AsyncOpenAI
    from tqdm.asyncio import tqdm_asyncio

    client = AsyncOpenAI()  # reads OPENAI_API_KEY from the environment
    semaphore = asyncio.Semaphore(OPENAI_CONCURRENCY)
    return await tqdm_asyncio.gather(*[_openai_complete(client, semaphore, model_id, r) for r in requests])


def run_openai(model_id: str, jobs: list[tuple[list[dict], Path]]) -> None:
    """Runs each (requests, output path) job through the OpenAI chat API, concurrently."""
    for requests, path in jobs:
        results = asyncio.run(_openai_batch(model_id, requests))
        failed = sum(r["response"].startswith("Error: ") for r in results)
        if failed:
            print(f"  warning: {failed} requests failed and are stored as 'Error: ...'; "
                  f"delete {path.name} and rerun to retry")
        write_raw(path, results)


# --- Stages ---------------------------------------------------------------------------------


def pending(jobs: list[tuple[list[dict], Path]]) -> list[tuple[list[dict], Path]]:
    """Jobs whose output file does not exist yet."""
    return [(requests, path) for requests, path in jobs if not path.exists()]


def generate(model: config.Model, dataset: pd.DataFrame) -> None:
    """The model's numerical and open-ended answers, then the GPT-4.1 baseline's open-ended answers."""
    jobs = [(answer_requests(dataset, split, task), config.raw_output_file(model.safe_name, split, task))
            for split in config.SPLITS for task in TASKS]
    todo = pending(jobs)
    if todo and model.provider == "openai":
        run_openai(model.model_id, todo)
    elif todo:
        run_vllm(model.model_id, model.adapter_id, todo)

    baseline = config.safe_name(config.BASELINE_MODEL)
    baseline_jobs = [(answer_requests(dataset, split, "open-ended"), config.raw_output_file(baseline, split, "open-ended"))
                     for split in config.SPLITS]
    todo = pending(baseline_jobs)
    if todo:
        run_openai(config.BASELINE_MODEL, todo)


def answers_by_id(path: Path) -> dict:
    """Raw output file as {custom_id: response}."""
    return {item["custom_id"]: item["response"] for item in read_raw(path)}


def build_judge_jobs(model: config.Model, dataset: pd.DataFrame, codebook: dict) -> list[tuple[list[dict], Path]]:
    """Judge requests for both splits and orders, built from the cached open-ended answers."""
    baseline = config.safe_name(config.BASELINE_MODEL)
    jobs = []
    for split in config.SPLITS:
        requests = judge_requests(dataset, codebook, split,
                                  answers_by_id(config.raw_output_file(model.safe_name, split, "open-ended")),
                                  answers_by_id(config.raw_output_file(baseline, split, "open-ended")))
        jobs += [(requests[position], config.judge_output_file(model.safe_name, split, position)) for position in requests]
    return jobs


def judge(model: config.Model, dataset: pd.DataFrame, codebook: dict) -> None:
    """Position-swapped LLM-judge verdicts on the model's open-ended answers."""
    jobs = pending(build_judge_jobs(model, dataset, codebook))
    if jobs:
        run_vllm(config.JUDGE_MODEL, None, jobs)


def dry_run(model: config.Model, dataset: pd.DataFrame, codebook: dict) -> bool:
    """Builds every request without running anything and checks its ids against cached files."""
    baseline = config.safe_name(config.BASELINE_MODEL)
    jobs = [(answer_requests(dataset, split, task), config.raw_output_file(model.safe_name, split, task))
            for split in config.SPLITS for task in TASKS]
    jobs += [(answer_requests(dataset, split, "open-ended"), config.raw_output_file(baseline, split, "open-ended"))
             for split in config.SPLITS]
    answers_cached = all(config.raw_output_file(name, split, "open-ended").exists()
                         for name in [model.safe_name, baseline] for split in config.SPLITS)
    if answers_cached:
        jobs += build_judge_jobs(model, dataset, codebook)
    else:
        print("  judge requests: skipped (open-ended answers not cached yet)")

    all_match = True
    for requests, path in jobs:
        if path.exists():
            match = {r["custom_id"] for r in requests} == {item["custom_id"] for item in read_raw(path)}
            status = "ids match cached file" if match else "ID MISMATCH with cached file"
            all_match &= match
        else:
            status = "not cached"
        print(f"  {len(requests):6d} requests  {status}: {path.relative_to(config.RAW_INFERENCE_DIR)}")
    return all_match


def find_model(name: str, sft: bool) -> config.Model:
    """Model by safe name, or by paper label (base model unless --sft)."""
    for model in config.ALL_MODELS:
        if model.safe_name.lower() == name.lower():
            return model
    for model in config.ALL_MODELS:
        if model.label.lower() == name.lower() and model.is_sft == sft:
            return model
    raise SystemExit(f"Unknown model '{name}'; use a safe name or paper label from config.ALL_MODELS.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", required=True, help="safe name or paper label, e.g. 'Sailor2-8B'")
    parser.add_argument("--sft", action="store_true", help="with a paper label: the SFT (LoRA) model")
    parser.add_argument("--stage", choices=["generate", "judge", "all"], default="all")
    parser.add_argument("--dry-run", action="store_true", help="build and check requests without running them")
    args = parser.parse_args()

    try:  # optional: read OPENAI_API_KEY from a .env file
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass

    model = find_model(args.model, args.sft)
    dataset = pd.read_csv(config.PREFERENCE_DATASET)
    codebook = json.loads(config.CODEBOOK.read_text(encoding="utf-8"))
    print(f"Model: {model.safe_name}")

    if args.dry_run:
        if not dry_run(model, dataset, codebook):
            raise SystemExit("Request ids differ from the cached raw files.")
        return
    if args.stage in ("generate", "all"):
        generate(model, dataset)
    if args.stage in ("judge", "all"):
        judge(model, dataset, codebook)


if __name__ == "__main__":
    main()
