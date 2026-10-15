# SubgroupValuesSG

Code and models for **"Can Persona-Prompted LLMs Emulate Subgroup Values? An Empirical Analysis of Generalisability and Fairness in Cultural Alignment"** (ACL 2026).

[Paper (ACL Anthology)](https://aclanthology.org/2026.acl-long.1127/) · [arXiv](https://arxiv.org/abs/2604.12851) · [Model outputs (Hugging Face)](https://huggingface.co/datasets/Incomple/SubgroupValuesSG-outputs) · [Models](#models)

Using the Singapore subset of the World Values Survey (WVS) Wave 7, we build a dataset of 20,877 (question, subgroup) samples covering 98 demographic subgroups, fine-tune seven open-source LLMs to predict each subgroup's modal answer, and test how well this generalises to unseen subgroups and open-ended answers, and how fairly the gains are distributed.

## What is released

The WVS conditions of use do not allow its data or derived datasets to be redistributed, so we release the code that recreates our dataset exactly from the official survey file.

| Artefact | Where |
|---|---|
| Code, training configs, generated paper tables, checksums | this repository |
| Question codebook and metadata (from WorldValuesBench), anonymised human-evaluation labels | `data/` |
| Answers of the 18 evaluated models and the LLM judge's verdicts | [Hugging Face](https://huggingface.co/datasets/Incomple/SubgroupValuesSG-outputs) (610 MB) |
| Seven LoRA adapters | [Models](#models) |

## Models

| Paper name | Base model | SFT adapter (LoRA, rank 8) |
|---|---|---|
| Llama-3.1-8B | [meta-llama/Llama-3.1-8B-Instruct](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct) | [Incomple/Llama-3.1-8B-Instruct_sft_sg_values](https://huggingface.co/Incomple/Llama-3.1-8B-Instruct_sft_sg_values) |
| Llama-3.2-3B | [meta-llama/Llama-3.2-3B-Instruct](https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct) | [Incomple/Llama-3.2-3B-Instruct_sft_sg_values](https://huggingface.co/Incomple/Llama-3.2-3B-Instruct_sft_sg_values) |
| SEA-LION-v3-8B | [aisingapore/Llama-SEA-LION-v3-8B-IT](https://huggingface.co/aisingapore/Llama-SEA-LION-v3-8B-IT) | [Incomple/Llama-SEA-LION-v3-8B-IT_sft_sg_values](https://huggingface.co/Incomple/Llama-SEA-LION-v3-8B-IT_sft_sg_values) |
| Phi-4-mini | [microsoft/Phi-4-mini-instruct](https://huggingface.co/microsoft/Phi-4-mini-instruct) | [Incomple/Phi-4-mini-instruct_sft_sg_values](https://huggingface.co/Incomple/Phi-4-mini-instruct_sft_sg_values) |
| Qwen2.5-7B | [Qwen/Qwen2.5-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct) | [Incomple/Qwen2.5-7B_sft_sg_values](https://huggingface.co/Incomple/Qwen2.5-7B_sft_sg_values) |
| Sailor2-8B | [sail/Sailor2-8B-Chat](https://huggingface.co/sail/Sailor2-8B-Chat) | [Incomple/Sailor2-8B-Chat_sft_sg_values](https://huggingface.co/Incomple/Sailor2-8B-Chat_sft_sg_values) |
| SeaLLMs-v3-7B | [SeaLLMs/SeaLLMs-v3-7B-Chat](https://huggingface.co/SeaLLMs/SeaLLMs-v3-7B-Chat) | [Incomple/SeaLLMs-v3-7B-Chat_sft_sg_values](https://huggingface.co/Incomple/SeaLLMs-v3-7B-Chat_sft_sg_values) |

Closed-source baselines: `gpt-4.1-2025-04-14`, `gpt-4o-2024-08-06`, `gpt-4.1-mini-2025-04-14`, `gpt-4o-mini-2024-07-18`. Open-ended answers are judged against GPT-4.1 by [RedHatAI/Mistral-Small-3.1-24B-Instruct-2503-quantized.w8a8](https://huggingface.co/RedHatAI/Mistral-Small-3.1-24B-Instruct-2503-quantized.w8a8).

## Reproducing the paper

1. Clone the repository and install the requirements:
   ```bash
   git clone https://github.com/Social-AI-Studio/SubgroupValuesSG && cd SubgroupValuesSG
   python3.11 -m venv .venv && source .venv/bin/activate
   pip install -r requirements.txt
   ```
2. Download the WVS Wave 7 **Singapore** Excel file from the [WVS website](https://www.worldvaluessurvey.org/WVSDocumentationWV7.jsp) (Data and Documentation → Data Download → Wave 7; registration required) and save it as `data/raw/F00013217-WVS_Wave_7_Singapore_Excel_v5.1.xlsx` (SHA-256 `65456cd2542a447890926f6ba6edbfe0f5f6f139d2dfb933eb43f4f339543e1d`; other versions may not reproduce our dataset exactly).
3. Download the model outputs (keep `--include`, otherwise the dataset card overwrites this README):
   ```bash
   hf download Incomple/SubgroupValuesSG-outputs --repo-type dataset --local-dir . --include "results/*"
   ```
4. Run `bash reproduce.sh`. It recreates the dataset, scores all models, writes the tables to `outputs/tables/` and checks all 149 files against `checksums.sha256`.

To recreate only the dataset, skip step 3 and run `python 01_process_wvs.py && python 02_build_dataset.py`.

## Pipeline

| Script | What it does | Needs | Time |
|---|---|---|---|
| `01_process_wvs.py` | Subgroup answer distributions, modal answers, Modal Diversity Scores | WVS file | 1 min |
| `02_build_dataset.py` | Preference dataset and SFT train/eval files | 01 | seconds |
| `03_generate.py` | Model answers, GPT-4.1 reference answers, judge verdicts | GPU, OpenAI API | hours |
| `04_score.py` | Accuracy, NMAE, win rates and subgroup disparity per model | 02, model outputs | 1 min |
| `05_data_tables.py` | Value-landscape and dataset tables | 01, 02 | seconds |
| `06_result_tables.py` | Results, fairness, subgroup and refusal tables | 04 | seconds |
| `07_robustness.py` | Bootstrap CIs, MDS sensitivity, label stability | 01, 04 | seconds |
| `08_human_eval.py` | Agreement between human annotators and the LLM judge | judge verdicts | seconds |
| `verify.py` | Checks outputs against `checksums.sha256` | – | seconds |

Shared code is in `svalues/`: paths, strata and models (`config.py`), prompts (`prompts.py`), metrics (`metrics.py`) and table formatting (`tables.py`).

### Paper tables

| Paper | Output in `outputs/tables/` | Script |
|---|---|---|
| Table 1 – high- and no-conflict questions | `conflict_examples` | 05 |
| Table 2 – value conflict by category | `category_divisiveness` | 05 |
| Table 3 – dataset composition | `dataset_composition` | 05 |
| Table 4 – results on the OOD split | `results_ood_simplified` | 06 |
| Table 5 – subgroup disparity (OOD) | `disparity_ood` | 06 |
| Table 6 – subgroup impact of SFT (OOD) | `subgroup_impact` | 06 |
| Table 7 – human evaluation | `human_eval_combined` | 08 |
| Table 8 – subgroup counts | `subgroup_counts_full` | 05 |
| Tables 9–10 – results and disparity (train split) | `results_train_split`, `disparity_train_split` | 06 |
| Tables 11–12 – refusals | `refused_questions`, `refusal_rates` | 06 |
| Tables 13–20 – subgroup impact per stratum | `sft_impact_*` | 06 |
| Table 21 – MDS sensitivity | `mds_ordinal_comparison` | 07 |
| Tables 22–23 – bootstrap CIs | `bootstrap_ci`, `bootstrap_ci_oe` | 07 |
| Table 24 – hyperparameters | `training/configs/` | – |
| Figure 2 – prompt templates | `svalues/prompts.py` | – |

## Re-running the models (GPU)

```bash
pip install -r requirements-gpu.txt
python 03_generate.py --model Sailor2-8B --sft --stage generate   # answers (GPU; baseline via OpenAI)
python 03_generate.py --model Sailor2-8B --sft --stage judge      # judge verdicts (48 GB GPU)
python 04_score.py --models Sailor2-8B
```

- Models are named by paper label (`--sft` for the adapter) or file name. OpenAI models need `OPENAI_API_KEY`. `--dry-run` checks the requests against the released files without running a model.
- Existing output files are skipped. To regenerate a model, move away its folder and its judge files (`results/raw_inference/<judge>__judge/*_judge_<model>_vs_*`).

## Training

The adapters were trained with [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory) 0.9.3 using `training/configs/` (one epoch, learning rate 1e-6, LoRA rank 8 on all linear layers, batch size 2 per GPU, 4 gradient-accumulation steps). After running 01 and 02:

```bash
CUDA_VISIBLE_DEVICES=0 llamafactory-cli train training/configs/llama-3.1-8b.yaml
```


## Data sources

- **World Values Survey.** Please cite: Haerpfer, C., Inglehart, R., Moreno, A., Welzel, C., Kizilova, K., Diez-Medrano J., M. Lagos, P. Norris, E. Ponarin & B. Puranen (eds.). 2022. World Values Survey: Round Seven - Country-Pooled Datafile Version 5.0. Madrid, Spain & Vienna, Austria: JD Systems Institute & WVSA Secretariat. doi:10.14281/18241.24
- **WorldValuesBench.** `data/codebook.json` and `data/question_metadata.json` are adapted from [WorldValuesBench](https://github.com/Demon702/WorldValuesBench), with the answer labels of Q7–Q26 harmonised to "Yes"/"No". These 20 questions are excluded from the dataset, leaving 214.

## Citation

```bibtex
@inproceedings{tanCanPersonaPromptedLLMs2026,
  title = {Can {{Persona-Prompted LLMs Emulate Subgroup Values}}? {{An Empirical Analysis}} of {{Generalisability}} and {{Fairness}} in {{Cultural Alignment}}},
  shorttitle = {Can {{Persona-Prompted LLMs Emulate Subgroup Values}}?},
  booktitle = {Proceedings of the 64th {{Annual Meeting}} of the {{Association}} for {{Computational Linguistics}} ({{Volume}} 1: {{Long Papers}})},
  author = {Tan, Bryan Chen Zhengyu and Liu, Zhengyuan and Yi, Xiaoyuan and Yao, Jing and Xie, Xing and Chen, Nancy F. and Lee, Roy Ka-Wei},
  editor = {Liakata, Maria and Moreira, Viviane P. and Zhang, Jiajun and Jurgens, David},
  year = 2026,
  month = jul,
  pages = {24571--24590},
  publisher = {Association for Computational Linguistics},
  address = {San Diego, California, United States},
  doi = {10.18653/v1/2026.acl-long.1127},
  urldate = {2026-09-14},
  abstract = {Despite their global prevalence, many Large Language Models (LLMs) are aligned to a monolithic, often Western-centric set of values. This paper investigates the more challenging task of fine-grained value alignment: examining whether LLMs can emulate the distinct cultural values of demographic subgroups. Using Singapore as a case study and the World Values Survey (WVS), we examine the value landscape and show that even state-of-the-art models like GPT-4.1 achieve only 57.4\% accuracy in predicting subgroup modal preferences. We construct a dataset of over 20,000 samples to train and evaluate a range of models. We demonstrate that simple fine-tuning on structured numerical preferences yields substantial gains, improving accuracy on unseen, out-of-distribution subgroups by an average of 17.4\%. These gains partially transfer to open-ended generation. However, we find significant pre-existing performance biases, where models better emulate young, male, Chinese, and Christian personas. Furthermore, while fine-tuning improves average performance, it widens the disparity between subgroups when measured by distance-aware metrics. Our work offers insights into the limits and fairness implications of subgroup-level cultural alignment.},
  isbn = {979-8-89176-390-6},
}
```

## Licence

Code: MIT (`LICENSE`). Model outputs, human-evaluation labels and generated tables: CC BY-NC 4.0 (`DATA_LICENSE.md`).
