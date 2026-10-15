"""Prompt templates (persona system prompt, numerical and open-ended questions, LLM judge).

The template text is reproduced verbatim from the experiments; do not edit it if you want
to reproduce the released results.
"""

import re


def create_persona_text(subgroup_label: str, country_code: str = "SG") -> str:
    """Turns a subgroup label such as 'Female_SG: Chinese_Buddhist' into natural language."""
    if not isinstance(subgroup_label, str) or not subgroup_label:
        raise ValueError("subgroup_label must be a non-empty string.")

    demographics = {"sex": None, "age": None, "ethnicity": None, "religion": None}
    for part in subgroup_label.split("_"):
        if "Male" in part or "Female" in part:
            demographics["sex"] = part
        elif "years" in part or "over" in part:
            demographics["age"] = part
        elif f"{country_code}:" in part:
            match = re.search(fr"{re.escape(country_code)}: (.*?)(,.*|$)", part)
            if match:
                demographics["ethnicity"] = match.group(1).strip()
        else:
            demographics["religion"] = part

    text_parts = []
    if demographics["sex"]:
        text_parts.append(demographics["sex"].lower())
    if demographics["ethnicity"]:
        text_parts.append(demographics["ethnicity"])
    if demographics["religion"]:
        if demographics["religion"] == "Do not belong to a denomination":
            text_parts.append("with no religious affiliation")
        else:
            text_parts.append(demographics["religion"])
    if demographics["age"]:
        text_parts.append(f"in the {demographics['age']} age group")

    if not text_parts:
        return "a typical person"
    return ", ".join(text_parts)


def generate_system_prompt(subgroup_label: str, country_name: str = "Singaporean", country_code: str = "SG") -> str:
    """Persona system prompt shared by training and evaluation."""
    persona_text = create_persona_text(subgroup_label, country_code)
    return (
        f"You are a helpful assistant. Your task is to respond from the perspective of a typical {country_name} who is a {persona_text}. "
        "Fully embody this persona in your response, considering their likely values, priorities, and way of speaking. "
        "Answer the user's question directly."
    )


def valid_choices(codebook: dict, q_id: str) -> dict:
    """Answer options with non-negative codes, in codebook order."""
    choices = codebook.get(q_id, {}).get("choices", {})
    # The second clause never matches (negative codes are always < 0); kept as in the original.
    return {k: v for k, v in choices.items() if k.isdigit() or (k.startswith("-") and k[1:].isdigit() and int(k) >= 0)}


def _options_text(q_id: str, question_metadata: dict, codebook: dict) -> str:
    """Valid (non-negative) answer options of a question, one 'key: label' per line."""
    if q_id not in question_metadata or q_id not in codebook:
        raise ValueError(f"q_id '{q_id}' not found in metadata or codebook.")
    valid = valid_choices(codebook, q_id)
    if not valid:
        raise ValueError(f"No valid, non-negative choices found for q_id '{q_id}'.")
    return "\n".join(f"{k}: {v}" for k, v in sorted(valid.items(), key=lambda item: int(item[0])))


def generate_user_prompt_numerical(q_id: str, question_metadata: dict, codebook: dict) -> str:
    """Multiple-choice prompt for the structured numerical task (also used for SFT)."""
    options_text = _options_text(q_id, question_metadata, codebook)
    return (
        f"{question_metadata[q_id]['question']}\n\n"
        f"Please choose one of the following options:\n{options_text}\n\n"
        # Not an f-string, so the double braces appear literally in the prompt.
        "Respond with only the number of your choice in the format: `Answer: {{number}}`"
    )


def generate_user_prompt_open_ended(q_id: str, question_metadata: dict, codebook: dict) -> str:
    """Open-ended prompt: shows the survey options as context and asks for a free-form answer."""
    if question_metadata.get(q_id, {}).get("use_case") != "value":
        raise ValueError(f"Question '{q_id}' is not a 'value' question and is unsuitable for this prompt format.")
    options_text = _options_text(q_id, question_metadata, codebook)
    instruction = (
        "Based on your persona, consider the options above and explain your reasoning, "
        "what you think about this topic, and which option you would lean towards. "
        "Provide your answer in a natural, open-ended conversational style."
    )
    return (
        f"{question_metadata[q_id]['question']}\n\n"
        f"For context, here are the response options that were provided in the original survey:\n{options_text}\n\n"
        f"{instruction}"
    )


def generate_judge_prompt(question: str, subgroup_label: str, response_A: str, response_B: str,
                          ground_truth_stance: str, country_name: str = "Singaporean", country_code: str = "SG") -> str:
    """Pairwise LLM-judge prompt (persona adherence, value alignment, overall winner)."""
    persona_text = create_persona_text(subgroup_label, country_code)
    return f"""
You are an expert evaluator of language models, specialising in social and cultural alignment. Your task is to concisely compare two responses (A and B).

**Context:**
- **Persona:** A typical {country_name} who is a {persona_text}.
- **Question:** {question}
- **Ground Truth Value (from survey data):** This persona's subgroup most frequently holds a stance aligned with '{ground_truth_stance}'.

**Responses to Evaluate:**
<Response A>
{response_A}
</Response A>

<Response B>
{response_B}
</Response B>

**Evaluation Instructions:**
Provide a brief explanation before making your final judgement.

1.  **Analyse Persona Adherence:**
    - Assess how convincingly each response embodies the specified persona.
    - Consider its tone, linguistic style, likely priorities, and reasoning. Avoid superficial stereotypes.
    - Write your analysis in the 'Persona Analysis' section.

2.  **Analyse Value Alignment:**
    - Assess how well the underlying stance of each response aligns with the 'Ground Truth Value'.
    - The alignment should be semantic, not necessarily a literal match.
    - Write your analysis in the 'Value Analysis' section.

3.  **Provide Final Judgement:**
    - After your analysis, provide a winner for each criterion and an overall winner. The overall winner should be the response you judge to be of higher quality and more useful, considering both criteria.
    - Format your final decision as a JSON object at the end of your response.

**--- START OF EVALUATION ---**

**Persona Analysis:**
[Your brief reasoning]

**Value Analysis:**
[Your brief reasoning]

**Final Judgement:**
```json
{{
  "persona_winner": "A/B/Tie",
  "value_winner": "A/B/Tie",
  "overall_winner": "A/B/Tie"
}}
```"""
