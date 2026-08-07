"""Shared prompts so SFT (`train_normalized_qa_sft`) and vLLM inference stay aligned."""

from __future__ import annotations

# Matches train_peerqa_generator_sft.py (generator SFT target).
GENERATOR_SYSTEM = (
    "You write exam-style question and answer pairs for scientific text. "
    "Answers must not introduce facts outside the given excerpt."
)


def generator_user_block(excerpt: str) -> str:
    return (
        "You are given an excerpt from a document (passages may be fragmented). "
        "Produce ONE question and ONE short answer that are fully grounded in the excerpt only.\n\n"
        "Excerpt:\n"
        + excerpt.strip()
        + '\n\nRespond with a single JSON object with keys "question" and "answer" only. '
        "No markdown fences."
    )


DROP_GENERATOR_PROMPT_VERSION = "drop_reasoning/v1"

DROP_GENERATOR_SYSTEM = (
    "You write DROP-style reading comprehension questions. "
    "Answers must be derivable only from the given passage using counting, comparison, "
    "or simple arithmetic over facts explicitly stated in the text. "
    "Answers must be short (a number, name, or short phrase)."
)

DROP_GENERATOR_USER_TEMPLATE = (
    "Passage:\n{passage}\n\n"
    "Write ONE question that requires counting, adding or subtracting numbers from the passage, "
    "comparing two values, or finding a first/last/most/least. "
    "The answer must be fully supported by the passage only.\n\n"
    'Respond with a single JSON object with keys "question" and "answer" only. No markdown.'
)


def drop_generator_user_block(passage: str) -> str:
    return DROP_GENERATOR_USER_TEMPLATE.format(passage=passage.strip())


OHIOLINE_GENERATOR_PROMPT_VERSION = "ohioline_ft_pairs/v1"

OHIOLINE_GENERATOR_SYSTEM = (
    "You write exam-style question and answer pairs for Ohio State University Extension "
    "(Ohioline) factsheet text. These pairs will be used to fine-tune a document-grounded "
    "QA model. Answers must not introduce facts outside the given excerpt. "
    "Prefer questions that require the excerpt (not pure general knowledge). "
    "Prefer clear, specific questions and concise answers useful as training targets."
)


def ohioline_generator_user_block(excerpt: str, *, n_pairs: int = 2) -> str:
    n = max(1, int(n_pairs))
    return (
        "You are given an excerpt from an Ohioline factsheet (passages may be fragmented). "
        f"Produce exactly {n} distinct question–answer pairs for fine-tuning a "
        "document-grounded QA model.\n\n"
        f"Requirements:\n"
        f"- All answers must be fully grounded in the excerpt only.\n"
        f"- The {n} questions should cover different parts of the excerpt "
        "(try to cover the entire chunk; avoid near-duplicates).\n"
        f"- Prefer document-specific facts, recommendations, numbers, or procedures "
        "over trivial yes/no questions.\n"
        f"- Keep answers concise and suitable as SFT targets.\n\n"
        "Excerpt:\n"
        + excerpt.strip()
        + "\n\n"
        'Respond with a single JSON object of the form '
        '{"pairs":[{"question":"...","answer":"..."}, ...]} '
        f"with exactly {n} items in \"pairs\". No markdown fences."
    )


# Matches train_normalized_qa_sft.py `system_prompt` + `row_to_messages` user block.
ANSWERER_SYSTEM = (
    "You answer questions using only the provided context. "
    "Stay grounded in that text and do not invent information."
)


def answerer_user_block(context: str, question: str) -> str:
    return (
        "Context:\n"
        + context.strip()
        + "\n\nQuestion: "
        + question.strip()
        + "\n\nAnswer the question using only the context above. Be direct and concise."
    )


# Context-gap benchmarking: same answerer style without a document (general knowledge).
QUESTION_ONLY_SYSTEM = (
    "You answer questions clearly and concisely. "
    "No document is provided; use general knowledge where helpful."
)


def answerer_question_only_block(question: str) -> str:
    return (
        "Answer the following question. Be direct and concise.\n\n"
        "Question: "
        + question.strip()
    )
