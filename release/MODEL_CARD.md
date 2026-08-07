---
base_model: meta-llama/Llama-3.2-3B-Instruct
library_name: peft
tags:
  - lora
  - question-answering
  - synthetic-data
  - llama
pipeline_tag: text-generation
---

# QA generator LoRA (`qa-generator-lora`)

PEFT LoRA adapter that turns a short document excerpt into one grounded
`{"question","answer"}` JSON object. Intended for **synthetic QA data generation**
before QS-LoRA / SFT training.

## Requirements

- Base: [`meta-llama/Llama-3.2-3B-Instruct`](https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct)
  (accept Meta license; set `HF_TOKEN` if gated).
- Merge with `pipeline/merge_lora.py` or `scripts/merge_generator.sh`, then serve with **vLLM**.

## Files in this release zip

| File | Role |
|------|------|
| `adapter_model.safetensors` | LoRA weights (~93 MB) |
| `adapter_config.json` | PEFT config (`r=16`, `lora_alpha=32`, …) |
| `tokenizer*` / `chat_template.jinja` | Tokenizer + chat template |

## Prompt (aligned with training)

System: exam-style Q/A; answers must stay inside the excerpt.  
User: excerpt + request for a single JSON object with `question` and `answer`.

See `pipeline/prompts.py`.

## License

Adapter is a Llama derivative — Meta Llama Community License. Code that loads it is MIT.
