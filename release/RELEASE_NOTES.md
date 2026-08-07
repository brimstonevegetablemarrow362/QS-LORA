# Release v0.1.0 — QA generator LoRA

## Assets

| Asset | Notes |
|-------|-------|
| `qa-generator-lora.zip` | PEFT adapter + tokenizer (~89 MB compressed) |
| `qa-generator-lora.zip.sha256` | Checksum |

> Full merged ~6 GB checkpoint is **not** attached (GitHub 2 GB/file limit + Llama redistribution). Merge locally after download.

## Install from this release

```bash
# after cloning the repo
scripts/download_release_model.sh ./qa-generator-lora.zip
scripts/merge_generator.sh
# start vLLM on ./out/merged-qa-generator
scripts/docs_to_qa.sh examples/sample_docs/sample.md examples/qa/sample.jsonl
```

## Verify

```bash
sha256sum -c qa-generator-lora.zip.sha256
```

## License

See `NOTICE` and Meta Llama 3.2 license for the adapter / base model.
