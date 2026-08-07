# Publishing this tree to GitHub

1. Create an empty **public** repo (e.g. `qs-lora`) on GitHub — do not add a README there.
2. From this directory:

```bash
cd /users/PAS2699/pratham2210/qs-lora
git init
git add .
# zip + safetensors are gitignored; they stay on disk for the Release upload
git status   # confirm no .env / secrets
git commit -m "Initial public release: QA generator pipeline + QS-LoRA code"
git branch -M main
git remote add origin git@github.com:YOUR_USER/qs-lora.git
git push -u origin main
```

3. Create a Release `v0.1.0` and upload:
   - `release/qa-generator-lora.zip`
   - `release/qa-generator-lora.zip.sha256`
   Paste body from `release/RELEASE_NOTES.md`.

4. (Optional) Also upload the adapter folder to Hugging Face and link it from the README.
