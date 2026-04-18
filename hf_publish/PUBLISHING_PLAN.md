# Hugging Face Publishing Checklist

1. Confirm `hf_publish/` contains only `README.md`, `app.py`, `requirements.txt`, and `PUBLISHING_PLAN.md`.
2. Create either a new Gradio Space or a lightweight companion repo on Hugging Face.
3. Upload only the four files from `hf_publish/`.
4. Do not upload `data/`, `artifacts/checkpoints/`, `artifacts/hf_cache/`, prediction CSVs, processed CSVs, or any text-bearing artifacts.
5. After upload, verify that the README and app clearly state that no datasets, text data, checkpoints, or model weights are included.
6. If you want clickable external links later, replace the repository-relative file references with canonical source-repo URLs after publishing.
