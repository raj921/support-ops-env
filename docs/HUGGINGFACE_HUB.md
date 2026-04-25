# Hugging Face Hub — alignment with official docs

This repo is pushed to **GitHub** and to a **Hugging Face Space** (`git remote add hf https://huggingface.co/spaces/...`). Hub rules differ from plain GitHub: **large and binary files must not live as raw blobs in Git history**.

## 1. Git push to the Hub (Spaces / models / datasets)

Follow **[Getting started with repositories](https://huggingface.co/docs/hub/main/en/repositories-getting-started)**:

- Install **Git** and **Git LFS**.
- Install **[Git-Xet](https://huggingface.co/docs/hub/main/en/xet/using-xet-storage#git-xet)** (Hub’s supported path for large files), then in each clone:

  ```bash
  git xet install
  ```

- For files **over ~10 MB**, or binaries not already covered, use **[`git xet track`](https://huggingface.co/docs/hub/main/en/repositories-getting-started#set-up)** (or the patterns in root **`.gitattributes`**) so Git stores **pointer files**, not the blob.

If push fails with *“Your push was rejected because it contains binary files”* / *“Please use … xet”*:

1. Install **Git-Xet** and run `git xet install` in the repository.
2. Track offending extensions (e.g. `git xet track "*.png"`).
3. If the blob is **already in history**, remove it from history (e.g. rewrite / squash) or the remote will keep rejecting **old commits** that still contain the binary.

**Python uploads:** [`hf upload`](https://huggingface.co/docs/hub/main/en/repositories-getting-started#adding-files-to-a-repository-cli) handles large files without manual LFS/Xet setup:

```bash
pip install -U "huggingface_hub>=0.32"   # brings hf_xet per Hub Xet docs
hf auth login
hf upload <user>/<repo> ./path/to/large.bin --repo-type space
```

See also: **[Using Xet storage](https://huggingface.co/docs/hub/main/en/xet/using-xet-storage)** and **[Storage backends](https://huggingface.co/docs/hub/main/en/storage-backends)**.

## 2. TRL GRPO + vLLM (training)

Official TRL section: **[GRPO Trainer — Speed up training with vLLM](https://huggingface.co/docs/trl/main/en/grpo_trainer#speed-up-training-with-vllm-powered-generation)**.

- Install: `pip install "trl[vllm]>=0.29.0"` (see TRL docs).
- This project **additionally pins** `vllm>=0.11,<0.19` because current TRL warns on **vLLM 0.19+** and colocate mode can raise `NameError: LLM is not defined` until TRL catches up.

## 3. `openenv-core` and small deps

`openenv-core` may expect **`tomli`** / **`tomli-w`** on Python 3.12; Colab/Kaggle install cells include them to satisfy the resolver.

## Doc index

| Topic | URL |
|------|-----|
| Hub (overview) | https://huggingface.co/docs/hub/index |
| Repositories (Git + Xet) | https://huggingface.co/docs/hub/main/en/repositories-getting-started |
| Xet (Git install) | https://huggingface.co/docs/hub/main/en/xet/using-xet-storage |
| TRL GRPO | https://huggingface.co/docs/trl/main/en/grpo_trainer |
