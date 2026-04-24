# Kaggle Kernel (CLI push)

This folder is shaped for [`kaggle kernels push`](https://github.com/Kaggle/kaggle-api/blob/main/docs/README.md):

- `kernel-metadata.json` — kernel id, GPU, internet, main notebook filename
- `support_ops_kaggle.ipynb` — copy of the repo-root notebook (keep in sync when you edit the root file)

## One-time setup

1. Install the CLI: `pip install kaggle`
2. Put API credentials in `~/.kaggle/kaggle.json` (download from Kaggle **Account → API → Create New Token**).

## Before first push

The **`id`** in `kernel-metadata.json` must be `YOUR_KAGGLE_USERNAME/support-ops-env-grpo`. This repo uses **`rajkumar295/support-ops-env-grpo`** (change the username if yours differs).

## Push

From the **repository root**, after `~/.kaggle/kaggle.json` exists:

```bash
./scripts/push_kaggle_kernel.sh
```

Or manually:

```bash
kaggle kernels push -p kernels/support-ops-grpo
```

After editing the **root** `support_ops_kaggle.ipynb`, refresh the copy here:

```bash
cp support_ops_kaggle.ipynb kernels/support-ops-grpo/support_ops_kaggle.ipynb
```

Then push again.

## Kaggle notebook UI

Add secret **`HF_TOKEN`** in the kernel’s **Add-ons → Secrets** (or attach when editing on the site). Internet and GPU are requested in metadata; confirm they stay enabled on Kaggle after the first push if the UI overrides them.
