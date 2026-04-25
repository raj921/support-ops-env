# Kaggle Kernel (CLI push)

Folder shaped for [`kaggle kernels push`](https://github.com/Kaggle/kaggle-api/blob/main/docs/README.md):

- `kernel-metadata.json` — kernel id, GPU, internet, main notebook filename
- `driftshield_kaggle.ipynb` — copy of the repo-root notebook (keep in sync when you edit the root file)

## One-time setup

1. `pip install kaggle`
2. Put API credentials in `~/.kaggle/kaggle.json` (Kaggle → Settings → API → Create New Token).

## Before first push

The **`id`** in `kernel-metadata.json` must be `YOUR_KAGGLE_USERNAME/driftshield-grpo`. This repo uses **`rajkumar295/driftshield-grpo`**; change the username if yours differs.

## Push

From the **repository root**:

```bash
./scripts/push_kaggle_kernel.sh
```

Or manually:

```bash
kaggle kernels push -p kernels/driftshield-grpo
```

After editing the **root** `driftshield_kaggle.ipynb`, refresh the copy here:

```bash
cp driftshield_kaggle.ipynb kernels/driftshield-grpo/driftshield_kaggle.ipynb
```

## Kaggle notebook UI

Add secret **`HF_TOKEN`** in the kernel's **Add-ons → Secrets** (or attach when editing on the site). Internet and GPU are requested in metadata; confirm they stay enabled after the first push.
