# Contributing

## Local workflow

```bash
python3 -m venv .venv
. .venv/bin/activate
pip install -e '.[dev]'
pre-commit install
```

## Before opening a PR or submitting

Run the lightweight local submission check:

```bash
scripts/validate-local.sh --skip-docker
```

Run the full version on a Docker-enabled machine:

```bash
scripts/validate-local.sh
```

## Git safety

- Avoid destructive commands such as `git reset --hard` and `git clean -fd`.
- Prefer small commits and validate locally before changing deployment configuration.
- Treat the Docker image and Hugging Face Space as submission artifacts, not scratch environments.
