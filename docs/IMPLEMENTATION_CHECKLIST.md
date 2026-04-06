# Implementation Checklist

## Slice 1: Submission reliability

- [ ] `pip install -e '.[dev]'`
- [ ] `pytest -q`
- [ ] `openenv validate .`
- [ ] `scripts/validate-local.sh --skip-docker`
- [ ] `scripts/validate-local.sh` on a Docker-enabled machine

## Slice 2: Benchmark quality

- [ ] verify each task has a strong-score happy path
- [ ] verify unsafe or irrelevant actions reduce final score
- [ ] verify duplicate handling is required for medium and hard tasks
- [ ] verify hard task reply does not overclaim a confirmed breach

## Slice 3: Docs and demo polish

- [ ] README reflects final baseline and deployment instructions
- [ ] benchmark brief and glossary are present
- [ ] Hugging Face Space secrets are configured
- [ ] remote `/health` and `/reset` checks succeed

## Slice 4: Final submission

- [ ] push to Hugging Face Space
- [ ] run remote validator against the Space URL
- [ ] record final baseline scores and URL in the submission notes
