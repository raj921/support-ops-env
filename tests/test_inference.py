from support_ops_env.inference import log_end, log_start, log_step, resolve_api_key


def test_resolve_api_key_prefers_hf_token():
    assert resolve_api_key({"HF_TOKEN": "hf-secret", "OPENAI_API_KEY": "openai-secret"}) == "hf-secret"


def test_resolve_api_key_uses_openai_fallback():
    assert resolve_api_key({"OPENAI_API_KEY": "openai-secret"}) == "openai-secret"


def test_logging_format(capsys):
    log_start(task="easy_vip_sso", env="support_ops_env", model="gpt-4.1-mini")
    log_step(step=1, action='{"action_type":"submit"}', reward=0.5, done=False, error=None)
    log_end(success=True, steps=1, score=0.5, rewards=[0.5])

    lines = capsys.readouterr().out.strip().splitlines()
    assert lines[0] == "[START] task=easy_vip_sso env=support_ops_env model=gpt-4.1-mini"
    assert lines[1].startswith("[STEP] step=1 action=")
    assert "reward=0.50" in lines[1]
    assert "done=false" in lines[1]
    assert "error=null" in lines[1]
    assert lines[2] == "[END] success=true steps=1 score=0.500 rewards=0.50"
