from support_ops_env.train import _milestone_reward, parse_tool_calls


def test_parse_tool_calls_accepts_clean_json():
    parsed = parse_tool_calls(
        '{"assistant_message":"Open the case.","tool_calls":[{"name":"inbox.open_case","args":{"case_id":"I-9001"}}],"answer":null}'
    )
    assert parsed["_parse_ok"] is True
    assert parsed["assistant_message"] == "Open the case."
    assert parsed["tool_calls"][0]["name"] == "inbox.open_case"


def test_parse_tool_calls_strips_think_and_prose():
    raw = """
    <think>First I should inspect the case.</think>
    Here is the action:
    {"assistant_message":"Inspect.","tool_calls":[{"name":"policy.search","args":{"query":"admin grant"}}],"answer":null}
    """
    parsed = parse_tool_calls(raw)
    assert parsed["_parse_ok"] is True
    assert parsed["assistant_message"] == "Inspect."
    assert parsed["tool_calls"][0]["name"] == "policy.search"


def test_parse_tool_calls_reports_failure_on_invalid_output():
    parsed = parse_tool_calls("not json at all")
    assert parsed["_parse_ok"] is False
    assert parsed["tool_calls"] == []
    assert parsed["answer"] is None
    assert "JSONDecodeError" in parsed["parse_error"]


def test_milestone_reward_increases_with_structural_progress():
    empty = _milestone_reward([])
    partial = _milestone_reward(
        [
            {"tool_calls": [{"name": "inbox.open_case", "args": {}}], "answer": None},
            {"tool_calls": [{"name": "policy.search", "args": {}}], "answer": None},
        ]
    )
    strong = _milestone_reward(
        [
            {"tool_calls": [{"name": "inbox.open_case", "args": {}}], "answer": None},
            {
                "tool_calls": [
                    {"name": "crm.get_account", "args": {}},
                    {"name": "policy.search", "args": {}},
                    {"name": "workflow.set_priority", "args": {}},
                    {"name": "workflow.assign_team", "args": {}},
                    {"name": "workflow.set_status", "args": {}},
                    {"name": "workflow.add_tags", "args": {}},
                    {"name": "comms.draft_reply", "args": {}},
                ],
                "answer": {"done": True},
            },
        ]
    )
    assert empty == 0.0
    assert partial > empty
    assert strong > partial
