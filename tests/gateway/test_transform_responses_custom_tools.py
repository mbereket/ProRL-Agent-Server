"""Responses API ``custom`` (freeform) tools through the gateway.

Codex declares ``apply_patch`` as a ``custom`` tool: the model must produce free
text (a patch), which codex receives as ``custom_tool_call.input`` and answers
with ``custom_tool_call_output``. Chat-completion models only call functions,
so the gateway presents the tool as a function with one ``input`` string, maps
the model's call back to a ``custom_tool_call`` item, and re-renders echoed
history turns with the exact arguments the model generated.
"""

from __future__ import annotations

import json

from polar.gateway.transform.openai_responses import (
    OpenAIResponsesTransformer,
    ResponsesStreamState,
    custom_tool_names,
)
from polar.gateway.transform.reasoning import assistant_replay_entries

PATCH = "*** Begin Patch\n*** Update File: moto/organizations/models.py\n@@\n-    pass\n+    return True\n*** End Patch\n"
GRAMMAR = 'start: begin_patch hunk+ end_patch\nbegin_patch: "*** Begin Patch" LF\n'

# Shape of the request codex 0.125 sends for a model whose apply_patch tool type
# is "freeform": a function tool plus the custom apply_patch tool.
CODEX_REQUEST = {
    "_polar_model_served": "Qwen/Qwen3.5-9B",
    "instructions": "You are Codex.",
    "input": [{"type": "message", "role": "user", "content": [{"type": "input_text", "text": "Fix the bug."}]}],
    "tools": [
        {
            "type": "function",
            "name": "exec_command",
            "description": "Run a shell command.",
            "parameters": {"type": "object", "properties": {"cmd": {"type": "string"}}, "required": ["cmd"]},
        },
        {
            "type": "custom",
            "name": "apply_patch",
            "description": "Use the `apply_patch` tool to edit files.",
            "format": {"type": "grammar", "syntax": "lark", "definition": GRAMMAR},
        },
    ],
    "tool_choice": "auto",
    "stream": False,
}


def test_custom_tool_is_declared_to_the_model_as_a_function_with_input() -> None:
    transformed = OpenAIResponsesTransformer().transform_request(dict(CODEX_REQUEST))

    names = [t["function"]["name"] for t in transformed["tools"]]
    assert names == ["exec_command", "apply_patch"]
    apply_patch = transformed["tools"][1]["function"]
    assert apply_patch["parameters"]["required"] == ["input"]
    assert apply_patch["parameters"]["properties"]["input"]["type"] == "string"
    assert apply_patch["description"].startswith("Use the `apply_patch` tool")
    assert "lark grammar" in apply_patch["description"]
    assert GRAMMAR.strip() in apply_patch["description"]
    # The function tool is untouched.
    assert transformed["tools"][0]["function"]["parameters"]["required"] == ["cmd"]


def test_custom_tool_choice_maps_to_the_function_name() -> None:
    body = dict(CODEX_REQUEST, tool_choice={"type": "custom", "name": "apply_patch"})
    transformed = OpenAIResponsesTransformer().transform_request(body)
    assert transformed["tool_choice"] == {"type": "function", "function": {"name": "apply_patch"}}


def test_unsupported_tool_types_are_still_dropped() -> None:
    body = dict(CODEX_REQUEST, tools=CODEX_REQUEST["tools"] + [{"type": "web_search"}, {"type": "mcp", "server_url": "x"}])
    transformed = OpenAIResponsesTransformer().transform_request(body)
    assert [t["function"]["name"] for t in transformed["tools"]] == ["exec_command", "apply_patch"]


def test_model_call_to_custom_tool_becomes_custom_tool_call_item() -> None:
    arguments = json.dumps({"input": PATCH})
    response = OpenAIResponsesTransformer().transform_response(
        {
            "id": "chatcmpl-1",
            "choices": [
                {
                    "message": {
                        "content": None,
                        "tool_calls": [
                            {"id": "call-patch", "function": {"name": "apply_patch", "arguments": arguments}},
                            {"id": "call-cmd", "function": {"name": "exec_command", "arguments": '{"cmd": "pytest"}'}},
                        ],
                    }
                }
            ],
        },
        CODEX_REQUEST,
    )

    patch_item, cmd_item = response["output"]
    assert patch_item["type"] == "custom_tool_call"
    assert patch_item["call_id"] == "call-patch"
    assert patch_item["name"] == "apply_patch"
    assert patch_item["input"] == PATCH
    assert "arguments" not in patch_item
    assert cmd_item["type"] == "function_call"
    assert cmd_item["arguments"] == '{"cmd": "pytest"}'


def test_malformed_custom_arguments_pass_through_as_input() -> None:
    response = OpenAIResponsesTransformer().transform_response(
        {
            "id": "chatcmpl-1",
            "choices": [
                {
                    "message": {
                        "content": None,
                        "tool_calls": [{"id": "c", "function": {"name": "apply_patch", "arguments": PATCH}}],
                    }
                }
            ],
        },
        CODEX_REQUEST,
    )
    assert response["output"][0]["input"] == PATCH


def test_streamed_custom_tool_call_is_emitted_whole_at_finalize() -> None:
    state = ResponsesStreamState(model="m", custom_tool_names=custom_tool_names(CODEX_REQUEST["tools"]))
    arguments = json.dumps({"input": PATCH})
    events = state.process_chunk(
        {"choices": [{"delta": {"tool_calls": [{"index": 0, "id": "call-patch", "function": {"name": "apply_patch", "arguments": arguments[:10]}}]}}]},
        is_first=True,
    )
    events += state.process_chunk(
        {"choices": [{"delta": {"tool_calls": [{"index": 0, "function": {"arguments": arguments[10:]}}]}}]},
    )
    events += state.finalize()

    types = [e["type"] for e in events]
    assert "response.function_call_arguments.delta" not in types
    added = next(e for e in events if e["type"] == "response.output_item.added")
    assert added["item"]["type"] == "custom_tool_call" and added["item"]["input"] == ""
    done_input = next(e for e in events if e["type"] == "response.custom_tool_call_input.done")
    assert done_input["input"] == PATCH
    done_item = next(e for e in events if e["type"] == "response.output_item.done")
    assert done_item["item"]["type"] == "custom_tool_call"
    assert done_item["item"]["input"] == PATCH
    completed = events[-1]
    assert completed["type"] == "response.completed"
    assert completed["response"]["output"][0]["type"] == "custom_tool_call"
    assert completed["response"]["output"][0]["call_id"] == "call-patch"


def test_echoed_custom_tool_turn_re_renders_the_exact_generated_arguments() -> None:
    # The model generated arguments with its own spacing; codex echoes only the
    # unpacked input. The gateway's per-call replay entry restores the original.
    generated = '{"input":"' + json.dumps(PATCH)[1:-1] + '"}'
    replay = assistant_replay_entries(
        {"choices": [{"message": {"content": "Applying the fix.", "tool_calls": [{"id": "call-patch", "function": {"name": "apply_patch", "arguments": generated}}]}}]}
    )
    assert replay["call:call-patch"]["arguments"] == generated
    assert replay["call:call-patch"]["content"] == "Applying the fix."

    body = dict(
        CODEX_REQUEST,
        _polar_reasoning_replay=replay,
        input=[
            {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "Fix the bug."}]},
            {"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": "Applying the fix."}]},
            {"type": "custom_tool_call", "call_id": "call-patch", "name": "apply_patch", "input": PATCH},
            {"type": "custom_tool_call_output", "call_id": "call-patch", "output": "Success. Updated the following files:\nM moto/organizations/models.py"},
            {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "Now run the tests."}]},
        ],
    )
    messages = OpenAIResponsesTransformer().transform_request(body)["messages"]

    assistant = next(m for m in messages if m["role"] == "assistant" and m.get("tool_calls"))
    assert assistant["tool_calls"] == [
        {"id": "call-patch", "type": "function", "function": {"name": "apply_patch", "arguments": generated}}
    ]
    tool_msg = next(m for m in messages if m["role"] == "tool")
    assert tool_msg["tool_call_id"] == "call-patch"
    assert tool_msg["content"].startswith("Success. Updated")
    assert messages[-1] == {"role": "user", "content": "Now run the tests."}


def test_echoed_custom_tool_turn_without_replay_rebuilds_input_arguments() -> None:
    body = dict(
        CODEX_REQUEST,
        input=[
            {"type": "custom_tool_call", "call_id": "call-patch", "name": "apply_patch", "input": PATCH},
            {"type": "custom_tool_call_output", "call_id": "call-patch", "output": "Done"},
        ],
    )
    messages = OpenAIResponsesTransformer().transform_request(body)["messages"]
    tool_call = messages[-2]["tool_calls"][0]
    assert tool_call["function"]["name"] == "apply_patch"
    assert json.loads(tool_call["function"]["arguments"]) == {"input": PATCH}
    assert messages[-1] == {"role": "tool", "tool_call_id": "call-patch", "content": "Done"}


def test_custom_tool_turn_renders_like_a_function_tool_turn_for_training() -> None:
    # Training builds traces from the chat-side request; a custom tool turn must
    # be indistinguishable from a function tool turn with the same arguments.
    arguments = json.dumps({"input": PATCH})
    replay = {"call:c1": {"content": "", "reasoning": "", "arguments": arguments}}
    as_custom = OpenAIResponsesTransformer().transform_request(
        dict(CODEX_REQUEST, _polar_reasoning_replay=replay, input=[
            {"type": "custom_tool_call", "call_id": "c1", "name": "apply_patch", "input": PATCH},
            {"type": "custom_tool_call_output", "call_id": "c1", "output": "ok"},
        ])
    )
    as_function = OpenAIResponsesTransformer().transform_request(
        dict(CODEX_REQUEST, input=[
            {"type": "function_call", "call_id": "c1", "name": "apply_patch", "arguments": arguments},
            {"type": "function_call_output", "call_id": "c1", "output": "ok"},
        ])
    )
    assert as_custom["messages"] == as_function["messages"]
    assert as_custom["tools"] == as_function["tools"]
