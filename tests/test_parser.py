"""Parser tests for the uniform-mode response reader.

Every "real model" case below is a shape actually observed from the configured
suite during Phase 2/4 verification, not an invented one. The point of these
tests is to pin the distinction the error decomposition depends on: a response
whose JSON we failed to *find* must not be counted as a syntactic model error,
while a response that genuinely isn't a call must still be counted as one.
"""

import sys

sys.path.insert(0, "src")

from tur.harness.runner import parse_response, _extract_json, _top_level_objects


def _parse(text):
    return parse_response({"text": text}, "uniform")


def test_plain_object():
    p = _parse('{"tool": "op_a", "args": {"ref": 7}}')
    assert p.parse_ok and p.tool == "op_a" and p.args == {"ref": 7}


def test_markdown_fence():
    p = _parse('```json\n{"tool": "op_a", "args": {"ref": 7}}\n```')
    assert p.parse_ok and p.tool == "op_a" and p.args == {"ref": 7}


def test_prose_around_object():
    p = _parse('Sure! I will call the tool now:\n'
               '{"tool": "op_a", "args": {"ref": 7}}\nLet me know if that helps.')
    assert p.parse_ok and p.tool == "op_a"


def test_reasoning_block_with_braces():
    """The qwen3.6 case: <think> containing draft JSON.

    The old first-brace-to-last-brace slice spanned from the brace inside the
    reasoning to the final brace of the answer and parsed as neither.
    """
    text = ('<think>\nI should call op_a. Maybe {"tool": "op_WRONG"} is right?\n'
            'No, the schema says {"ref": int}.\n</think>\n'
            '{"tool": "op_a", "args": {"ref": 7}}')
    assert _extract_json(text).strip() == '{"tool": "op_a", "args": {"ref": 7}}'
    p = _parse(text)
    assert p.parse_ok and p.tool == "op_a" and p.args == {"ref": 7}


def test_reasoning_block_fenced_answer():
    text = ('<thinking>considering {"a": 1}</thinking>\n'
            'Here is the call:\n```json\n{"tool": "op_b", "args": {"ref": 12}}\n```')
    p = _parse(text)
    assert p.parse_ok and p.tool == "op_b" and p.args == {"ref": 12}


def test_unclosed_reasoning_is_a_real_failure():
    """Truncated mid-reasoning: no answer exists, so this must stay a failure."""
    p = _parse('<think>I need to work out which tool {"tool": "op_a"} hmm')
    assert not p.parse_ok


def test_nested_args_not_mistaken_for_the_call():
    spans = _top_level_objects('{"tool": "op_a", "args": {"ref": 7}}')
    assert spans == ['{"tool": "op_a", "args": {"ref": 7}}']


def test_braces_inside_strings_do_not_break_depth():
    p = _parse('{"tool": "op_a", "args": {"ref": 7}, "note": "use {this} not \\"that\\""}')
    assert p.parse_ok and p.tool == "op_a" and p.args == {"ref": 7}


def test_last_object_wins_when_model_revises():
    p = _parse('{"tool": "op_WRONG", "args": {"ref": 1}}\n'
               'Actually, correcting that:\n{"tool": "op_a", "args": {"ref": 7}}')
    assert p.parse_ok and p.tool == "op_a" and p.args == {"ref": 7}


def test_non_json_prose_is_still_a_parse_failure():
    assert not _parse("I would call op_a with ref 7.").parse_ok


def test_code_response_is_still_a_parse_failure():
    """The llama-3.1-8b case: emits Python instead of a call.

    This must remain a failure. It is a real result about the model's
    structured-output reliability, and the parser must not rescue it.
    """
    text = ('```python\nclass ToolUsingAgent:\n    def __init__(self):\n'
            '        self.alpha = self._alpha\n')
    assert not _parse(text).parse_ok


def test_empty_response_is_a_parse_failure():
    assert not _parse("").parse_ok


def test_backend_error_is_flagged_not_scored_as_syntax():
    p = parse_response({"text": "", "_backend_error": "timeout"}, "uniform")
    assert not p.parse_ok and p.is_backend_error


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"  {fn.__name__}: OK")
    print(f"\nALL {len(fns)} PARSER TESTS PASSED")
