from bench.cli import main
from bench.models import call_model


def test_call_model_reports_usage_and_cost():
    result = call_model("anthropic/claude-haiku-4-5", "hi", mock_response="Hello")
    assert result.text == "Hello"
    assert result.input_tokens > 0 and result.output_tokens > 0
    assert result.cost_usd is not None and result.cost_usd > 0


def test_hello_command_runs_mocked(capsys):
    assert main(["hello", "--mock"]) == 0
    out = capsys.readouterr().out
    assert "cost:" in out and "mocked" in out


def test_format_error_keeps_multiline_provider_message():
    from bench.models import format_error

    e = ValueError('GeminiException - {\n  "error": {\n    "message": "models/x is not found"\n  }\n}')
    assert "models/x is not found" in format_error(e)
    assert "\n" not in format_error(e)


def test_format_error_drops_embedded_traceback():
    from bench.models import format_error

    e = ValueError('Missing key.\nTraceback (most recent call last):\n  File "x.py", line 1')
    assert format_error(e) == "ValueError: Missing key."
