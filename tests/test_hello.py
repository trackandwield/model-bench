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
