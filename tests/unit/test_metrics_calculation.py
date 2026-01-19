"""
Unit tests for domain metrics calculation helpers.
"""

from scripts.eval_utils import exact_match, is_command_only, normalize_command


def test_exact_match_identical() -> None:
    assert exact_match("ls -la", "ls -la")


def test_exact_match_whitespace_normalized() -> None:
    assert exact_match("ls   -la  ", "  ls -la")
    assert exact_match("ls -la\n", "ls -la  ")


def test_exact_match_different() -> None:
    assert not exact_match("ls -la", "ls -l")
    assert not exact_match("find .", "grep foo")


def test_exact_match_case_sensitive() -> None:
    assert not exact_match("ls -la", "LS -LA")


def test_exact_match_empty() -> None:
    assert exact_match("", "")
    assert not exact_match("ls", "")


def test_normalize_command_collapses_whitespace() -> None:
    assert normalize_command("ls   -la") == "ls -la"


def test_command_only_valid() -> None:
    assert is_command_only("ls -la /home")
    assert is_command_only("find . -name '*.py'")
    assert is_command_only("grep -r 'pattern' /var/log")


def test_command_only_with_markdown() -> None:
    assert not is_command_only("```bash\nls -la\n```")
    assert not is_command_only("```\nls -la\n```")


def test_command_only_with_explanation() -> None:
    assert not is_command_only("Here's the command: ls -la")
    assert not is_command_only("You can use: ls -la")
    assert not is_command_only("This command lists files: ls -la")
    assert not is_command_only("To do this, use ls -la")


def test_command_only_empty() -> None:
    assert not is_command_only("")
    assert not is_command_only("   ")
    assert not is_command_only("\n\n")


def test_command_only_multiline_with_continuation() -> None:
    multiline_valid = """find . -name '*.py' \\
  -type f \\
  -exec grep 'pattern' {} \\;"""
    assert is_command_only(multiline_valid)


def test_command_only_multiline_with_pipe() -> None:
    piped_command = "cat file.txt | grep 'error' | wc -l"
    assert is_command_only(piped_command)


def test_command_only_excessive_newlines() -> None:
    with_explanation = """ls -la

This command lists all files in the current directory."""
    assert not is_command_only(with_explanation)


def test_command_only_multiple_unrelated_lines() -> None:
    multiple = """ls -la
cd /home
pwd
whoami"""
    assert not is_command_only(multiple)


def test_aggregate_metrics_all_correct() -> None:
    results = [
        {"exact_match": True, "command_only": True, "syntax_valid": True},
        {"exact_match": True, "command_only": True, "syntax_valid": True},
        {"exact_match": True, "command_only": True, "syntax_valid": True},
    ]

    exact_match_accuracy = sum(r["exact_match"] for r in results) / len(results)
    command_only_rate = sum(r["command_only"] for r in results) / len(results)
    syntax_validity = sum(r["syntax_valid"] for r in results) / len(results)

    assert exact_match_accuracy == 1.0
    assert command_only_rate == 1.0
    assert syntax_validity == 1.0


def test_aggregate_metrics_partial() -> None:
    results = [
        {"exact_match": True, "command_only": True, "syntax_valid": True},
        {"exact_match": False, "command_only": True, "syntax_valid": True},
        {"exact_match": True, "command_only": False, "syntax_valid": True},
        {"exact_match": True, "command_only": True, "syntax_valid": False},
    ]

    exact_match_accuracy = sum(r["exact_match"] for r in results) / len(results)
    command_only_rate = sum(r["command_only"] for r in results) / len(results)
    syntax_validity = sum(r["syntax_valid"] for r in results) / len(results)

    assert exact_match_accuracy == 0.75
    assert command_only_rate == 0.75
    assert syntax_validity == 0.75


def test_aggregate_metrics_all_failed() -> None:
    results = [
        {"exact_match": False, "command_only": False, "syntax_valid": False},
        {"exact_match": False, "command_only": False, "syntax_valid": False},
    ]

    exact_match_accuracy = sum(r["exact_match"] for r in results) / len(results)
    command_only_rate = sum(r["command_only"] for r in results) / len(results)
    syntax_validity = sum(r["syntax_valid"] for r in results) / len(results)

    assert exact_match_accuracy == 0.0
    assert command_only_rate == 0.0
    assert syntax_validity == 0.0
