from scripts.eval_utils import is_command_only


def test_command_only_true_for_simple_command() -> None:
    assert is_command_only("ls -la")


def test_command_only_false_for_explanation() -> None:
    assert not is_command_only("Here is the command: ls -la")


def test_command_only_false_for_code_block() -> None:
    assert not is_command_only("```bash\nls -la\n```")


def test_command_only_false_for_sentence() -> None:
    assert not is_command_only("Run ls -la.")
