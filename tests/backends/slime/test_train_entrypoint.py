"""The slime entrypoint registers the toolkit's flags and runs the given script's train()."""

import sys
from types import ModuleType

from agentcore_rl_toolkit.backends.slime import train


def test_main_runs_the_script_with_custom_arguments(tmp_path, monkeypatch):
    out = tmp_path / "trained_with"
    script = tmp_path / "train.py"
    script.write_text(
        "def train(args):\n"
        f"    open({str(out)!r}, 'w').write(args)\n"
        "if __name__ == '__main__':\n"
        "    raise SystemExit('must not run as a script')\n"
    )
    seen = {}

    def parse_args(add_custom_arguments=None):
        seen["hook"] = add_custom_arguments
        return "parsed-args"

    arguments = ModuleType("slime.utils.arguments")
    arguments.parse_args = parse_args
    monkeypatch.setitem(sys.modules, "slime.utils.arguments", arguments)
    monkeypatch.setattr(sys, "argv", ["train", str(script), "--sfc-kl-beta", "0.3"])
    train.main()

    assert seen["hook"] is train.add_arguments
    assert sys.argv == ["train", "--sfc-kl-beta", "0.3"]  # the script path is consumed
    assert out.read_text() == "parsed-args"
