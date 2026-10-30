"""Run slime's ``train.py`` or ``train_async.py`` with the toolkit's flags registered.

    python3 -m agentcore_rl_toolkit.backends.slime.train ${SLIME_DIR}/train.py [slime args] \
        --sfc-num-auxiliary-token-samples 8 --sfc-kl-beta 0.3

slime's ``parse_args`` drops flags it does not define; this entrypoint passes
``add_custom_arguments`` so they reach every actor through ``args``.
"""

import importlib.util
import sys

from agentcore_rl_toolkit.backends.slime.integration.score_centering import add_arguments


def main() -> None:
    script = sys.argv.pop(1)
    spec = importlib.util.spec_from_file_location("slime_train_script", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # imported, not run: its __main__ block is skipped
    from slime.utils.arguments import parse_args

    module.train(parse_args(add_custom_arguments=add_arguments))


if __name__ == "__main__":
    main()
