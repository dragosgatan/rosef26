"""The authors' `python -m ddo_defense.cli`, with one optional switch.

usage: .venv/bin/python scripts/ddo_cli.py [--no-confusion-filter] apply ...

--no-confusion-filter trains the confusion loss on every harmful/harmless probe
instead of only those the model with untrained decoys still refuses/answers
(`DDOOptimizer.fit(confusion_filter_train=False)`, an option the CLI does not
expose). third_party/ddo itself is not edited.
"""

import sys

from ddo_defense.cli import main
from ddo_defense.defense.optimizer import DDOOptimizer

if __name__ == "__main__":
    argv = sys.argv[1:]
    if "--no-confusion-filter" in argv:
        argv.remove("--no-confusion-filter")
        _fit = DDOOptimizer.fit

        def fit(self, *args, **kwargs):
            kwargs["confusion_filter_train"] = False
            print("  confusion filter OFF: training on every probe prompt", flush=True)
            return _fit(self, *args, **kwargs)

        DDOOptimizer.fit = fit
    raise SystemExit(main(argv))
