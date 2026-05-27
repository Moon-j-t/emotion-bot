"""Patience early-stop logic smoke test (same rules as MTL notebook)."""
from __future__ import annotations

import sys
from pathlib import Path

_SCRIPTS = Path(__file__).resolve().parent
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from mtl_notebook_utils import patience_stop_epoch  # noqa: E402


def main() -> None:
  # improve ep1, then 4 flat epochs -> stop at ep5
    losses = [3.0, 2.5, 2.6, 2.61, 2.62, 9.0]
    stopped, best_ep = patience_stop_epoch(losses, patience=4)
    assert stopped == 6, f"expected stop epoch 6, got {stopped}"
    assert best_ep == 2, f"expected best epoch 2, got {best_ep}"

    # reset after improvement
    losses2 = [3.0, 2.8, 2.9, 2.7, 2.71, 2.72, 2.73, 2.74]
    stopped2, best2 = patience_stop_epoch(losses2, patience=4)
    assert stopped2 == 8, f"expected stop epoch 8, got {stopped2}"
    assert best2 == 4, f"expected best epoch 4, got {best2}"

    # never triggers within run
    losses3 = [3.0, 2.0, 1.5, 1.0]
    stopped3, _ = patience_stop_epoch(losses3, patience=4)
    assert stopped3 is None

    print("OK: patience logic matches notebook")


if __name__ == "__main__":
    main()
