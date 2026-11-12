"""python -m worker --role render|util  (run from the repo root; see README "Web-app worker")."""
from __future__ import annotations

import argparse
import logging
import os
import sys

from .config import REPO_ROOT, ROLES, load

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m worker", description="Character-swap worker for the web-app")
    ap.add_argument("--role", choices=ROLES, default="render",
                    help="render: run_swap.sh jobs (one per cluster); util: edit planner + character preparation")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s %(name)s | %(message)s",
                        datefmt="%H:%M:%S")
    try:
        config = load(args.role)
    except ValueError as error:
        logging.getLogger("worker").error("%s", error)
        return 2
    if config.role == "util":                 # the planner and insightface on the free GPU, before torch is imported
        os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
        os.environ.setdefault("CUDA_VISIBLE_DEVICES", config.planner_gpu)
    from .loop import Worker
    return Worker(config).run()


if __name__ == "__main__":
    sys.exit(main())
