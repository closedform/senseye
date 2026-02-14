"""Headless node entry point: python -m senseye.node"""

import sys


def main() -> None:
    # Inject --headless before argument parsing.
    if "--headless" not in sys.argv:
        sys.argv.insert(1, "--headless")
    from senseye.main import main as run_main

    run_main()


if __name__ == "__main__":
    main()
