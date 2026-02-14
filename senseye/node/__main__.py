"""Headless node entry point: python -m senseye.node"""

import sys

# Inject --headless before parsing
sys.argv.insert(1, "--headless")

from senseye.main import main

main()
