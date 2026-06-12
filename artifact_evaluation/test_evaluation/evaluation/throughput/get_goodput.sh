#!/bin/bash
set -euo pipefail

"${PCCHECK_PYTHON:-python3}" run_goodput_micro.py "$@"
