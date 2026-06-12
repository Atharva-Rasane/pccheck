#!/bin/bash
set -euo pipefail

"${PCCHECK_PYTHON:-python3}" run_throughput_micro.py "$@"
