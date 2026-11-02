#!/usr/bin/env bash
# Dataset-free test suites, as run by .github/workflows/ci.yml (job "python").
# Needs no data/external, eval/cache or eval/fresh_data, no audio device, no board.
#
#   PY=.venv/bin/python bash scripts/ci/python_tests.sh        # local
#   PY=python bash scripts/ci/python_tests.sh                  # CI (setup-python)
#
# The live tests rewrite tracked files under live/runs/ and live/cache/ (see docs/REPO_AUDIT.md).
# Locally, run this in a scratch copy (git worktree or git archive) if you do not want those diffs.
set -euo pipefail
PY="${PY:-python3}"
cd "$(dirname "$0")/../.."
export MPLBACKEND=Agg PYTHONDONTWRITEBYTECODE=1

step() { echo; echo "== $*"; }

step "synthetic regression (run_v0.py; prints metrics, no assertions)"
"$PY" run_v0.py
step "harmony unit tests"
"$PY" -m unittest discover -s harmony/tests
step "echo tests"
"$PY" echo/tests/test_echo.py
step "eval: v2 unit tests, stats, bootstrap vs scipy"
"$PY" eval/tests/test_v2.py
"$PY" eval/tests/test_stats.py
"$PY" eval/audit/test_bootstrap_scipy.py
step "firmware protocol codec + fake device over a pty"
"$PY" -m unittest firmware/tools/test_proto.py
"$PY" -m unittest firmware/tools/test_fake_device.py
step "live app: regenerate synthetic inputs (byte-identical to the tracked ones on macOS), then --quick tests"
"$PY" live/tests/make_inputs.py
"$PY" live/tests/run_tests.py --quick
step "hardware: schematic text lint and the committed netlist against the PLAN.md pin map"
"$PY" hardware/scripts/lint_schematic.py
"$PY" hardware/scripts/check_netlist.py --report /dev/null
step "mech: meshes watertight, DXF plates valid, no collisions over the nod range"
"$PY" mech/cad/check_meshes.py
"$PY" mech/cad/check_dxf.py
"$PY" mech/cad/check_collisions.py > /dev/null
echo
echo "python tests: ALL OK"
