#!/bin/sh
# Copy all runs written by EV3_Control_runs.py from the brick (nothing is deleted there).
# Usage (from the report_hw02 folder):  sh code/fetch_all.sh
set -e
EV3="${EV3:-ev3dev.local}"
mkdir -p data/encoder/runs
scp -q "robot@${EV3}:see/runs/*" data/encoder/runs/
echo "$(ls data/encoder/runs/*_both_motors_path.csv | wc -l | tr -d ' ') runs copied; battery log:"
cat data/encoder/runs/runs_log.csv
