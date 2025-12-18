#!/bin/bash

# Find and kill run_experiments.py processes
echo "Stopping run_experiments.py..."
pids=$(pgrep -f "run_experiments.py")
if [ -n "$pids" ]; then
    echo "Found processes: $pids"
    kill -9 $pids
    echo "Killed run_experiments.py"
else
    echo "No run_experiments.py processes found."
fi

# Find and kill predict_zero_shot.py processes
echo "Stopping predict_zero_shot.py..."
pids=$(pgrep -f "predict_zero_shot.py")
if [ -n "$pids" ]; then
    echo "Found processes: $pids"
    kill -9 $pids
    echo "Killed predict_zero_shot.py"
else
    echo "No predict_zero_shot.py processes found."
fi

# Find and kill evaluate_realiad.py processes (in case evaluation is running)
echo "Stopping evaluate_realiad.py..."
pids=$(pgrep -f "evaluate_realiad.py")
if [ -n "$pids" ]; then
    echo "Found processes: $pids"
    kill -9 $pids
    echo "Killed evaluate_realiad.py"
else
    echo "No evaluate_realiad.py processes found."
fi

echo "All experiment processes stopped."

