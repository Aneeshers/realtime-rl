#!/bin/bash
set -e

echo "Running all plot scripts..."

echo "Running plot_main_results.py..."
python3 plot_main_results.py

echo "Running plot_method.py..."
python3 plot_method.py

echo "Running plot_scaling.py..."
python3 plot_scaling.py

echo "Running plot_strategy.py..."
python3 plot_strategy.py

echo "Running plot_tetris_committed.py..."
python3 plot_tetris_committed.py

echo "Running plot_pacman_interpretability.py..."
python3 plot_pacman_interpretability.py

echo "Running plot_tetris_interpretability.py..."
python3 plot_tetris_interpretability.py

echo "Running plot_interpretability.py (combined)..."
python3 plot_interpretability.py

echo "All plots generated successfully!"
