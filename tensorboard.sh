#!/usr/bin/env zsh

dir="$HOME/ray_results/$(exa --sort oldest ~/ray_results | head -1)"
echo "Starting tensorboard in directory $dir"
tensorboard --logdir "$dir"
