#!/bin/bash

# Define an array of experiment configurations without variance reduction
nonmyopic_configurations=(
    #################################################################################
    #################### VARIANCE REDUCTION BASED EXPERIMENTS ######################
    #################################################################################
    # Beginning of Horizon 0 Nonmyopic Experiments
    "--output-dir=../no-truncated-horizons --optimize --variance-reduction --function-name=braninhoo --horizon 1 --trials 30"
)

# Loop over the experiment configurations and run each one in the background
for config in "${nonmyopic_configurations[@]}"; do
  # Run the Julia code in the background with the given configuration and output filename
  julia ../nonmyopic_bayesopt.jl $config
done