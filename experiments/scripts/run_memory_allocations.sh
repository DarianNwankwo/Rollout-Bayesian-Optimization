# This script is responsible for executing experiments that produce data that will
# allow us to analyze the memory and time considerations when solving our
# nonmyopic acquisition function. Namely, we care about the following scenarios:
#   1. Increasing the horizon for some fixed number of monte carlo simulations
#   2. Increasing the monte carlo simulations for each horizon ∈ {0, 1, 2, 3}
# 
# NOTE: This experiment depends on our choice of optimizer. We still need to
# implement sample average approximation (SAA) to get a sense of how quicker
# and potentially less memory allocations we can get away with.

myopic_configurations=(
  # Scenario 1
  "--output-dir ../timing-and-allocations --horizon 0 --optimize --starts 8 --budget 15 --trials 10 --mc-samples 1024 --function-name ackley10d"
  "--output-dir ../timing-and-allocations --horizon 1 --optimize --starts 8 --budget 15 --trials 10 --mc-samples 1024 --function-name ackley10d"
  "--output-dir ../timing-and-allocations --horizon 2 --optimize --starts 8 --budget 15 --trials 10 --mc-samples 1024 --function-name ackley10d"
  "--output-dir ../timing-and-allocations --horizon 3 --optimize --starts 8 --budget 15 --trials 10 --mc-samples 1024 --function-name ackley10d"
  # Scenario 2 Horizon 0
  "--output-dir ../timing-and-allocations --horizon 0 --optimize --starts 8 --budget 15 --trials 10 --mc-samples 32 --function-name ackley10d"
  "--output-dir ../timing-and-allocations --horizon 0 --optimize --starts 8 --budget 15 --trials 10 --mc-samples 64 --function-name ackley10d"
  "--output-dir ../timing-and-allocations --horizon 0 --optimize --starts 8 --budget 15 --trials 10 --mc-samples 128 --function-name ackley10d"
  "--output-dir ../timing-and-allocations --horizon 0 --optimize --starts 8 --budget 15 --trials 10 --mc-samples 256 --function-name ackley10d"
  "--output-dir ../timing-and-allocations --horizon 0 --optimize --starts 8 --budget 15 --trials 10 --mc-samples 512 --function-name ackley10d"
  "--output-dir ../timing-and-allocations --horizon 0 --optimize --starts 8 --budget 15 --trials 10 --mc-samples 1024 --function-name ackley10d"
  # Scenario 2 Horizon 1
  "--output-dir ../timing-and-allocations --horizon 1 --optimize --starts 8 --budget 15 --trials 10 --mc-samples 32 --function-name ackley10d"
  "--output-dir ../timing-and-allocations --horizon 1 --optimize --starts 8 --budget 15 --trials 10 --mc-samples 64 --function-name ackley10d"
  "--output-dir ../timing-and-allocations --horizon 1 --optimize --starts 8 --budget 15 --trials 10 --mc-samples 128 --function-name ackley10d"
  "--output-dir ../timing-and-allocations --horizon 1 --optimize --starts 8 --budget 15 --trials 10 --mc-samples 256 --function-name ackley10d"
  "--output-dir ../timing-and-allocations --horizon 1 --optimize --starts 8 --budget 15 --trials 10 --mc-samples 512 --function-name ackley10d"
  "--output-dir ../timing-and-allocations --horizon 1 --optimize --starts 8 --budget 15 --trials 10 --mc-samples 1024 --function-name ackley10d"
  # Scenario 2 Horizon 2
  "--output-dir ../timing-and-allocations --horizon 2 --optimize --starts 8 --budget 15 --trials 10 --mc-samples 32 --function-name ackley10d"
  "--output-dir ../timing-and-allocations --horizon 2 --optimize --starts 8 --budget 15 --trials 10 --mc-samples 64 --function-name ackley10d"
  "--output-dir ../timing-and-allocations --horizon 2 --optimize --starts 8 --budget 15 --trials 10 --mc-samples 128 --function-name ackley10d"
  "--output-dir ../timing-and-allocations --horizon 2 --optimize --starts 8 --budget 15 --trials 10 --mc-samples 256 --function-name ackley10d"
  "--output-dir ../timing-and-allocations --horizon 2 --optimize --starts 8 --budget 15 --trials 10 --mc-samples 512 --function-name ackley10d"
  "--output-dir ../timing-and-allocations --horizon 2 --optimize --starts 8 --budget 15 --trials 10 --mc-samples 1024 --function-name ackley10d"
  # Scenario 2 Horizon 3
  "--output-dir ../timing-and-allocations --horizon 3 --optimize --starts 8 --budget 15 --trials 10 --mc-samples 32 --function-name ackley10d"
  "--output-dir ../timing-and-allocations --horizon 3 --optimize --starts 8 --budget 15 --trials 10 --mc-samples 64 --function-name ackley10d"
  "--output-dir ../timing-and-allocations --horizon 3 --optimize --starts 8 --budget 15 --trials 10 --mc-samples 128 --function-name ackley10d"
  "--output-dir ../timing-and-allocations --horizon 3 --optimize --starts 8 --budget 15 --trials 10 --mc-samples 256 --function-name ackley10d"
  "--output-dir ../timing-and-allocations --horizon 3 --optimize --starts 8 --budget 15 --trials 10 --mc-samples 512 --function-name ackley10d"
  "--output-dir ../timing-and-allocations --horizon 3 --optimize --starts 8 --budget 15 --trials 10 --mc-samples 1024 --function-name ackley10d"
)

for config in "${myopic_configurations[@]}"; do
  julia ../nonmyopic_bayesopt.jl $config
done