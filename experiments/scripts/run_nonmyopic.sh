nonmyopic_configurations=(
    "--function-name ackley5d --budget 100 --starts 64 --trials 60 --horizon 0 --mc-samples 5"
    "--function-name braninhoo --budget 100 --starts 64 --trials 60 --horizon 0 --mc-samples 5"
    "--function-name hartmann6d --budget 100 --starts 64 --trials 60 --horizon 0 --mc-samples 5"
    "--function-name sixhump --budget 100 --starts 64 --trials 60 --horizon 0 --mc-samples 5"
    "--function-name levy10d --budget 100 --starts 64 --trials 60 --horizon 0 --mc-samples 5"
    "--function-name goldsteinprice --budget 100 --starts 64 --trials 60 --horizon 0 --mc-samples 5"
    "--function-name griewank3d --budget 100 --starts 64 --trials 60 --horizon 0 --mc-samples 5"
)

for config in "${nonmyopic_configurations[@]}"; do
  julia ../nonmyopic_bayesopt.jl $config
done