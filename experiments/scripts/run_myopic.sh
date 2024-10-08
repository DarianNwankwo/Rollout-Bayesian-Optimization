myopic_configurations=(
    # Configuration short long run experiments
    # "--function-name ackley5d --budget 100 --starts 64 --trials 60"
    # "--function-name braninhoo --budget 100 --starts 64 --trials 60"
    # "--function-name hartmann6d --budget 100 --starts 64 --trials 60"
    # "--function-name sixhump --budget 100 --starts 64 --trials 60"
    # "--function-name levy10d --budget 100 --starts 64 --trials 60"
    # "--function-name goldsteinprice --budget 100 --starts 64 --trials 60"
    "--function-name griewank3d --budget 100 --starts 64 --trials 60"
)

for config in "${myopic_configurations[@]}"; do
  julia ../myopic_bayesopt.jl $config
done