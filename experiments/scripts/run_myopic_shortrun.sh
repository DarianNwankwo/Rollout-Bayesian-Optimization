myopic_configurations=(
    # Configuration short long run experiments
    "--output-dir=../data-for-various-horizons-with-kg --optimize --function-name=ackley4d --budget 15 --nworkers 8 --starts 8 --trials 60"
    "--output-dir=../data-for-various-horizons-with-kg --optimize --function-name=braninhoo --budget 15 --nworkers 8 --starts 8 --trials 60"
    "--output-dir=../data-for-various-horizons-with-kg --optimize --function-name=bukinn6 --budget 15 --nworkers 8 --starts 8 --trials 60"
    "--output-dir=../data-for-various-horizons-with-kg --optimize --function-name=eggholder --budget 15 --nworkers 8 --starts 8 --trials 60"
    "--output-dir=../data-for-various-horizons-with-kg --optimize --function-name=goldsteinprice --budget 15 --nworkers 8 --starts 8 --trials 60"
    "--output-dir=../data-for-various-horizons-with-kg --optimize --function-name=gramacylee --budget 15 --nworkers 8 --starts 8 --trials 60"
    "--output-dir=../data-for-various-horizons-with-kg --optimize --function-name=holdertable --budget 15 --nworkers 8 --starts 8 --trials 60"
    "--output-dir=../data-for-various-horizons-with-kg --optimize --function-name=levyn13 --budget 15 --nworkers 8 --starts 8 --trials 60"
    "--output-dir=../data-for-various-horizons-with-kg --optimize --function-name=rastrigin4d --budget 15 --nworkers 8 --starts 8 --trials 60"
    "--output-dir=../data-for-various-horizons-with-kg --optimize --function-name=rosenbrock --budget 15 --nworkers 8 --starts 8 --trials 60"
    "--output-dir=../data-for-various-horizons-with-kg --optimize --function-name=schwefel3d --budget 15 --nworkers 8 --starts 8 --trials 60"
    "--output-dir=../data-for-various-horizons-with-kg --optimize --function-name=schwefel4d --budget 15 --nworkers 8 --starts 8 --trials 60"
    "--output-dir=../data-for-various-horizons-with-kg --optimize --function-name=sixhump --budget 15 --nworkers 8 --starts 8 --trials 60"
    "--output-dir=../data-for-various-horizons-with-kg --optimize --function-name=trid1d --budget 15 --nworkers 8 --starts 8 --trials 60"
    "--output-dir=../data-for-various-horizons-with-kg --optimize --function-name=trid4d --budget 15 --nworkers 8 --starts 8 --trials 60"
)

for config in "${myopic_configurations[@]}"; do
  julia ../myopic_bayesopt.jl $config
done