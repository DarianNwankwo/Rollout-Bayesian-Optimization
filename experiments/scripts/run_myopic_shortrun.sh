myopic_configurations=(
    # Configuration short long run experiments
    "--output-dir=../myopic-shortrun-timing-with-kg --optimize --function-name=gramacylee"
    "--output-dir=../myopic-shortrun-timing-with-kg --optimize --function-name=rosenbrock"
    "--output-dir=../myopic-shortrun-timing-with-kg --optimize --function-name=sixhump"
    "--output-dir=../myopic-shortrun-timing-with-kg --optimize --function-name=braninhoo"
    "--output-dir=../myopic-shortrun-timing-with-kg --optimize --function-name=goldsteinprice"
    "--output-dir=../myopic-shortrun-timing-with-kg --optimize --function-name=schwefel4d"
)

for config in "${myopic_configurations[@]}"; do
  julia ../myopic_bayesopt.jl $config
done