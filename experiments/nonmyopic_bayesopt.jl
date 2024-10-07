using ArgParse


function parse_command_line(args)
    parser = ArgParseSettings("Myopic Bayesian Optimization CLI")

    @add_arg_table! parser begin
        "--nworkers"
            action = :store_arg
            help = "Number of workers to use"
            default = Sys.CPU_THREADS
            arg_type = Int
        "--seed"
            action = :store_arg
            help = "Seed for random number generation"
            default = 1906
            arg_type = Int
        "--optimize"
            action = :store_true
            help = "If set, the surrogate's hyperparameters will be optimized"
        "--starts"
            action = :store_arg
            help = "Number of random starts for inner policy optimization (default: 8)"
            default = 16
            arg_type = Int
        "--trials"
            action = :store_arg
            help = "Number of trials with a different initial start (default: 60)"
            default = 60
            arg_type = Int
        "--budget"
            action = :store_arg
            help = "Maximum budget for bayesian optimization (default: 20)"
            default = 15
            arg_type = Int
        "--output-dir"
            action = :store_arg
            help = "Output directory for GAPs and model observations"
            required = true
        "--mc-samples"
            action = :store_arg
            help = "Number of Monte Carlo samples for the acquisition function (default: 50 * Horizon)"
            default = 200
            arg_type = Int
        "--horizon"
            action = :store_arg
            help = "Horizon for the rollout (default: 1)"
            default = 0
            arg_type = Int
        "--batch-size"
            action = :store_arg
            help = "Batch size for the rollout (default: 1)"
            default = 8
            arg_type = Int
        "--function-name"
            action = :store_arg
            help = "Name of the function to optimize"
            required = true
        "--sgd-iterations"
            action = :store_arg
            help = "Number of iterations for SGD (default: 50)"
            default = 50
            arg_type = Int
        "--variance-reduction"
            action = :store_true
            help = "Use EI as a control variate for variance reduction"
        "--deterministic-solve"
            action = :store_true
            help = "Use SAA to solve acquisition function"
    end

    parsed_args = parse_args(args, parser)
    return parsed_args
end


cli_args = parse_command_line(ARGS)


using Distributions
using LinearAlgebra
using Plots
using Sobol
using Random
using CSV
using DataFrames
using Dates
using Distributed
using SharedArrays


# Distributed.addprocs(cli_args["nworkers"])


# @everywhere include("../testfns.jl")
# @everywhere include("../rollout.jl")
# @everywhere include("../utils.jl")
include("../rollout_bayesian_optimization.jl")


function random_solver(s::RBFsurrogate, lbs, ubs; initial_guesses)
    return vec(randsample(1, length(lbs), lbs, ubs)), 0.
end


function write_metadata_to_file(cli_args)
    # Extract the parameters from the command-line arguments
    budget = cli_args["budget"]
    number_of_trials = cli_args["trials"]
    number_of_starts = cli_args["starts"]

    # Get directory for experiment
    final_directory = dirname(@__FILE__) * "/myopic/" * cli_args["function-name"]

    # Define the file path
    file_path = joinpath(final_directory, "metadata.txt")

    # Open the file and write the metadata
    open(file_path, "w") do file
        println(file, "Budget: ", budget)
        println(file, "Number of Trials: ", number_of_trials)
        println(file, "Number of Starts: ", number_of_starts)
    end
end


function main()
    cli_args = parse_command_line(ARGS)
    Random.seed!(cli_args["seed"])
    BUDGET = cli_args["budget"]
    NUMBER_OF_TRIALS = cli_args["trials"]
    NUMBER_OF_STARTS = cli_args["starts"]
    INITIAL_OBSERVATIONS = 5
    # Directory for experimental results experiments/myopic/ackley2d/...
    EXPERIMENT_DIRECTORY = dirname(@__FILE__) * "/myopic/" * cli_args["function-name"]
    # Create the directory structure if it doesn't already exist
    mkpath(EXPERIMENT_DIRECTORY)

    # Establish the synthetic functions we want to evaluate our algorithms on.
    testfn_payloads = Dict(
        "gramacylee" => (name="gramacylee", fn=TestGramacyLee, args=()),
        "rastrigin1d" => (name="rastrigin1d", fn=TestRastrigin, args=(1)),
        "rastrigin4d" => (name="rastrigin4d", fn=TestRastrigin, args=(4)),
        "ackley1d" => (name="ackley1d", fn=TestAckley, args=(1)),
        "ackley2d" => (name="ackley2d", fn=TestAckley, args=(2)),
        "ackley3d" => (name="ackley3d", fn=TestAckley, args=(3)),
        "ackley4d" => (name="ackley4d", fn=TestAckley, args=(4)),
        "ackley5d" => (name="ackley5d", fn=TestAckley, args=(5)),
        "ackley8d" => (name="ackley8d", fn=TestAckley, args=(8)),
        "ackley10d" => (name="ackley10d", fn=TestAckley, args=(10)),
        "ackley16d" => (name="ackley16d", fn=TestAckley, args=(16)),
        "rosenbrock" => (name="rosenbrock", fn=TestRosenbrock, args=()),
        "sixhump" => (name="sixhump", fn=TestSixHump, args=()),
        "braninhoo" => (name="braninhoo", fn=TestBraninHoo, args=()),
        "hartmann3d" => (name="hartmann3d", fn=TestHartmann3D, args=()),
        "goldsteinprice" => (name="goldsteinprice", fn=TestGoldsteinPrice, args=()),
        "beale" => (name="beale", fn=TestBeale, args=()),
        "easom" => (name="easom", fn=TestEasom, args=()),
        "styblinskitang1d" => (name="styblinskitang1d", fn=TestStyblinskiTang, args=(1)),
        "styblinskitang2d" => (name="styblinskitang2d", fn=TestStyblinskiTang, args=(2)),
        "styblinskitang3d" => (name="styblinskitang3d", fn=TestStyblinskiTang, args=(3)),
        "styblinskitang4d" => (name="styblinskitang4d", fn=TestStyblinskiTang, args=(4)),
        "styblinskitang10d" => (name="styblinskitang10d", fn=TestStyblinskiTang, args=(10)),
        "bukinn6" => (name="bukinn6", fn=TestBukinN6, args=()),
        "crossintray" => (name="crossintray", fn=TestCrossInTray, args=()),
        "eggholder" => (name="eggholder", fn=TestEggHolder, args=()),
        "holdertable" => (name="holdertable", fn=TestHolderTable, args=()),
        "schwefel1d" => (name="schwefel1d", fn=TestSchwefel, args=(1)),
        "schwefel2d" => (name="schwefel2d", fn=TestSchwefel, args=(2)),
        "schwefel3d" => (name="schwefel3d", fn=TestSchwefel, args=(3)),
        "schwefel4d" => (name="schwefel4d", fn=TestSchwefel, args=(4)),
        "schwefel10d" => (name="schwefel10d", fn=TestSchwefel, args=(10)),
        "levyn13" => (name="levyn13", fn=TestLevyN13, args=()),
        "trid1d" => (name="trid1d", fn=TestTrid, args=(1)),
        "trid2d" => (name="trid2d", fn=TestTrid, args=(2)),
        "trid3d" => (name="trid3d", fn=TestTrid, args=(3)),
        "trid4d" => (name="trid4d", fn=TestTrid, args=(4)),
        "trid10d" => (name="trid10d", fn=TestTrid, args=(10)),
        "mccormick" => (name="mccormick", fn=TestMccormick, args=()),
        "hartmann6d" => (name="hartmann6d", fn=TestHartmann6D, args=()),
        "hartmann4d" => (name="hartmann4d", fn=TestHartmann4D, args=()),
        "bohachevsky" => (name="bohachevsky", fn=TestBohachevsky, args=()),
        "griewank3d" => (name="griewank3d", fn=TestGriewank, args=(3)),
        "shekel4d" => (name="shekel4d", fn=TestShekel, args=()),
        "dropwave" => (name="dropwave", fn=TestDropWave, args=()),
        "griewank1d" => (name="griewank1d", fn=TestGriewank, args=(1)),
        "griewank2d" => (name="griewank1d", fn=TestGriewank, args=(2)),
        "levy10d" => (name="levy10d", fn=TestLevy, args=(10)),
    )
    
    # Build the test function object
    payload = testfn_payloads[cli_args["function-name"]]
    println("Running experiment for $(payload.name)...")
    testfn = payload.fn(payload.args...)
    lbs, ubs = testfn.bounds[:,1], testfn.bounds[:,2]
    # Indices for running test on specific acquisitions
    strategy_indices = [1, 2, 3, 4]


    metrics_to_collect = ["times", "gaps", "allocations", "simple_regret", "minimum_observations"]
    acquisitions = ["ei", "poi", "lcb", "random"]
    dr_hypers = [[0.], [0.], [2.], [0.]]
    decision_rules = [EI(), POI(), LCB(), RandomAcquisition()]
    # Allocate space for timing information
    timing_statistics = [zeros(BUDGET) for acq in acquisitions]
    # Create the CSV for the current test function being evaluated
    # Create the CSV for the current test function being evaluated
    # Create the CSV for the simple regrets
    # Create the CSV for the allocations
    # Create the CSV for storing the minimum observation found at each iteration
    for metric in metrics_to_collect
        [create_csv(EXPERIMENT_DIRECTORY * "/$(acq)_$(metric)", BUDGET) for acq in acquisitions]
    end

    # Write the metadata to disk
    write_metadata_to_file(cli_args)

    # Initial guesses for inner optimization
    inner_solve_xstarts = generate_initial_guesses(NUMBER_OF_STARTS, lbs, ubs)
    initial_samples = [lbs .+ (ubs .- lbs) .* rand(testfn.dim, INITIAL_OBSERVATIONS) for _ in 1:NUMBER_OF_TRIALS]

    # Preallocated variables for measurements we care about saving: time, gaps, simple_regret, allocations
    gaps = zeros(BUDGET)
    simple_regrets = zeros(BUDGET)
    allocations = zeros(BUDGET)
    time_elapsed = zeros(BUDGET)
    minimum_observations = zeros(BUDGET)
    metrics_to_collect_container = [time_elapsed, gaps, allocations, simple_regrets, minimum_observations]

    xnext = zeros(testfn.dim)
    σn2 = 1e-6
    kernel = Matern52()
    kernel_lbs, kernel_ubs = [0.1], [5.]
    Xinit = zeros(testfn.dim, INITIAL_OBSERVATIONS)
    true_minimum = testfn.f(testfn.xopt[1])

    # Preallocate entire surrogate object and reuse
    sur = Surrogate(kernel, zeros(testfn.dim, 1), [0.]; capacity=BUDGET, σn2=σn2)

    for (acq_index, decision_rule) in enumerate(decision_rules)
        acq_name = acquisitions[acq_index]
        set_decision_rule!(sur, decision_rule)

        println("Conduction Experiments with Acquisition = ", decision_rule.name)
        for trial in 1:NUMBER_OF_TRIALS
            println("($(payload.name)) Trial $(trial) of $(NUMBER_OF_TRIALS)...")
            # Initialize surrogate model
            Xinit[:, :] = initial_samples[trial]
            yinit = testfn.f.(eachcol(Xinit))
            reset!(sur, Xinit, yinit)
            initial_best = minimum(yinit)

            # Perform Bayesian optimization iterations
            print("Budget Counter: ")
            for budget in 1:BUDGET
                # Solve the acquisition function
                output = @timed begin
                    multistart_base_solve!(
                        sur,
                        xnext,
                        spatial_lbs=lbs,
                        spatial_ubs=ubs,
                        guesses=inner_solve_xstarts,
                        θfixed=dr_hypers[acq_index]
                    )
                end
                time_elapsed[budget] = output.time
                allocations[budget] = output.bytes
                simple_regrets[budget] = simple_regret(
                    true_minimum,
                    minimum(get_active_observations(sur))
                )
                gaps[budget] = gap(
                    initial_best,
                    minimum(get_active_observations(sur)),
                    true_minimum
                )

                ynext = testfn.f(xnext)
                condition!(sur, xnext, ynext)
                optimize!(sur, lowerbounds=kernel_lbs, upperbounds=kernel_ubs)

                minimum_observations[budget] = minimum(get_active_observations(sur))
                print("|")
            end

            # A full trial of BO has completed, so save results
            println()
            for (metric_index, metric) in enumerate(metrics_to_collect)
                write_to_csv(
                    EXPERIMENT_DIRECTORY * "/$(acq_name)_$(metric)",
                    metrics_to_collect_container[metric_index]
            )
            end
        end
    end
end

main()