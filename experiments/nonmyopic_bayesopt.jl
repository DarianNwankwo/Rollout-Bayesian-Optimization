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
            default = 8
            arg_type = Int
        "--trials"
            action = :store_arg
            help = "Number of trials with a different initial start (default: 60)"
            default = 60
            arg_type = Int
        "--budget"
            action = :store_arg
            help = "Maximum budget for bayesian optimization (default: 20)"
            default = 20
            arg_type = Int
        "--output-dir"
            action = :store_arg
            help = "Output directory for GAPs and model observations"
            required = true
        "--mc-samples"
            action = :store_arg
            help = "Number of Monte Carlo samples for the acquisition function (default: 50 * Horizon)"
            default = 50
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
using SharedArrays
using Distributed


Distributed.addprocs(cli_args["nworkers"])


@everywhere include("../testfns.jl")
@everywhere include("../rollout.jl")
@everywhere include("../utils.jl")


function create_time_csv_file(
    parent_directory::String,
    child_directory::String,
    csv_filename::String,
    budget::Int
    )
    # Create directory for finished experiment
    self_filename, extension = splitext(basename(@__FILE__))
    dir_name = parent_directory * "/" * self_filename * "/" * child_directory
    mkpath(dir_name)

    # Write the header to the csv file
    path_to_csv_file = dir_name * "/" * csv_filename
    col_names = vcat(["trial"], ["$i" for i in 1:budget])

    CSV.write(
        path_to_csv_file,
        DataFrame(
            -ones(1, budget + 1),
            Symbol.(col_names)
        )    
    )

    return path_to_csv_file
end


function write_time_to_csv(
    times::Vector{T},
    trial_number::Int,
    path_to_csv_file::String
    ) where T <: Number
    # Write gap to csv
    CSV.write(
        path_to_csv_file,
        Tables.table(
            hcat([trial_number times'])
        ),
        append=true,
    )

    return nothing
end


function create_gap_csv_file(
    parent_directory::String,
    child_directory::String,
    csv_filename::String,
    budget::Int
    )
    # Create directory for finished experiment
    self_filename, extension = splitext(basename(@__FILE__))
    dir_name = parent_directory * "/" * self_filename * "/" * child_directory
    mkpath(dir_name)

    # Write the header to the csv file
    path_to_csv_file = dir_name * "/" * csv_filename
    col_names = vcat(["trial"], ["$i" for i in 0:budget])

    CSV.write(
        path_to_csv_file,
        DataFrame(
            -ones(1, budget + 2),
            Symbol.(col_names)
        )    
    )

    return path_to_csv_file
end


function create_allocation_csv_file(
    parent_directory::String,
    child_directory::String,
    csv_filename::String,
    budget::Int
    )
    # Create directory for finished experiment
    self_filename, extension = splitext(basename(@__FILE__))
    dir_name = parent_directory * "/" * self_filename * "/" * child_directory
    mkpath(dir_name)

    # Write the header to the csv file
    path_to_csv_file = dir_name * "/" * csv_filename
    col_names = vcat(["trial"], ["$i" for i in 1:budget])

    CSV.write(
        path_to_csv_file,
        DataFrame(
            -ones(1, budget + 1),
            Symbol.(col_names)
        )    
    )

    return path_to_csv_file
end

function write_allocations_to_csv(
    allocations::Vector{T},
    trial_number::Int,
    path_to_csv_file::String
    ) where T <: Number
    # Write gap to csv
    CSV.write(
        path_to_csv_file,
        Tables.table(
            hcat([trial_number allocations'])
        ),
        append=true,
    )

    return nothing
end


function write_metadata_to_file(cli_args)
    # Extract the parameters from the command-line arguments
    budget = cli_args["budget"]
    number_of_trials = cli_args["trials"]
    number_of_starts = cli_args["starts"]
    data_directory = cli_args["output-dir"]
    should_optimize = haskey(cli_args, "optimize") ? cli_args["optimize"] : false
    horizon = cli_args["horizon"]
    mc_samples = cli_args["mc-samples"] * (horizon + 1)
    batch_size = cli_args["batch-size"]
    sgd_iterations = cli_args["sgd-iterations"]
    should_reduce_variance = haskey(cli_args, "variance-reduction") ? cli_args["variance-reduction"] : false

    # Get directory for experiment
    self_filename, extension = splitext(basename(@__FILE__))
    final_directory = data_directory * "/" * self_filename

    # Define the file path
    file_path = joinpath(final_directory, "metadata.txt")

    # Open the file and write the metadata
    open(file_path, "w") do file
        println(file, "Budget: ", budget)
        println(file, "Number of Trials: ", number_of_trials)
        println(file, "Number of Starts: ", number_of_starts)
        println(file, "Data Directory: ", data_directory)
        println(file, "Should Optimize: ", should_optimize)
        println(file, "Horizon: ", horizon)
        println(file, "MC Samples: ", mc_samples)
        println(file, "Batch Size: ", batch_size)
        println(file, "SGD Iterations: ", sgd_iterations)
        println(file, "Should Reduce Variance: ", should_reduce_variance)
    end
end


function create_observation_csv_file(
    parent_directory::String,
    child_directory::String,
    csv_filename::String,
    budget::Int
    )
    # Get directory for experiment
    self_filename, extension = splitext(basename(@__FILE__))
    dir_name = parent_directory * "/" * self_filename * "/" * child_directory
    
    # Write the header to the csv file
    path_to_csv_file = dir_name * "/" * csv_filename
    col_names = vcat(["trial"], ["observation_pair_$i" for i in 1:budget])

    CSV.write(
        path_to_csv_file,
        DataFrame(
            -ones(1, budget + 1),
            Symbol.(col_names)
        )    
    )
    
    return path_to_csv_file
end


function write_gap_to_csv(
    gaps::Vector{T},
    trial_number::Int,
    path_to_csv_file::String
    ) where T <: Number
    # Write gap to csv
    CSV.write(
        path_to_csv_file,
        Tables.table(
            hcat([trial_number gaps'])
        ),
        append=true,
    )

    return nothing
end


function write_observations_to_csv(
    X::Matrix{T},
    y::Vector{T},
    trial_number::Int,
    path_to_csv_file::String
    ) where T <: Number
    # Write observations to csv
    d, N = size(X)
    X = hcat(trial_number * ones(d, 1), X)
    X = vcat(X, [trial_number y'])
    
    CSV.write(
        path_to_csv_file,
        Tables.table(X),
        append=true,
    )

    return nothing
end


function generate_initial_guesses(N::Int, lbs::Vector{T}, ubs::Vector{T},) where T <: Number
    ϵ = 1e-6
    seq = SobolSeq(lbs, ubs)
    initial_guesses = reduce(hcat, next!(seq) for i = 1:N)
    initial_guesses = hcat(initial_guesses, lbs .+ ϵ)
    initial_guesses = hcat(initial_guesses, ubs .- ϵ)

    return initial_guesses
end


function write_error_to_disk(filename::String, msg::String)
    # Open a text file in write mode
    open(filename, "w+") do file
        # Write a string to the file
        write(file, msg)
    end
end


function main(cli_args)
    Random.seed!(cli_args["seed"])
    BUDGET = cli_args["budget"]
    NUMBER_OF_TRIALS = cli_args["trials"]
    NUMBER_OF_STARTS = cli_args["starts"]
    DATA_DIRECTORY = cli_args["output-dir"]
    SHOULD_OPTIMIZE = haskey(cli_args, "optimize") ? cli_args["optimize"] : false
    HORIZON = cli_args["horizon"]
    MC_SAMPLES = cli_args["mc-samples"] # * (HORIZON + 1)
    BATCH_SIZE = cli_args["batch-size"]
    SGD_ITERATIONS = cli_args["sgd-iterations"]
    SHOULD_REDUCE_VARIANCE = haskey(cli_args, "variance-reduction") ? cli_args["variance-reduction"] : false
    FUNCTION_NAME = cli_args["function-name"]

    # Establish the synthetic functions we want to evaluate our algorithms on.
    testfn_payloads = (
        (name="gramacylee", fn=TestGramacyLee, args=()),
        (name="rastrigin", fn=TestRastrigin, args=(1)),
        (name="rastrigin4d", fn=TestRastrigin, args=(4)),
        (name="ackley1d", fn=TestAckley, args=(1)),
        (name="ackley2d", fn=TestAckley, args=(2)),
        (name="ackley3d", fn=TestAckley, args=(3)),
        (name="ackley4d", fn=TestAckley, args=(4)),
        (name="ackley8d", fn=TestAckley, args=(8)),
        (name="ackley16d", fn=TestAckley, args=(16)),
        (name="ackley10d", fn=TestAckley, args=(10)),
        (name="rosenbrock", fn=TestRosenbrock, args=()),
        (name="sixhump", fn=TestSixHump, args=()),
        (name="braninhoo", fn=TestBraninHoo, args=()),
        (name="hartmann3d", fn=TestHartmann3D, args=()),
        (name="goldsteinprice", fn=TestGoldsteinPrice, args=()),
        (name="beale", fn=TestBeale, args=()),
        (name="easom", fn=TestEasom, args=()),
        (name="styblinskitang1d", fn=TestStyblinskiTang, args=(1)),
        (name="styblinskitang2d", fn=TestStyblinskiTang, args=(2)),
        (name="styblinskitang3d", fn=TestStyblinskiTang, args=(3)),
        (name="styblinskitang4d", fn=TestStyblinskiTang, args=(4)),
        (name="styblinskitang10d", fn=TestStyblinskiTang, args=(10)),
        (name="bukinn6", fn=TestBukinN6, args=()),
        (name="crossintray", fn=TestCrossInTray, args=()),
        (name="eggholder", fn=TestEggHolder, args=()),
        (name="holdertable", fn=TestHolderTable, args=()),
        (name="schwefel1d", fn=TestSchwefel, args=(1)),
        (name="schwefel2d", fn=TestSchwefel, args=(2)),
        (name="schwefel3d", fn=TestSchwefel, args=(3)),
        (name="schwefel4d", fn=TestSchwefel, args=(4)),
        (name="schwefel10d", fn=TestSchwefel, args=(10)),
        (name="levyn13", fn=TestLevyN13, args=()),
        (name="trid1d", fn=TestTrid, args=(1)),
        (name="trid2d", fn=TestTrid, args=(2)),
        (name="trid3d", fn=TestTrid, args=(3)),
        (name="trid4d", fn=TestTrid, args=(4)),
        (name="trid10d", fn=TestTrid, args=(10)),
        (name="mccormick", fn=TestMccormick, args=()),
        (name="hartmann6d", fn=TestHartmann6D, args=()),
        (name="hartmann4d", fn=TestHartmann4D, args=()),
        (name="hartmann3d", fn=TestHartmann3D, args=()),
        (name="bohachevsky", fn=TestBohachevsky, args=()),
        (name="griewank3d", fn=TestGriewank, args=(3)),
        (name="shekel4d", fn=TestShekel, args=()),
        (name="dropwave", fn=TestDropWave, args=()),
        (name="griewank1d", fn=TestGriewank, args=(1)),
        (name="griewank1d", fn=TestGriewank, args=(2)),
    )

    # Gaussian process hyperparameters
    θ, σn2 = [1.], 1e-6
    ψ = kernel_matern52(θ)

    println("Running experiment for $(FUNCTION_NAME).")
    println("Configuration: ")
    println("    Budget: $(BUDGET)")
    println("    Number of trials: $(NUMBER_OF_TRIALS)")
    println("    Number of starts: $(NUMBER_OF_STARTS)")
    println("    Output directory: $(DATA_DIRECTORY)")
    println("    Optimize hyperparameters: $(SHOULD_OPTIMIZE)")
    println("    Horizon: $(HORIZON)")
    println("    Monte Carlo samples: $(MC_SAMPLES)")
    println("    Batch size: $(BATCH_SIZE)")
    println("    SGD iterations: $(SGD_ITERATIONS)")
    println("    Variance reduction: $(SHOULD_REDUCE_VARIANCE)")
    
    # Build the test function object
    payload = first([p for p in testfn_payloads if p.name == FUNCTION_NAME])
    testfn = payload.fn(payload.args...)
    lbs, ubs = testfn.bounds[:,1], testfn.bounds[:,2]

    # Generate low discrepancy sequence
    lds_rns = gen_low_discrepancy_sequence(MC_SAMPLES, testfn.dim, HORIZON + 1)

    # Allocate initial guesses for optimizer
    initial_guesses = generate_initial_guesses(NUMBER_OF_STARTS, lbs, ubs)

    # Allocate all initial samples
    initial_samples = randsample(NUMBER_OF_TRIALS, testfn.dim, lbs, ubs)

    # Allocate space for GAPS
    rollout_gaps = zeros(BUDGET + 1)

    # Allocate space for times
    rollout_times = zeros(BUDGET)

    # Allocate space for memory allocations
    rollout_allocations = zeros(BUDGET)

    # Create the CSV for the current test function being evaluated
    rollout_csv_file_path = create_gap_csv_file(
        DATA_DIRECTORY, payload.name, "rollout_h$(HORIZON)_gaps.csv", BUDGET
    )

    # Create the CSV for the current test function being evaluated observations
    rollout_observation_csv_file_path = create_observation_csv_file(
        DATA_DIRECTORY, payload.name, "rollout_h$(HORIZON)_observations.csv", BUDGET
    )

    # Create the CSV for the current test function being evaluated for timing
    rollout_time_file_path = create_time_csv_file(
        DATA_DIRECTORY, payload.name, "rollout_h$(HORIZON)_times.csv", BUDGET
    )

    # Create the CSV for the current test function being evaluated for memory allocations
    rollout_allocations_file_path = create_allocation_csv_file(
        DATA_DIRECTORY, payload.name, "rollout_h$(HORIZON)_allocations.csv", BUDGET
    )
    
    # Write the metadata to disk
    write_metadata_to_file(cli_args)

    # Initialize the trajectory parameters
    tp = TrajectoryParameters(
        initial_samples[:, 1], # Will be overriden later
        HORIZON,
        MC_SAMPLES,
        lds_rns,
        lbs,
        ubs,
    )

    # Initialize batch of points to evaluate the rollout acquisition function
    batch = generate_batch(BATCH_SIZE, lbs=tp.lbs, ubs=tp.ubs)

    # Initialize shared memory for solving base policy in parallel
    candidate_locations = SharedMatrix{Float64}(testfn.dim, NUMBER_OF_STARTS)
    candidate_values = SharedArray{Float64}(NUMBER_OF_STARTS)
    αxs = SharedArray{Float64}(tp.mc_iters)
    ∇αxs = SharedMatrix{Float64}(testfn.dim, tp.mc_iters)
    final_locations = SharedMatrix{Float64}(length(tp.x0), size(batch, 2))
    final_evaluations = SharedArray{Float64}(size(batch, 2))

    for trial in 1:NUMBER_OF_TRIALS
        try
            println("($(payload.name)) Trial $(trial) of $(NUMBER_OF_TRIALS)...")
            # Initialize surrogate model
            Xinit = initial_samples[:, trial:trial]
            yinit = testfn.f.(eachcol(Xinit))
            sur = fit_surrogate(ψ, Xinit, yinit; σn2=σn2)

            # Perform Bayesian optimization iterations
            print("Budget Counter: ")
            for budget in 1:BUDGET
                # Truncate the horizon as we approach the end of our budget
                tp.h = min(HORIZON, BUDGET - budget)

                # Solve the acquisition function
                timed_outcome = @timed begin
                xbest, fbest = distributed_rollout_solver(
                    sur=sur, tp=tp, xstarts=initial_guesses, batch=batch, max_iterations=SGD_ITERATIONS,
                    candidate_locations=candidate_locations, candidate_values=candidate_values,
                    αxs=αxs, ∇αxs=∇αxs, final_locations=final_locations, final_evaluations=final_evaluations,
                    varred=SHOULD_REDUCE_VARIANCE
                )
                end
                ybest = testfn.f(xbest)
                # Update the surrogate model
                sur = update_surrogate(sur, xbest, ybest)
                rollout_times[budget] = timed_outcome.time
                rollout_allocations[budget] = timed_outcome.bytes

                if SHOULD_OPTIMIZE
                    sur = optimize_hypers_optim(sur, kernel_matern52)
                end
                print("|")
            end
            println()
            
            # Compute the GAP of the surrogate model
            fbest = testfn.f(testfn.xopt[1])
            rollout_gaps[:] .= measure_gap(get_observations(sur), fbest)

            write_time_to_csv(rollout_times, trial, rollout_time_file_path)
            write_gap_to_csv(rollout_gaps, trial, rollout_csv_file_path)
            write_observations_to_csv(sur.X, get_observations(sur), trial, rollout_observation_csv_file_path)
            write_allocations_to_csv(rollout_allocations, trial, rollout_allocations_file_path)
        catch failure_error
            msg = "($(payload.name)) Trial $(trial) failed with error: $(failure_error)"
            self_filename, extension = splitext(basename(@__FILE__))
            filename = DATA_DIRECTORY * "/" * self_filename * "/" * payload.name * "_failed.txt"
            write_error_to_disk(filename, msg)
        end
    end

    println("Completed")
end

main(cli_args)