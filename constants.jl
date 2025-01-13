const RANDOM_ACQUISITION = "Random"

"""
When recovering previous solves associated with our surrogate, -1 corresponds to
our ground truth.
"""
const GROUND_TRUTH_OBSERVATIONS = -1

"""
The amount of space to allocate for our surrogate model in terms of the number of 
observations.
"""
const DEFAULT_CAPACITY = 100