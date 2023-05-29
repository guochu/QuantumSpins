push!(LOAD_PATH, dirname(Base.@__DIR__) * "/src")

using Test
using QuantumSpins
using KrylovKit
using TensorOperations
using LinearAlgebra: Diagonal

const QS = QuantumSpins

include("auxiliary.jl")
include("tensorops.jl")
include("mps.jl")

## algorithms
include("algorithm/iterativealgs.jl")
include("algorithm/groundstate.jl")
include("algorithm/timeevo.jl")
include("algorithm/twotimecorr.jl")

