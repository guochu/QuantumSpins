push!(LOAD_PATH, dirname(Base.@__DIR__) * "/src")

using Test
using QuantumSpins
using KrylovKit
using TensorOperations
using LinearAlgebra: Diagonal



include("check_tensorops.jl")

include("check_mpsops.jl")

include("check_mpsalgs.jl")

include("check_gs.jl")

include("check_timeevo.jl")

include("check_twotimecorr.jl")

