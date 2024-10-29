using Logging: @warn
using Parameters, KrylovKit, TensorOperations, Statistics
using LinearAlgebra: LinearAlgebra, Diagonal, dot, norm, tr, mul!, axpy!, normalize!, normalize, Symmetric, eigen
# import LinearAlgebra

module Defaults
	const maxiter = 100
	const D = 50
	const tolgauge = 1e-14
	const tol = 1e-12
	const verbosity = 1
	import KrylovKit: GMRES
	const solver = GMRES(tol=1e-12, maxiter=100)
end

# auxiliary
include("auxiliary/coeff.jl")
include("auxiliary/distance.jl")
include("auxiliary/truncation.jl")
include("auxiliary/deparallelise.jl")
include("auxiliary/tensorops.jl")
include("auxiliary/factorize.jl")
include("auxiliary/simplelanczos.jl")

# mps
include("states/abstractmps.jl")
include("states/transfer.jl")
include("states/bondview.jl")
include("states/finitemps.jl")
include("states/density_operator.jl")
include("states/orth.jl")
include("states/initializers.jl")
include("states/arithmetics.jl")

# mpo
include("mpo/abstractmpo.jl")
include("mpo/transfer.jl")
include("mpo/finitempo.jl")
include("mpo/compress.jl")
include("mpo/orth.jl")
include("mpo/initializers.jl")
include("mpo/arithmetics.jl")

# environments
include("envs/abstractdefs.jl")
include("envs/finiteenv.jl")
include("envs/overlap.jl")

# circuit for TEBD
include("circuit/abstractdefs.jl")
include("circuit/gate.jl")
include("circuit/circuit.jl")
include("circuit/apply_gates_no_to.jl")
include("circuit/apply_gates.jl")
include("circuit/gate_fusion.jl")


# operators
include("operators/abstractdefs.jl")
include("operators/qterm.jl")
include("operators/superterm.jl")
include("operators/abstractoperator.jl")
include("operators/quantumoperator.jl")
include("operators/superoperator.jl")
include("operators/expecs.jl")
include("operators/tompo.jl")
include("operators/arithmetics.jl")

# algorithms
include("algorithms/tebd.jl")
include("algorithms/derivatives.jl")
include("algorithms/groundstate.jl")
include("algorithms/excitedstates.jl")
include("algorithms/tdvp.jl")
include("algorithms/timeevo.jl")
include("algorithms/thermalstate.jl")
include("algorithms/twotimecorr.jl")
# iterative mpo mps algorithms
include("algorithms/iterativeariths/def.jl")
include("algorithms/iterativeariths/mpompsmult/mpompsmult.jl")
include("algorithms/iterativeariths/mpsadd/mpsadd.jl")
include("algorithms/iterativeariths/mpompomult/mpompomult.jl")
include("algorithms/iterativeariths/compression/compression.jl")

# # quantum circuit simulator
# include("quantumcircuitsimulator/interface/interface.jl")
# include("quantumcircuitsimulator/backends/mps.jl")
# include("quantumcircuitsimulator/algorithm/qft.jl")

# utilities
include("utilities/spin_siteops.jl")
include("utilities/models.jl")