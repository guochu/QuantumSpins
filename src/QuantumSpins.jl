module QuantumSpins

using Logging: @warn
using SparseArrays, Parameters, KrylovKit, TensorOperations, Statistics
using LinearAlgebra: Diagonal, dot, norm, tr, mul!, normalize!, normalize
import LinearAlgebra

# auxiliary
export contract, Coefficient, value, scalar_type, is_constant, NoTruncation, TruncateCutoff, TruncateDim, MPSTruncation, TruncationScheme
export permute, tie, deparallelise, tsvd!, texp, tqr!, tlq!, entropy, renyi_entropy
export dot, norm, tr, normalize!, normalize

# mps
export AbstractMPS, MPS, iscanonical, canonicalize!, bond_dimension, bond_dimensions, distance2, distance, increase_bond!
export physical_dimensions, DensityOperatorMPS, DensityOperator, infinite_temperature_state, prodmps, randommps

# mpo
export AbstractMPO, MPO, prodmpo, randommpo, id, expectation, svdcompress!
export MPOCompression, SVDCompression, Deparallelise, compress!

# circuit
export QuantumGate, QuantumCircuit, apply!, positions, shift, fuse_gates

# operators, easier interface for building quantum operators incrementally, and used for TEBD. Should it really be here in this package?
export QTerm, QuantumOperator, matrix, add!, qterms, superoperator, add_unitary!, add_dissipation!

# algorithms
export trotter_propagator, environments, DMRG, TDVP, sweep!, ground_state!
# time evolve stepper
export timeevo!, AbstractStepper, TEBDStepper, TDVPStepper, change_tspan_dt, TEBDCache, TDVPCache, timeevo_cache, correlation_2op_1t, exact_correlation_2op_1t
export mixed_thermalize, thermal_state, itimeevo!

export AbstractMPOMPSMult, IterativeMult, StableMult, SVDMult, iterative_mult, svd_mult, stable_mult, mpompsmult
export AbstractMPSAdd, iterative_add, svd_add, IterativeAdd, SVDAdd, mpsadd

# # quantum circuit simulator
# export ZERO, ONE, X, Y, Z, S, H, sqrtX, sqrtY, T, Rx, Ry, Rz, CONTROL, CZ, CNOT, CX, SWAP, iSWAP
# export CONTROLCONTROL, TOFFOLI, CCX, UP, DOWN, FSIM, GFSIM, PHASE
# export XGate, YGate, ZGate, SGate, SqrtXGate, SqrtYGate, HGate, TGate, RxGate, RyGate, RzGate, CZGate, PHASEGate
# export CNOTGate, SWAPGate, iSWAPGate, CRxGate, CRyGate, CRzGate, TOFFOLIGate, FREDKINGate
# export CONTROLGate, CPHASEGate, CCPHASEGate, CONTROLCONTROLGate, FSIMGate, GFSIMGate
# export from_external, Gate, QMeasure, measure!, amplitude, statevector_mps
# export QFT

# utilities
export spin_half_matrices, ising_chain, heisenberg_chain, boundary_driven_xxz

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

# mps
include("mps/abstractdefs.jl")
include("mps/transfer.jl")
include("mps/bondview.jl")
include("mps/finitemps.jl")
include("mps/density_operator.jl")
include("mps/orth.jl")
include("mps/initializers.jl")
include("mps/arithmetics.jl")

# mpo
include("mpo/abstractdefs.jl")
include("mpo/transfer.jl")
include("mpo/finitempo.jl")
include("mpo/compress.jl")
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
include("operators/qterm.jl")
include("operators/quantumoperator.jl")
include("operators/superoperator.jl")
include("operators/expecs.jl")
include("operators/tompo.jl")

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
include("algorithms/mpompsmult/mpompsmult.jl")
include("algorithms/mpsadd/mpsadd.jl")

# # quantum circuit simulator
# include("quantumcircuitsimulator/interface/interface.jl")
# include("quantumcircuitsimulator/backends/mps.jl")
# include("quantumcircuitsimulator/algorithm/qft.jl")

# utilities
include("utilities/spin_siteops.jl")
include("utilities/models.jl")

end