include("svdadd.jl")
include("iterativeadd.jl")
include("stableadd.jl")

_add(mpsxs::Vector{<:MPS}, alg::OneSiteIterativeArith) = iterative_add(mpsxs, alg)
_add(mpsxs::Vector{<:MPS}, alg::SVDArith) = svd_add(mpsxs, alg)
_add(mpsxs::Vector{<:MPS}, alg::OneSiteStableArith) = stable_add(mpsxs, alg)

mpsadd(mpsxs::Vector{<:MPS}, alg::AbstractMPSArith) = _add(mpsxs, alg)
mpsadd(mpsxs::Vector{<:MPS}; alg::AbstractMPSArith=IterativeArith()) = _add(mpsxs, alg)

