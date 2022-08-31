
include("abstractadd.jl")
include("svdadd.jl")
include("iterativeadd.jl")

_add(mpsxs::Vector{<:MPS}, alg::OneSiteIterativeAdd) = iterative_add(mpsxs, alg)
_add(mpsxs::Vector{<:MPS}, alg::SVDAdd) = svd_add(mpsxs, alg)

mpsadd(mpsxs::Vector{<:MPS}, alg::OneSiteIterativeAdd) = _add(mpsxs, alg)
mpsadd(psxs::Vector{<:MPS}; alg::AbstractMPSAdd=IterativeAdd()) = add(mpsxs, alg)

