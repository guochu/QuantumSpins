
include("abstractadd.jl")
include("svdadd.jl")
include("iterativeadd.jl")

_add(mpsxs::Vector{<:MPS}, alg::OneSiteIterativeArith) = iterative_add(mpsxs, alg)
_add(mpsxs::Vector{<:MPS}, alg::SVDArith) = svd_add(mpsxs, alg)

mpsadd(mpsxs::Vector{<:MPS}, alg::OneSiteIterativeArith) = _add(mpsxs, alg)
mpsadd(psxs::Vector{<:MPS}; alg::AbstractMPSArith=IterativeArith()) = add(mpsxs, alg)

