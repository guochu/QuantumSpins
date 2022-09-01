
include("abstractmult.jl")
include("iterativemult.jl")
include("svdmult.jl")
include("stablemult.jl")


_mult(mpo::AbstractMPO, mps::AbstractMPS, alg::OneSiteIterativeArith) = iterative_mult(mpo, mps, alg)
_mult(mpo::AbstractMPO, mps::AbstractMPS, alg::SVDArith) = svd_mult(mpo, mps, alg)
_mult(mpo::AbstractMPO, mps::AbstractMPS, alg::OneSiteStableArith) = stable_mult(mpo, mps, alg)
mpompsmult(mpo::AbstractMPO, mps::AbstractMPS, alg::AbstractMPSArith) = _mult(mpo, mps, alg)
mpompsmult(mpo::AbstractMPO, mps::AbstractMPS; alg::AbstractMPSArith=IterativeArith()) = _mult(mpo, mps, alg)

