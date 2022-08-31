
include("abstractmult.jl")
include("iterativemult.jl")
include("svdmult.jl")
include("stablemult.jl")


_mult(mpo::AbstractMPO, mps::AbstractMPS, alg::Union{OneSiteIterativeMult, TwoSiteIterativeMult}) = iterative_mult(mpo, mps, alg)
_mult(mpo::AbstractMPO, mps::AbstractMPS, alg::SVDMult) = svd_mult(mpo, mps, alg)
_mult(mpo::AbstractMPO, mps::AbstractMPS, alg::Union{OneSiteStableMult, TwoSiteStableMult}) = stable_mult(mpo, mps, alg)
mpompsmult(mpo::AbstractMPO, mps::AbstractMPS, alg::AbstractMPOMPSMult) = _mult(mpo, mps, alg)
mpompsmult(mpo::AbstractMPO, mps::AbstractMPS; alg::AbstractMPOMPSMult=IterativeMult()) = _mult(mpo, mps, alg)

