include("abstractdef.jl")
include("iterativemult.jl")

mpompomult(mpo::AbstractMPO, mps::AbstractMPO, alg::OneSiteIterativeArith) = iterative_mult(mpo, mps, alg)