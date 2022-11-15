include("iterativemps.jl")
include("iterativempo.jl")
include("svdcompress.jl")
include("stablecompress.jl")


compress(m::Union{MPO, MPS}, alg::AbstractMPSArith=OneSiteStableArith()) = stable_compress(m, alg)
