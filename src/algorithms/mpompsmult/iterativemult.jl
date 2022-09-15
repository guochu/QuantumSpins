# iterative MPO MPS multiplication
# one-site algorithm
# two-site version is used for debug, which is much more expensive than one-site version


struct OneSiteIterativeArith{F} <: AbstractMPSArith
	D::Int 
	maxiter::Int 
	fact::F 
	verbosity::Int 
	tol::Float64 
end

OneSiteIterativeArith(;D::Int=100, maxiter::Int=5, fact::AbstractMatrixFactorization=SVDFact(), verbosity::Int=1, tol::Real=1.0e-8) = OneSiteIterativeArith(
    D, maxiter, fact, verbosity, tol)
IterativeArith(;kwargs...) = OneSiteIterativeArith(; kwargs...) 

changeD(x::OneSiteIterativeArith; D::Int) = OneSiteIterativeArith(D=D, maxiter=x.maxiter, fact=x.fact, verbosity=x.verbosity, tol=x.tol)

# @with_kw struct TwoSiteIterativeMult <: AbstractMPSArith
# 	D::Int = 100
# 	maxiter::Int = 5
# 	verbosity::Int = 1
# 	tol::Float64 = 1.0e-8
# end
# changeD(x::TwoSiteIterativeMult; D::Int) = TwoSiteIterativeMult(D=D, maxiter=x.maxiter, verbosity=x.verbosity, tol=x.tol)

# IterativeMult(;single_site::Bool=true, kwargs...) = single_site ? OneSiteIterativeArith(; kwargs...) : TwoSiteIterativeMult(; kwargs...)


struct MPOMPSIterativeMultCache{_MPO, _IMPS, _OMPS, _H} <: AbstractMPOMPSMultCache
	mpo::_MPO
	imps::_IMPS
	omps::_OMPS
	hstorage::_H
end



scalar_type(m::MPOMPSIterativeMultCache) = scalar_type(m.omps)

sweep!(m::MPOMPSIterativeMultCache, alg::AbstractMPSArith, workspace = scalar_type(m)[]) = iterative_error_2(
    vcat(_leftsweep!(m, alg, workspace), _rightsweep!(m, alg, workspace)))

# function compute!(m::MPOMPSIterativeMultCache, alg::AbstractMPSArith, workspace = scalar_type(m)[])
# 	kvals = Float64[]
# 	iter = 0
# 	tol = 1.
# 	while (iter < alg.maxiter) && (tol >= alg.tol)
# 		tol = sweep!(m, alg, workspace)
# 		push!(kvals, tol)
# 		iter += 1
# 		(alg.verbosity > 2) && println("finish the $iter-th sweep with error $tol", "\n")
# 	end
#     if (alg.verbosity >= 2) && (iter < alg.maxiter)
#         println("early converge in $iter-th sweeps with error $tol")
#     end
#     if (alg.verbosity > 2) && (tol >= alg.tol)
#         println("fail to converge, required precision: $(alg.tol), actual precision $tol in $iter sweeps.")
#     end
# 	return kvals
# end
compute!(m::MPOMPSIterativeMultCache, alg::AbstractMPSArith, workspace = scalar_type(m)[]) = iterative_compute!(m, alg, workspace)

function iterative_mult(mpo::MPO, mps::MPS, alg::OneSiteIterativeArith = OneSiteIterativeArith())
	mpsout = randommps(promote_type(scalar_type(mpo), scalar_type(mps)), ophysical_dimensions(mpo), D=alg.D)
	# canonicalize!(mpsout, normalize=true)
    rightorth!(mpsout, alg=QRFact())
	m = MPOMPSIterativeMultCache(mpo, mps, mpsout, init_hstorage_right(mpsout, mpo, mps))
	kvals = compute!(m, alg)
	return m.omps, kvals[end]
end


function _leftsweep!(m::MPOMPSIterativeMultCache, alg::OneSiteIterativeArith, workspace = scalar_type(m)[])
    mpsA = m.imps
    mpo = m.mpo
    mpsB = m.omps
    Cstorage = m.hstorage
    L = length(mpo)
    kvals = Float64[]
    for site in 1:L-1
        (alg.verbosity > 2) && println("Sweeping from left to right at bond: $site.")
        mpsj = reduceD_single_site(mpsA[site], mpo[site], Cstorage[site], Cstorage[site+1])
        push!(kvals, norm(mpsj))
        (alg.verbosity > 2) && println("residual is $(kvals[end])...")
		mpsB[site], r = tqr!(mpsj, (1,2), (3,), workspace)
        Cstorage[site+1] = updateleft(Cstorage[site], mpsB[site], mpo[site], mpsA[site])
    end
    return kvals	
end

function _rightsweep!(m::MPOMPSIterativeMultCache, alg::OneSiteIterativeArith, workspace = scalar_type(m)[])
    mpsA = m.imps
    mpo = m.mpo
    mpsB = m.omps
    Cstorage = m.hstorage
    L = length(mpo)
    kvals = Float64[]
    r = zeros(scalar_type(mpsB), 0, 0)
    isa(alg.fact, SVDFact) && maybe_init_boundary_s!(mpsB)
    for site in L:-1:2
        (alg.verbosity > 2) && println("Sweeping from right to left at bond: $site.")
        mpsj = reduceD_single_site(mpsA[site], mpo[site], Cstorage[site], Cstorage[site+1])
        push!(kvals, norm(mpsj))
        (alg.verbosity > 2) && println("residual is $(kvals[end])...")
        if isa(alg.fact, QRFact)
        	r, mpsB[site] = tlq!(mpsj, (1,), (2,3), workspace)
        elseif isa(alg.fact, SVDFact)
            u, s, v, err = tsvd!(mpsj, (1,), (2,3), workspace, trunc=alg.fact.trunc)
            mpsB[site] = v
            r = u * Diagonal(s)
            mpsB.s[site] = s
        end
        Cstorage[site] = updateright(Cstorage[site+1], mpsB[site], mpo[site], mpsA[site])
    end
    # println("norm of r is $(norm(r))")
    @tensor tmp[1,2,4] := mpsB[1][1,2,3] * r[3,4]
    mpsB[1] = tmp
    return kvals	
end

# function _leftsweep!(m::MPOMPSIterativeMultCache, alg::TwoSiteIterativeMult, workspace = scalar_type(m)[])
#     mpsA = m.imps
#     mpo = m.mpo
#     mpsB = m.omps
#     Cstorage = m.hstorage
#     L = length(mpo)
#     kvals = Float64[]
#     trunc = MPSTruncation(D=alg.D, ϵ=alg.tol/10)
#     for site in 1:L-2
#         (alg.verbosity > 2) && println("Sweeping from left to right at bond: $site.")
#         @tensor twompsA[1,2,4,5] := mpsA[site][1,2,3] * mpsA[site+1][3,4,5]
#         twompsB = reduceD_single_bond(twompsA, mpo[site], mpo[site+1], Cstorage[site], Cstorage[site+2])
#         push!(kvals, norm(twompsB))
#         (alg.verbosity >= 2) && println("residual is $(kvals[end])...")
#         mpsB[site], s, mpsB[site+1], bet = tsvd!(twompsB, (1,2),(3,4), workspace, trunc=trunc)
#         Cstorage[site+1] = updateleft(Cstorage[site], mpsB[site], mpo[site], mpsA[site])
#     end
#     return kvals
# end

# function _rightsweep!(m::MPOMPSIterativeMultCache, alg::TwoSiteIterativeMult, workspace = scalar_type(m)[])
#     mpsA = m.imps
#     mpo = m.mpo
#     mpsB = m.omps
#     Cstorage = m.hstorage
# 	L = length(mpo)
#     kvals = Float64[]
#     trunc = MPSTruncation(D=alg.D, ϵ=alg.tol/10)
#     maybe_init_boundary_s!(mpsB)
#     for site in L-1:-1:1
#         (alg.verbosity > 2) && println("we are sweeping from right to left at bond: $site.")
#         @tensor twompsA[1,2,4,5] := mpsA[site][1,2,3] * mpsA[site+1][3,4,5]
#         twompsB = reduceD_single_bond(twompsA, mpo[site], mpo[site+1], Cstorage[site], Cstorage[site+2])
#         push!(kvals, norm(twompsB))
#         (alg.verbosity >= 2) && println("residual is $(kvals[end])...")
#         mpsB[site], s, mpsB[site+1], bet = tsvd!(twompsB, (1,2),(3,4), workspace, trunc=trunc)
#         mpsB.s[site+1] = s
#         Cstorage[site+1] = updateright(Cstorage[site+2], mpsB[site+1], mpo[site+1], mpsA[site+1])
#     end
#     sm = diag(mpsB.s[2])
#     @tensor tmp[1,2,4] := mpsB[1][1,2,3] * sm[3,4]
#     mpsB[1] = tmp
#     return kvals	
# end

function reduceD_single_site(A::AbstractArray{<:Number, 3}, X::AbstractArray{<:Number, 4}, Cleft::AbstractArray{<:Number, 3}, 
	Cright::AbstractArray{<:Number, 3})
    @tensor tmp[1,6,8] := ((Cleft[1,2,3] * A[3,4,5]) * X[2,6,7,4]) * Cright[8,7,5]
    return tmp
end

# function reduceD_single_bond(A::AbstractArray{<:Number, 4}, X1::AbstractArray{<:Number, 4}, X2::AbstractArray{<:Number, 4}, 
# 	Cleft::AbstractArray{<:Number, 3}, Cright::AbstractArray{<:Number, 3})
# 	@tensor temp[1,2,4,5,6] := Cleft[1,2,3] * A[3,4,5,6]
# 	@tensor temp2[1,6,7,4,5] := temp[1,2,3,4,5] * X1[2,6,7,3]
# 	@tensor temp3[1,2,6,7,5] := temp2[1,2,3,4,5] * X2[3,6,7,4]
# 	@tensor temp4[1,2,3,6] := temp3[1,2,3,4,5] * Cright[6,4,5]
# 	return temp4
# end

function init_hstorage_right(mpsB, mpo, mpsA)
	T = promote_type(scalar_type(mpsA), scalar_type(mpo), scalar_type(mpsB))
	L = length(mpo)
	hstorage = Vector{Array{T, 3}}(undef, L+1)
	hstorage[1] = ones(T, 1, 1, 1)
	hstorage[L+1] = ones(T, 1, 1, 1)
	for i in L:-1:2
		hstorage[i] = updateright(hstorage[i+1], mpsB[i], mpo[i], mpsA[i])
	end
	return hstorage
end

iterative_error_2(m::AbstractVector) = std(m) / abs(mean(m))
