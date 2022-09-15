

struct MPOMPOIterativeMultCache{_MPO, _IMPO, _OMPO, _H} <: AbstractMPOMPOMultCache
	mpo::_MPO
	impo::_IMPO
	ompo::_OMPO
	hstorage::_H
end

scalar_type(m::MPOMPOIterativeMultCache) = scalar_type(m.ompo)

sweep!(m::MPOMPOIterativeMultCache, alg::AbstractMPSArith, workspace = scalar_type(m)[]) = iterative_error_2(
    vcat(_leftsweep!(m, alg, workspace), _rightsweep!(m, alg, workspace)))

# function compute!(m::MPOMPOIterativeMultCache, alg::AbstractMPSArith, workspace = scalar_type(m)[])
#     kvals = Float64[]
#     iter = 0
#     tol = 1.
#     while (iter < alg.maxiter) && (tol >= alg.tol)
#         tol = sweep!(m, alg, workspace)
#         push!(kvals, tol)
#         iter += 1
#         (alg.verbosity > 2) && println("finish the $iter-th sweep with error $tol", "\n")
#     end
#     if (alg.verbosity >= 2) && (iter < alg.maxiter)
#         println("early converge in $iter-th sweeps with error $tol")
#     end
#     if (alg.verbosity > 2) && (tol >= alg.tol)
#         println("fail to converge, required precision: $(alg.tol), actual precision $tol in $iter sweeps.")
#     end
#     return kvals
# end
compute!(m::MPOMPOIterativeMultCache, alg::AbstractMPSArith, workspace = scalar_type(m)[]) = iterative_compute!(m, alg, workspace)

function iterative_mult(mpo::MPO, mps::MPO, alg::OneSiteIterativeArith = OneSiteIterativeArith())
    T = promote_type(scalar_type(mpo), scalar_type(mps))
    mpsout = randommpo(T, ophysical_dimensions(mpo), iphysical_dimensions(mps), D=alg.D)
    # canonicalize!(mpsout, normalize=true)
    rightorth!(mpsout, alg=QRFact())
    m = MPOMPOIterativeMultCache(mpo, mps, mpsout, init_hstorage_right(mpsout, mpo, mps))
    kvals = compute!(m, alg)
    return m.ompo, kvals[end]
end


function _leftsweep!(m::MPOMPOIterativeMultCache, alg::OneSiteIterativeArith, workspace = scalar_type(m)[])
    mpoA = m.impo
    mpo = m.mpo
    mpoB = m.ompo
    Cstorage = m.hstorage
    L = length(mpo)
    kvals = Float64[]
    for site in 1:L-1
        (alg.verbosity > 2) && println("Sweeping from left to right at bond: $site.")
        mpsj = reduceH_single_site(mpoA[site], mpo[site], Cstorage[site], Cstorage[site+1])
        push!(kvals, norm(mpsj))
        (alg.verbosity > 2) && println("residual is $(kvals[end])...")
		q, r = tqr!(mpsj, (1,2,4), (3,), workspace)
        mpoB[site] = permute(q, (1,2,4,3))
        Cstorage[site+1] = updateHleft(Cstorage[site], mpoB[site], mpo[site], mpoA[site])
    end
    return kvals	
end

function _rightsweep!(m::MPOMPOIterativeMultCache, alg::OneSiteIterativeArith, workspace = scalar_type(m)[])
    mpoA = m.impo
    mpo = m.mpo
    mpoB = m.ompo
    Cstorage = m.hstorage
    L = length(mpo)
    kvals = Float64[]
    r = zeros(scalar_type(mpoB), 0, 0)
    for site in L:-1:2
        (alg.verbosity > 2) && println("Sweeping from right to left at bond: $site.")
        mpsj = reduceH_single_site(mpoA[site], mpo[site], Cstorage[site], Cstorage[site+1])
        push!(kvals, norm(mpsj))
        (alg.verbosity > 2) && println("residual is $(kvals[end])...")
        if isa(alg.fact, QRFact)
        	r, mpoB[site] = tlq!(mpsj, (1,), (2,3,4), workspace)
        elseif isa(alg.fact, SVDFact)
            u, s, v, err = tsvd!(mpsj, (1,), (2,3,4), workspace, trunc=alg.fact.trunc)
            mpoB[site] = v
            r = u * Diagonal(s)
        end
        Cstorage[site] = updateHright(Cstorage[site+1], mpoB[site], mpo[site], mpoA[site])
    end
    # println("norm of r is $(norm(r))")
    mpoB[1] = @tensor tmp[1,2,5,4] := mpoB[1][1,2,3,4] * r[3,5]
    return kvals	
end

function reduceH_single_site(A::AbstractArray{<:Number}, m::AbstractArray{<:Number, 4}, cleft::AbstractArray{<:Number, 3}, cright::AbstractArray{<:Number, 3})
	@tensor tmp[1,7,9,6] := ((cleft[1,2,3] * A[3,4,5,6]) * m[2,7,8,4]) * cright[9,8,5]
    return tmp
end

function updateHleft(cleft::AbstractArray{<:Number, 3}, B::AbstractArray{<:Number, 4}, m::AbstractArray{<:Number, 4}, A::AbstractArray{<:Number, 4})
    @tensor tmp[9,8,5] := ((cleft[1,2,3] * A[3,4,5,6]) * m[2,7,8,4]) * conj(B[1,7,9,6])
    return tmp
end

function updateHright(cright::AbstractArray{<:Number, 3}, B::AbstractArray{<:Number, 4}, m::AbstractArray{<:Number, 4}, A::AbstractArray{<:Number, 4})
    @tensor tmp[1,7,9] := ((conj(B[1,2,3,4]) * cright[3,5,6]) * m[7,2,5,8] ) * A[9,8,6,4]
    return tmp
end

function init_hstorage_right(B::MPO, mpo::MPO, A::MPO)
    @assert length(B) == length(mpo) == length(A)
    L = length(mpo)
    T = promote_type(scalar_type(B), scalar_type(mpo), scalar_type(A))
    hstorage = Vector{Array{T, 3}}(undef, L+1)
    hstorage[1] = ones(1,1,1)
    hstorage[L+1] = ones(1,1,1)
    for i in L:-1:2
        hstorage[i] = updateHright(hstorage[i+1], B[i], mpo[i], A[i])
    end
    return hstorage
end

