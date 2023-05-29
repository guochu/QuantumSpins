abstract type AbstractMPSAddCache end

struct MPSIterativeAddCache{_MPSA, _MPSB, _H} <: AbstractMPSAddCache
    omps::_MPSA
    imps::Vector{_MPSB}
    hstorage::_H
end

MPSIterativeAddCache(omps::MPS, imps::Vector{<:MPS}) = MPSIterativeAddCache(omps, imps, [init_cstorage_right(omps, y) for y in imps])

Base.eltype(x::MPSIterativeAddCache) = eltype(x.omps)


iterative_add(a::MPS, b::MPS, alg::OneSiteIterativeArith=OneSiteIterativeArith()) = iterative_add([a, b], alg)
function iterative_add(mpsxs::Vector{MPS{T, R}}, alg::OneSiteIterativeArith=OneSiteIterativeArith()) where {T, R}
    isempty(mpsxs) && error("no input state.")
    (length(mpsxs) == 1) && return mpsxs[1]
    ds = physical_dimensions(mpsxs[1])
    for i in 2:length(mpsxs)
    	@assert ds == physical_dimensions(mpsxs[i])
    end
    mpsout = randommps(T, physical_dimensions(mpsxs[1]), D=alg.D)
    rightorth!(mpsout, alg=Orthogonalize(QR()))

    m = MPSIterativeAddCache(mpsout, mpsxs)
    kvals = compute!(m, alg)
    return m.omps, kvals[end]
end

sweep!(m::MPSIterativeAddCache, alg::OneSiteIterativeArith, workspace = eltype(m)[]) = vcat(_leftsweep!(m, alg, workspace), _rightsweep!(m, alg, workspace))


compute!(m::MPSIterativeAddCache, alg::OneSiteIterativeArith, workspace = eltype(m)[]) = iterative_compute!(m, alg, workspace)

function _leftsweep!(x::MPSIterativeAddCache, alg::OneSiteIterativeArith, workspace = eltype(m)[])
	mpsxs = x.imps
	mpsy = x.omps
	cstorages = x.hstorage
	N = length(mpsxs)
	L = length(mpsy)
	kvals = Float64[]
    for site in 1:L-1
        (alg.verbosity > 3) && println("Sweeping from left to right at bond: $site.")
        mpsj = _compute_one_site_mpsj(mpsxs, cstorages, site)
        push!(kvals, norm(mpsj))
        (alg.verbosity > 1) && println("residual is $(kvals[end])...")
		mpsy[site], r = tqr!(mpsj, (1,2), (3,), workspace)
        for n in 1:N
            cstorages[n][site+1] = updateleft(cstorages[n][site], mpsy[site], mpsxs[n][site])
        end
    end
    return kvals
end

function _rightsweep!(x::MPSIterativeAddCache, alg::OneSiteIterativeArith, workspace = eltype(m)[])
	mpsxs = x.imps
	mpsy = x.omps
	cstorages = x.hstorage
	N = length(mpsxs)
	L = length(mpsy)

    kvals = Float64[]
    l = zeros(eltype(mpsy), 0, 0)

    for site in L:-1:2
        (alg.verbosity > 3) && println("Sweeping from right to left at bond: $site.")
        mpsj = _compute_one_site_mpsj(mpsxs, cstorages, site)
        push!(kvals, norm(mpsj))
        (alg.verbosity > 1) && println("residual is $(kvals[end])...")
        # l, mpsy[site] = tlq!(mpsj, (1,), (2,3), workspace)


        l, mpsy[site] = tlq!(mpsj, (1,), (2,3), workspace)

        for n in 1:N
            cstorages[n][site] = updateright(cstorages[n][site+1], mpsy[site], mpsxs[n][site])
        end
    end	
    @tensor tmp[1,2,4] := mpsy[1][1,2,3] * l[3,4] 
    mpsy[1] = tmp
    return kvals
end



function _compute_one_site_mpsj(mpsxs::Vector, hstorage::Vector, site::Int)
	@tensor r[1,3,5] := hstorage[1][site][1,2] * mpsxs[1][site][2,3,4] * hstorage[1][site+1][5,4]
    for n in 2:length(hstorage)
        @tensor r[1,3,5] += hstorage[n][site][1,2] * mpsxs[n][site][2,3,4] * hstorage[n][site+1][5,4]
    end
    # r ./= length(hstorage)
    return r
end

