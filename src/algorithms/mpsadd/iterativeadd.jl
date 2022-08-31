

@with_kw struct OneSiteIterativeAdd <: AbstractMPSAdd
	D::Int = 100
	maxiter::Int = 5
	verbosity::Int = 1
	tol::Float64 = 1.0e-8
end

IterativeAdd(;kwargs...) = OneSiteIterativeAdd(;kwargs...)

struct MPSIterativeAddCache{_MPSA, _MPSB, _H} <: AbstractMPSAddCache
    omps::_MPSA
    imps::Vector{_MPSB}
    hstorage::_H
end

MPSIterativeAddCache(omps::MPS, imps::Vector{<:MPS}) = MPSIterativeAddCache(omps, imps, [init_cstorage_right(omps, y) for y in imps])

scalar_type(x::MPSIterativeAddCache) = scalar_type(x.omps)


iterative_add(a::MPS, b::MPS, alg::OneSiteIterativeAdd=OneSiteIterativeAdd()) = iterative_add([a, b], alg)
function iterative_add(mpsxs::Vector{MPS{T, R}}, alg::OneSiteIterativeAdd=OneSiteIterativeAdd()) where {T, R}
    isempty(mpsxs) && error("no input state.")
    (length(mpsxs) == 1) && return mpsxs[1]
    ds = physical_dimensions(mpsxs[1])
    for i in 2:length(mpsxs)
    	@assert ds == physical_dimensions(mpsxs[i])
    end
    mpsout = randommps(T, physical_dimensions(mpsxs[1]), D=alg.D)
    rightorth_qr!(mpsout)

    m = MPSIterativeAddCache(mpsout, mpsxs)
    kvals = compute!(m, alg)
    return m.omps, kvals[end]
end

sweep!(m::MPSIterativeAddCache, alg::OneSiteIterativeAdd, workspace = scalar_type(m)[]) = iterative_error_2(
    vcat(_leftsweep!(m, alg, workspace), _rightsweep!(m, alg, workspace)))

function compute!(m::MPSIterativeAddCache, alg::OneSiteIterativeAdd, workspace = scalar_type(m)[])
	kvals = Float64[]
	iter = 0
	tol = 1.
	while (iter < alg.maxiter) && (tol >= alg.tol)
		tol = sweep!(m, alg, workspace)
		push!(kvals, tol)
		iter += 1
		(alg.verbosity > 2) && println("finish the $iter-th sweep with error $tol", "\n")
	end
    if (alg.verbosity >= 2) && (iter < alg.maxiter)
        println("early converge in $iter-th sweeps with error $tol")
    end
    if (alg.verbosity > 2) && (iter >= alg.maxiter)
        println("fail to converge, required precision: $(alg.tol), actual precision $tol in $iter sweeps.")
    end
	return kvals
end

function _leftsweep!(x::MPSIterativeAddCache, alg::OneSiteIterativeAdd, workspace = scalar_type(m)[])
	mpsxs = x.imps
	mpsy = x.omps
	cstorages = x.hstorage
	N = length(mpsxs)
	L = length(mpsy)
	kvals = Float64[]
    for site in 1:L-1
        (alg.verbosity > 2) && println("Sweeping from left to right at bond: $site.")
        mpsj = _compute_one_site_mpsj(mpsxs, cstorages, site)
        push!(kvals, norm(mpsj))
        (alg.verbosity >= 2) && println("residual is $(kvals[end])...")
		mpsy[site], r = tqr!(mpsj, (1,2), (3,), workspace)
        for n in 1:N
            cstorages[n][site+1] = updateleft(cstorages[n][site], mpsy[site], mpsxs[n][site])
        end
    end
    return kvals
end

function _rightsweep!(x::MPSIterativeAddCache, alg::OneSiteIterativeAdd, workspace = scalar_type(m)[])
	mpsxs = x.imps
	mpsy = x.omps
	cstorages = x.hstorage
	N = length(mpsxs)
	L = length(mpsy)

    kvals = Float64[]
    l = zeros(scalar_type(mpsy), 0, 0)
    maybe_init_boundary_s!(mpsy)
    for site in L:-1:2
        (alg.verbosity > 2) && println("Sweeping from right to left at bond: $site.")
        mpsj = _compute_one_site_mpsj(mpsxs, cstorages, site)
        push!(kvals, norm(mpsj))
        (alg.verbosity > 2) && println("residual is $(kvals[end])...")
        # l, mpsy[site] = tlq!(mpsj, (1,), (2,3), workspace)
        u, s, v, err = tsvd!(mpsj, (1,), (2,3), workspace, trunc=NoTruncation())
        mpsy[site] = v
        l = u * Diagonal(s)
        mpsy.s[site] = s

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

