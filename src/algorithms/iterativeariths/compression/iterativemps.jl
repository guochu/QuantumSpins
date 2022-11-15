
const MPSOverlapCache{_M, _H} = OverlapCache{_M, _M, _H} where {_M <: MPS}

"""
	iterative_compress(mps, alg)
	iterative_compress(mpo, alg)

Compress the given mps/mpo using iterative one-site or teo-site algorithm.

The given mps/mpo should be right-canonical, the initial guess is the copy
of the given mps/mpo.
"""
function iterative_compress(mps::MPS, alg::AbstractMPSArith=OneSiteIterativeArith())
	mpsout = randommps(scalar_type(mps), physical_dimensions(mpo), D=alg.D)
	canonicalize!(mpsout)
	return iterative_compress!(mpsout, mps, alg)
end 
function iterative_compress(h::MPO, alg::AbstractMPSArith=OneSiteIterativeArith())
	hout = randommpo(scalar_type(h), ophysical_dimensions(h), iphysical_dimensions(h), D=alg.D)
	rightorth!(hout, alg=QRFact())
	return iterative_compress!(hout, h, alg)
end
function iterative_compress!(omps::M, imps::M, alg::AbstractMPSArith = OneSiteIterativeArith()) where {M <: Union{MPS, MPO}}
	m = environments(omps, imps)
	kvals = iterative_compute!(m, alg)
	return bra(m), kvals[end]
end

sweep!(m::OverlapCache, alg::OneSiteIterativeArith) = vcat(leftsweep!(m, alg), rightsweep!(m, alg))

# A is assumed to be the input and B the output
function leftsweep!(m::MPSOverlapCache, alg::OneSiteIterativeArith, workspace = scalar_type(m)[])
	omps = bra(m)
	imps = ket(m)
	Cstorage = m.cstorage
	kvals = Float64[]
	L = length(m)
	for site in 1:L-1
		(alg.verbosity > 3) && println("sweeping from left to right at site: $site.")
        mpsj = reduceD_single_site(imps[site], Cstorage[site], Cstorage[site+1])
        push!(kvals, norm(mpsj))
        (alg.verbosity > 1) && println("residual after optimization on site $site is $(kvals[end])")
		omps[site], r = tqr!(mpsj, (1,2), (3,), workspace)
        Cstorage[site+1] = updateleft(Cstorage[site], omps[site], imps[site])		
	end
	return kvals
end

function rightsweep!(m::MPSOverlapCache, alg::OneSiteIterativeArith, workspace = scalar_type(m)[])
	omps = bra(m)
	imps = ket(m)
	Cstorage = m.cstorage
	kvals = Float64[]
	L = length(m)

    r = zeros(scalar_type(omps), 0, 0)
    isa(alg.fact, SVDFact) && maybe_init_boundary_s!(omps)
	for site in L:-1:2
		(alg.verbosity > 3) && println("sweeping from right to left at site: $site.")
		mpsj = reduceD_single_site(imps[site], Cstorage[site], Cstorage[site+1])
		push!(kvals, norm(mpsj))
		(alg.verbosity > 1) && println("residual after optimization on site $site is $(kvals[end])")

        if isa(alg.fact, QRFact)
        	r, omps[site] = tlq!(mpsj, (1,), (2,3), workspace)
        elseif isa(alg.fact, SVDFact)
            u, s, v, err = tsvd!(mpsj, (1,), (2,3), workspace, trunc=alg.fact.trunc)
            omps[site] = v
            r = u * Diagonal(s)
            omps.s[site] = s
        else
            error("unsupported factorization method $(typeof(alg.fact))")
        end
		Cstorage[site] = updateright(Cstorage[site+1], omps[site], imps[site])
	end
    omps[1] = @tensor tmp[1,2,4] := omps[1][1,2,3] * r[3,4]
    return kvals	

end


function reduceD_single_site(A::MPSTensor, Cleft::AbstractMatrix, Cright::AbstractMatrix)
    @tensor tmp[1 3; 5] := Cleft[1,2] * A[2,3,4] * Cright[5,4]
    return tmp
end

