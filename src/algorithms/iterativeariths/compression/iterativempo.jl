
const MPOOverlapCache{_M, _H} = OverlapCache{_M, _M, _H} where {_M <: MPO}


# A is assumed to be the input and B the output
function leftsweep!(m::MPOOverlapCache, alg::OneSiteIterativeArith, workspace = scalar_type(m)[])
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
		tmp, r = tqr!(mpsj, (1,2,4), (3,), workspace)
		omps[site] = permute(tmp, (1,2), (4,3))
        Cstorage[site+1] = updateleft(Cstorage[site], omps[site], imps[site])		
	end
	return kvals
end

function rightsweep!(m::MPOOverlapCache, alg::OneSiteIterativeArith, workspace = scalar_type(m)[])
	omps = bra(m)
	imps = ket(m)
	Cstorage = m.cstorage
	kvals = Float64[]
	L = length(m)
	r = zeros(scalar_type(omps), 0, 0)
	for site in L:-1:2
		(alg.verbosity > 3) && println("sweeping from right to left at site: $site.")
		mpsj = reduceD_single_site(imps[site], Cstorage[site], Cstorage[site+1])
		push!(kvals, norm(mpsj))
		(alg.verbosity > 1) && println("residual after optimization on site $site is $(kvals[end])")

        r, omps[site] = tlq!(mpsj, (1,), (2,3,4), workspace)
        Cstorage[site] = updateright(Cstorage[site+1], omps[site], imps[site])
	end
	omps[1] = @tensor tmp[1,2,5,4] := omps[1][1,2,3,4] * r[3,5]
    return kvals	

end


"""
	two site mps tensor convention
		2 3
	1        4
"""
function reduceD_single_site(A::MPOTensor, Cleft::AbstractMatrix, Cright::AbstractMatrix)
    @tensor tmp[1 3; 6 5] := Cleft[1,2] * A[2,3,4,5] * Cright[6,4]
    return tmp
end
