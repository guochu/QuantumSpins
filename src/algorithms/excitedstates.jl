


function _project!(y, projectors)
	for p in projectors
		y = axpy!(-dot(y, p), p, y)
	end	
	return y
end


function leftsweep!(m::ProjectedExpectationCache, alg::DMRG)
	mpo = m.mpo
	mps = m.mps
	hstorage = m.env
	cstorages = m.cenvs
	projectors = m.projectors
	Energies = Float64[]
	delta = 0.
	for site in 1:length(mps)-1
		(alg.verbosity > 3) && println("sweeping from left to right at site: $site")
		p1 = [ac_prime(projectors[l][site], cstorages[l][site], cstorages[l][site+1]) for l in 1:length(cstorages)]
		sitemps = _project!(copy(mps[site]), p1)
		eigvals, vecs = eigsolve(x->_project!(ac_prime(x, mpo[site], hstorage[site], hstorage[site+1]), p1), sitemps, 1, :SR, Lanczos())
		push!(Energies, eigvals[1])
		(alg.verbosity > 1) && println("Energy after optimization on site $site is $(Energies[end])")
		delta = max(delta, calc_galerkin(m, site) )
		# prepare mps site tensor to be left canonical
		Q, R = tqr!(vecs[1], (1,2), (3,))
		mps[site] = Q
		mps[site+1] = @tensor tmp[-1 -2; -3] := R[-1, 1] * mps[site+1][1, -2, -3]
		# hstorage[site+1] = updateleft(hstorage[site], mps[site], mpo[site], mps[site])
		# for l in 1:length(cstorages)
		#     cstorages[l][site+1] = updateleft(cstorages[l][site], mps[site], projectors[l][site])
		# end
		updateleft!(m, site)
	end
	return Energies, delta
end

function rightsweep!(m::ProjectedExpectationCache, alg::DMRG)
	mpo = m.mpo
	mps = m.mps
	hstorage = m.env
	cstorages = m.cenvs
	projectors = m.projectors
	Energies = Float64[]
	delta = 0.
	for site in length(mps):-1:2
		(alg.verbosity > 3) && println("sweeping from right to left at site: $site")
		p1 = [ac_prime(projectors[l][site], cstorages[l][site], cstorages[l][site+1]) for l in 1:length(cstorages)]
		sitemps = _project!(copy(mps[site]), p1)
		eigvals, vecs = eigsolve(x->_project!(ac_prime(x, mpo[site], hstorage[site], hstorage[site+1]), p1), sitemps, 1, :SR, Lanczos())
		push!(Energies, eigvals[1])
		(alg.verbosity > 1) && println("Energy after optimization on site $site is $(Energies[end])")		
		delta = max(delta, calc_galerkin(m, site) )
		# prepare mps site tensor to be right canonical
		L, Q = tlq!(vecs[1], (1,), (2,3))
		mps[site] = permute(Q, (1,2), (3,))
		mps[site-1] = @tensor tmp[-1 -2; -3] := mps[site-1][-1, -2, 1] * L[1, -3]
		# hstorage[site] = updateright(hstorage[site+1], mps[site], mpo[site], mps[site])
		# for l in 1:length(cstorages)
		#     cstorages[l][site] = updateright(cstorages[l][site+1],  mps[site], projectors[l][site])
		# end
		updateright!(m, site)
	end
	return Energies, delta
end

function sweep!(m::Union{ExpectationCache, ProjectedExpectationCache}, alg::DMRGAlgorithm=DMRG(); kwargs...)
	Energies1, delta1 = leftsweep!(m, alg; kwargs...)
	Energies2, delta2 = rightsweep!(m, alg; kwargs...)
	return vcat(Energies1, Energies2), max(delta1, delta2)
end

