abstract type DMRGAlgorithm end

@with_kw struct DMRG <: DMRGAlgorithm 
	maxiter::Int = Defaults.maxiter
	tol::Float64 = Defaults.tol
	verbosity::Int = Defaults.verbosity
end

function calc_galerkin(m::Union{ExpectationCache, ProjectedExpectationCache}, site::Int)
	mpsj = m.mps[site]
	try
		return norm(_leftnull_qr!(copy(tie(mpsj, (2, 1))), 0.)' * tie(ac_prime(mpsj, m.mpo[site], m.hstorage[site], m.hstorage[site+1]), (2, 1) ) )
	catch
		return norm(tie(ac_prime(mpsj, m.mpo[site], m.hstorage[site], m.hstorage[site+1]), (1,2)) * _rightnull_lq!(tie(mpsj, (1,2)), 0.)' )
	end
end

# delayed evaluation of galerkin error.
function leftsweep!(m::ExpectationCache, alg::DMRG)
	mpo = m.mpo
	mps = m.mps
	hstorage = m.env
	Energies = Float64[]
	delta = 0.
	for site in 1:length(mps)-1
		(alg.verbosity > 2) && println("sweeping from left to right at site: $site.")
		eigvals, vecs = eigsolve(x->ac_prime(x, mpo[site], hstorage[site], hstorage[site+1]), mps[site], 1, :SR, Lanczos())
		push!(Energies, eigvals[1])
		(alg.verbosity > 2) && println("Energy after optimization on site $site is $(Energies[end]).")
		# galerkin error
		delta = max(delta, calc_galerkin(m, site) )
		# prepare mps site tensor to be left canonical
		Q, R = tqr!(vecs[1], (1,2), (3,))
		mps[site] = Q
		mps[site+1] = @tensor tmp[-1 -2; -3] := R[-1, 1] * mps[site+1][1, -2, -3]
		# hstorage[site+1] = updateleft(hstorage[site], mps[site], mpo[site], mps[site])
		updateleft!(m, site)
	end
	return Energies, delta
end

function rightsweep!(m::ExpectationCache, alg::DMRG)
	mpo = m.mpo
	mps = m.mps
	hstorage = m.env
	Energies = Float64[]
	delta = 0.
	for site in length(mps):-1:2
		(alg.verbosity > 2) && println("sweeping from right to left at site: $site.")
		eigvals, vecs = eigsolve(x->ac_prime(x, mpo[site], hstorage[site], hstorage[site+1]), mps[site], 1, :SR, Lanczos())
		push!(Energies, eigvals[1])
		(alg.verbosity > 2) && println("Energy after optimization on site $site is $(Energies[end]).")		
		# galerkin error
		delta = max(delta, calc_galerkin(m, site) )
		# prepare mps site tensor to be right canonical
		L, Q = tlq!(vecs[1], (1,), (2,3))
		mps[site] = permute(Q, (1,2), (3,))
		mps[site-1] = @tensor tmp[-1 -2; -3] := mps[site-1][-1, -2, 1] * L[1, -3]
		# hstorage[site] = updateright(hstorage[site+1], mps[site], mpo[site], mps[site])
		updateright!(m, site)
	end
	return Energies, delta
end


"""
	compute!(env::AbstractCache, alg::DMRGAlgorithm)
	execute dmrg iterations
"""
function compute!(env::AbstractCache, alg::DMRGAlgorithm)
	all_energies = Float64[]
	iter = 0
	delta = 2 * alg.tol
	while iter < alg.maxiter && delta > alg.tol
		Energies, delta = sweep!(env, alg)
		append!(all_energies, Energies)
		iter += 1
		(alg.verbosity > 2) && println("finish the $iter-th sweep with error $delta", "\n")
	end
	return all_energies, delta
end

"""
	ground_state!(state::FiniteMPS, h::Union{MPOHamiltonian, FiniteMPO}, alg::DMRGAlgorithm)
compute the ground state
"""
ground_state!(state::MPS, h::MPO, alg::DMRGAlgorithm=DMRG()) = compute!(environments(h, state), alg)


