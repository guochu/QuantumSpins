

stable_add(psiA::MPS, psiB::MPS, alg::OneSiteStableArith = OneSiteStableArith()) = stable_add([psiA, psiB], alg)
function stable_add(psis::Vector{<:MPS}, alg::OneSiteStableArith = OneSiteStableArith())
	svd_alg = get_svd_alg(alg)
	mpsout, err = svd_add(psis, svd_alg)
	m = MPSIterativeAddCache(mpsout, psis)
	mult_alg = get_iterative_alg(alg)
	kvals = compute!(m, mult_alg)
	return m.omps, kvals[end]
end