


# @with_kw struct TwoSiteStableMult <: AbstractMPSArith
# 	D::Int = 100
# 	ϵ::Float64 = 1.0e-8
# 	maxiter::Int = 5
# 	verbosity::Int = 1
# 	tol::Float64 = 1.0e-8
# end



function stable_mult(mpo::AbstractMPO, mps::AbstractMPS, alg::OneSiteStableArith = OneSiteStableArith())
	svd_alg = get_svd_alg(alg)
	mpsout, err = svd_mult(mpo, mps, svd_alg)
	m = MPOMPSIterativeMultCache(mpo, mps, mpsout, init_hstorage_right(mpsout, mpo, mps))
	mult_alg = get_iterative_alg(alg)
	kvals = compute!(m, mult_alg)
	return m.omps, kvals[end]
end
