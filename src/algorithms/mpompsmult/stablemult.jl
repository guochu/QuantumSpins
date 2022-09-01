

@with_kw struct OneSiteStableArith <: AbstractMPSArith
	D::Int = 100
	ϵ::Float64 = 1.0e-8
	maxiter::Int = 5
	verbosity::Int = 1
	tol::Float64 = 1.0e-8
end

# @with_kw struct TwoSiteStableMult <: AbstractMPSArith
# 	D::Int = 100
# 	ϵ::Float64 = 1.0e-8
# 	maxiter::Int = 5
# 	verbosity::Int = 1
# 	tol::Float64 = 1.0e-8
# end

StableArith(; kwargs...) = OneSiteStableArith(; kwargs...) 

# StableArith(; single_site::Bool=true, kwargs...) = single_site ? OneSiteStableArith(; kwargs...) : TwoSiteStableMult(; kwargs...)

changeD(x::OneSiteStableArith; D::Int) = OneSiteStableArith(D=D, maxiter=x.maxiter, ϵ=x.ϵ, verbosity=x.verbosity, tol=x.tol)
# changeD(x::TwoSiteStableMult; D::Int) = TwoSiteStableMult(D=D, maxiter=x.maxiter, ϵ=x.ϵ, verbosity=x.verbosity, tol=x.tol)


get_svd_alg(x::OneSiteStableArith) = SVDArith(D=x.D, ϵ=x.ϵ, verbosity=x.verbosity)
get_iterative_alg(x::OneSiteStableArith) = OneSiteIterativeArith(
	D=x.D, fact=SVD(trunc=MPSTruncation(D=x.D, ϵ=x.ϵ)), maxiter=x.maxiter, tol=x.tol, verbosity=x.verbosity)
# get_iterative_alg(x::TwoSiteStableMult) = TwoSiteIterativeMult(D=x.D, maxiter=x.maxiter, tol=x.tol, verbosity=x.verbosity)


function stable_mult(mpo::AbstractMPO, mps::AbstractMPS, alg::OneSiteStableArith = OneSiteStableArith())
	svd_alg = get_svd_alg(alg)
	mpsout, err = svd_mult(mpo, mps, svd_alg)
	m = MPOMPSIterativeMultCache(mpo, mps, mpsout, init_hstorage_right(mpsout, mpo, mps))
	mult_alg = get_iterative_alg(alg)
	kvals = compute!(m, mult_alg)
	return m.omps, kvals[end]
end
