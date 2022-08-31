

@with_kw struct OneSiteStableMult <: AbstractMPOMPSMult
	D::Int = 100
	ϵ::Float64 = 1.0e-8
	maxiter::Int = 5
	verbosity::Int = 1
	tol::Float64 = 1.0e-8
end

@with_kw struct TwoSiteStableMult <: AbstractMPOMPSMult
	D::Int = 100
	ϵ::Float64 = 1.0e-8
	maxiter::Int = 5
	verbosity::Int = 1
	tol::Float64 = 1.0e-8
end

StableMult(; single_site::Bool=true, kwargs...) = single_site ? OneSiteStableMult(; kwargs...) : TwoSiteStableMult(; kwargs...)

changeD(x::OneSiteStableMult; D::Int) = OneSiteStableMult(D=D, maxiter=x.maxiter, ϵ=x.ϵ, verbosity=x.verbosity, tol=x.tol)
changeD(x::TwoSiteStableMult; D::Int) = TwoSiteStableMult(D=D, maxiter=x.maxiter, ϵ=x.ϵ, verbosity=x.verbosity, tol=x.tol)


get_svd_alg(x::Union{OneSiteStableMult, TwoSiteStableMult}) = SVDMult(D=x.D, ϵ=x.ϵ, verbosity=x.verbosity)
get_iterative_alg(x::OneSiteStableMult) = OneSiteIterativeMult(D=x.D, maxiter=x.maxiter, tol=x.tol, verbosity=x.verbosity)
get_iterative_alg(x::TwoSiteStableMult) = TwoSiteIterativeMult(D=x.D, maxiter=x.maxiter, tol=x.tol, verbosity=x.verbosity)


function stable_mult(mpo::AbstractMPO, mps::AbstractMPS, alg::AbstractMPOMPSMult = OneSiteStableMult())
	svd_alg = get_svd_alg(alg)
	mpsout, err = svd_mult(mpo, mps, svd_alg)
	m = MPOMPSIterativeMultCache(mpo, mps, mpsout, init_hstorage_right(mpsout, mpo, mps))
	mult_alg = get_iterative_alg(alg)
	kvals = compute!(m, mult_alg)
	return m.omps, kvals[end]
end
