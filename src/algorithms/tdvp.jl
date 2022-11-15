abstract type TDVPAlgorithm end

"""
	struct TDVP{T<:Number} <: TDVPAlgorithm

TDVP algorithm
"""
@with_kw struct TDVP{T<:Number} <: TDVPAlgorithm
	stepsize::T 
	D::Int = Defaults.D
	ishermitian::Bool = false
	verbosity::Int = Defaults.verbosity
end

# TDVP(; stepsize::Number, D::Int=Defaults.D, ishermitian::Bool=false, verbosity::Int=1) = TDVP(stepsize, D, ishermitian, verbosity)

function _leftsweep!(m::ExpectationCache, alg::TDVP)
	increase_bond!(m, D=alg.D)
	mpo = m.mpo
	mps = m.mps
	hstorage = m.env
	dt = alg.stepsize	
	isherm = alg.ishermitian
	driver = isherm ? Lanczos() : Arnoldi()

	workspace = promote_type(scalar_type(mpo), scalar_type(mps))[]
	for site in 1:length(mps)-1
		(alg.verbosity > 3) && println("sweeping from left to right at site: $site.")
		tmp, info = exponentiate(x->ac_prime(x, mpo[site], hstorage[site], hstorage[site+1]), dt/2, mps[site], driver)

		mps[site], v = tqr!(tmp, (1,2), (3,), workspace)
		hnew = updateleft(hstorage[site], mps[site], mpo[site], mps[site])

		v, info = exponentiate(x->c_prime(x, hnew, hstorage[site+1]), -dt/2, v, driver)
		mps[site+1] = @tensor tmp[-1 -2; -3] := v[-1, 1] * mps[site+1][1, -2, -3]
		hstorage[site+1] = hnew
	end
	site = length(mps)
	mps[site], info = exponentiate(x->ac_prime(x, mpo[site], hstorage[site], hstorage[site+1]), dt, mps[site], driver)
end

function _rightsweep!(m::ExpectationCache, alg::TDVP)
	mpo = m.mpo
	mps = m.mps
	hstorage = m.env
	dt = alg.stepsize	
	isherm = alg.ishermitian
	driver = isherm ? Lanczos() : Arnoldi()
	
	workspace = promote_type(scalar_type(mpo), scalar_type(mps))[]
	for site in length(mps)-1:-1:1
		(alg.verbosity > 3) && println("sweeping from right to left at site: $site.")

		v, Q = tlq!(mps[site+1], (1,), (2,3), workspace) 
		mps[site+1] = Q
		hnew = updateright(hstorage[site+2], mps[site+1], mpo[site+1], mps[site+1])

		v, info = exponentiate(x->c_prime(x, hstorage[site+1], hnew), -dt/2, v, driver)
		hstorage[site+1] = hnew
		mps[site] = @tensor tmp[-1, -2; -3] := mps[site][-1,-2,1] * v[1,-3]

		mps[site], info = exponentiate(x->ac_prime(x, mpo[site], hstorage[site], hstorage[site+1]), dt/2, mps[site], driver)
	end
end



function sweep!(m::ExpectationCache, alg::TDVPAlgorithm; kwargs...)
	_leftsweep!(m, alg; kwargs...)
	_rightsweep!(m, alg; kwargs...)
end



