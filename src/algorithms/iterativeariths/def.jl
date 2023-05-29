abstract type AbstractMPSArith end

@with_kw struct OneSiteIterativeArith <: AbstractMPSArith
	D::Int = 100 
	maxiter::Int = 5
	verbosity::Int = 1
	tol::Float64 = 1.0e-8
end

IterativeArith(;kwargs...) = OneSiteIterativeArith(; kwargs...) 


changeD(x::OneSiteIterativeArith; D::Int) = OneSiteIterativeArith(D=D, maxiter=x.maxiter, verbosity=x.verbosity, tol=x.tol)


@with_kw struct SVDArith <: AbstractMPSArith
	D::Int = 100
	ϵ::Float64 = 1.0e-8
	verbosity::Int = 1
end

changeD(x::SVDArith; D::Int) = SVDArith(D=D, ϵ=x.ϵ, verbosity=x.verbosity)

get_trunc(x::SVDArith) = MPSTruncation(D=x.D, ϵ=x.ϵ)

@with_kw struct OneSiteStableArith <: AbstractMPSArith
	D::Int = 100
	ϵ::Float64 = 1.0e-8
	maxiter::Int = 5
	verbosity::Int = 1
	tol::Float64 = 1.0e-8
end


StableArith(; kwargs...) = OneSiteStableArith(; kwargs...) 

changeD(x::OneSiteStableArith; D::Int) = OneSiteStableArith(D=D, maxiter=x.maxiter, ϵ=x.ϵ, verbosity=x.verbosity, tol=x.tol)


get_svd_alg(x::OneSiteStableArith) = SVDArith(D=x.D, ϵ=x.ϵ, verbosity=x.verbosity)
get_iterative_alg(x::OneSiteStableArith) = OneSiteIterativeArith(D=x.D, maxiter=x.maxiter, tol=x.tol, verbosity=x.verbosity)


function iterative_compute!(m, alg, args...)
    kvals = Float64[]
    iter = 0
    tol = 1.
    while (iter < alg.maxiter) && (tol >= alg.tol)
        _kvals = sweep!(m, alg, args...)
        tol = iterative_error_2(_kvals)
        push!(kvals, tol)
        iter += 1
        (alg.verbosity > 1) && println("finish the $iter-th sweep with error $tol", "\n")
    end
    if (alg.verbosity >= 2) && (iter < alg.maxiter)
        println("early converge in $iter-th sweeps with error $tol")
    end
    if (alg.verbosity > 0) && (tol >= alg.tol)
        println("fail to converge, required precision: $(alg.tol), actual precision $tol in $iter sweeps")
    end
    return kvals
end

iterative_error_2(m::AbstractVector) = std(m) / abs(mean(m))
