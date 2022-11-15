abstract type AbstractMPSArith end
abstract type AbstractMPOMPSMultCache end


function iterative_compute!(m, alg, args...)
    kvals = Float64[]
    iter = 0
    tol = 1.
    while (iter < alg.maxiter) && (tol >= alg.tol)
        tol = sweep!(m, alg, args...)
        push!(kvals, tol)
        iter += 1
        (alg.verbosity > 1) && println("finish the $iter-th sweep with error $tol", "\n")
    end
    if (alg.verbosity >= 2) && (iter < alg.maxiter)
        println("early converge in $iter-th sweeps with error $tol")
    end
    if (alg.verbosity > 0) && (tol >= alg.tol)
        println("fail to converge, required precision: $(alg.tol), actual precision $tol in $iter sweeps.")
    end
    return kvals
end
