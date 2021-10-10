# mpo deparallelisation


function deparallelise_left!(x::AbstractMPO; tol::Real=DeparalleliseTol, verbosity::Int=0)
	for i = 1:(length(x)-1)
		(verbosity > 2) && println("deparallelisation sweep from left to right on site: $i.")
		# M, Tm = deparallelise(x[i], (3,), tol, verbosity=verbosity)
		M, Tm = deparallelise(x[i], (1,2,4), (3,); row_or_col=:col, tol=tol, verbosity=verbosity)
		if length(M) > 0
		    x[i] = permute(M, (1,2), (4,3))
		    @tensor tmp[-1 -2; -3 -4] := Tm[-1, 1] * x[i+1][1,-2,-3,-4]
		    x[i+1] = tmp
		else
			(verbosity >= 1) && println("mpo becomes empty after deparallelisation left.")
			return nothing
		end
	end
	return x
end

function deparallelise_right!(x::AbstractMPO; tol::Real=DeparalleliseTol, verbosity::Int=0)
	for i = length(x):-1:2
	    (verbosity > 2) && println("deparallelisation sweep from right to left on site: $i.")
	    Tm, M = deparallelise(x[i], (1,), (2,3,4); row_or_col=:row, tol=tol, verbosity=verbosity)
	    if length(M) > 0
	    	# ii = isomorphism()
	        x[i] = permute(M, (1,2), (3,4))
	        @tensor tmp[-1 -2; -3 -4] := x[i-1][-1,-2,1,-4] * Tm[1,-3]
	        x[i-1] = tmp
	    else
	    	(verbosity >= 1) && println("mpo becomes empty after deparallelisation right.")
	    	return nothing
	    end
	end
	return x
end

function deparallelise!(x::AbstractMPO; kwargs...)
	r = deparallelise_left!(x; kwargs...)
	if !isnothing(r)
		r = deparallelise_right!(x; kwargs...)
	end
	return r
end

"""
	deparallelise(x::AbstractMPO)
	reduce the bond dimension of mpo using deparallelisation
"""
deparallelise(x::AbstractMPO; kwargs...) = deparallelise!(copy(x); kwargs...)


abstract type MPOCompression end

function svdcompress!(h::MPO; tol::Real=DeparalleliseTol, verbosity::Int=0)
	L = length(h)
	(L > 1) || error("number of input mpo must be larger than 1.")
	Errs = Float64[]
	# we basicaly do not use a hard truncation D for MPO truncation
	trunc = MPSTruncation(D=10000, ϵ=tol)
	for i in 1:L-1
		u, s, v, err = tsvd!(h[i], (1,2,4), (3,), trunc=trunc)
		h[i] = permute(reshape(tie(u, (3,1)) * Diagonal(s), size(u)), (1,2), (4,3))
		h[i+1] = @tensor tmp[-1 -2; -3 -4] := v[-1, 1] * h[i+1][1,-2,-3,-4]
		push!(Errs, err)
	end
	for i in L:-1:2
		u, s, v, err = tsvd!(h[i], (1,), (2,3,4), trunc=trunc)
		h[i] = permute(reshape(Diagonal(s) * tie(v, (1,3)), size(v)), (1,2), (3,4))
		h[i-1] = @tensor tmp[-1 -2; -3 -4] := h[i-1][-1, -2, 1, -4] * u[1, -3]
		push!(Errs, err)
	end
	(verbosity >= 2) && println("maximum truncation error is $(maximum(Errs)).")
	return h
end


@with_kw struct SVDCompression <: MPOCompression
	tol::Float64 = DeparalleliseTol
	verbosity::Int = Defaults.verbosity
end

@with_kw struct Deparallelise <: MPOCompression
	tol::Float64 = DeparalleliseTol
	verbosity::Int = Defaults.verbosity
end


_compress!(h::MPO, alg::Deparallelise) = deparallelise!(h, tol=alg.tol, verbosity=alg.verbosity)
_compress!(h::MPO, alg::SVDCompression) = svdcompress!(h, tol=alg.tol, verbosity=alg.verbosity)
compress!(h::MPO; alg::MPOCompression=Deparallelise()) = _compress!(h, alg)

