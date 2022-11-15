# mpo deparallelisation


function deparallelise_left!(x::AbstractMPO; tol::Real=DeparalleliseTol, verbosity::Int=0)
	for i = 1:(length(x)-1)
		(verbosity > 3) && println("deparallelisation sweep from left to right on site: $i.")
		M, Tm = left_deparallelise(x[i], (1,2,4), (3,); tol=tol, verbosity=verbosity)
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
	    (verbosity > 3) && println("deparallelisation sweep from right to left on site: $i.")
	    Tm, M = right_deparallelise(x[i], (1,), (2,3,4); tol=tol, verbosity=verbosity)
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


abstract type AbstractCompression end


@with_kw struct SVDCompression <: AbstractCompression
	D::Int = Defaults.D
	tol::Float64 = DeparalleliseTol
	verbosity::Int = Defaults.verbosity
end

@with_kw struct Deparallelise <: AbstractCompression
	tol::Float64 = DeparalleliseTol
	verbosity::Int = Defaults.verbosity
end

get_trunc(alg::SVDCompression) = MPSTruncation(D=alg.D, ϵ=alg.tol)

_compress!(h::MPO, alg::Deparallelise) = deparallelise!(h, tol=alg.tol, verbosity=alg.verbosity)

function _compress!(h::MPO, alg::SVDCompression)
	L = length(h)
	(L > 1) || error("number of input mpo must be larger than 1.")
	Errs = Float64[]
	trunc = get_trunc(alg)
	# we basicaly do not use a hard truncation D for MPO truncation
	for i in 1:L-1
		(alg.verbosity > 3) && println("sweeping from left to right at site: $i.")
		q, r = tqr!(h[i], (1,2,4), (3,))
		h[i] = permute(q, (1,2), (4,3))
		h[i+1] = @tensor tmp[1 3; 4 5] := r[1,2] * h[i+1][2,3,4,5]
	end
	for i in L:-1:2
		(alg.verbosity > 3) && println("sweeping from right to left at site: $i.")
		u, s, v, err = tsvd!(h[i], (1,), (2,3,4), trunc=trunc)
		h[i] = permute(reshape(Diagonal(s) * tie(v, (1,3)), size(v)), (1,2), (3,4))
		h[i-1] = @tensor tmp[-1 -2; -3 -4] := h[i-1][-1, -2, 1, -4] * u[1, -3]
		push!(Errs, err)
	end
	n = norm(h[1])
	if (n ≈ zero(n))
		@warn "mpo norm becomes 0 after compression."
		return h
	end
	h[1] /= n
	factor = n^(1 / L)
	for i in 1:L
		h[i] *= factor
	end
	(alg.verbosity > 2) && println("maximum truncation error is $(maximum(Errs)).")
	return h
end

default_mpo_compression() = Deparallelise(tol=DeparalleliseTol) 

compress!(h::MPO, alg::AbstractCompression) = _compress!(h, alg)
compress!(h::MPO; alg::AbstractCompression=default_mpo_compression()) = compress!(h, alg)

