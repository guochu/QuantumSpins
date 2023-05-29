const LANCZOS_ORTH_TOL = 1.0e-12

function find_lanczos(w::Vector{<:Real}, wh::String)
	isempty(w) && error("input vector must not be empty")
	if wh == "LM"
	    return argmax(abs.(w))
	elseif wh == "SM"
		return argmin(abs.(w))
	elseif (wh == "SR" || wh == "SA")
		return 1
	else
		(wh == "SR" || wh == "SA") || error("which is only allowed to be 'LM', 'SM' or 'SR'")
		return length(w)
	end
end

# ****************************************************************
# first step of lanczos iteration. v is the initial vector.
# the output a, b, are the first element of alpha, beta. qiminus1
# is the normalized initial vector, and q1 is the second lanczos
# vector
# the norm of initial input vector is outputed as vm
# ****************************************************************
function lanczos_make_first_step(A, v)
	vm = norm(v)
	qiminus1 = v / vm
	qi = A(qiminus1)
	a = dot(qiminus1, qi)
	if abs(imag(a)) > LANCZOS_ORTH_TOL
		@warn "imaginary part the a is very large($(abs(imag(a))))"
	end
	a = real(a)
	qi -= qiminus1 * a
	b = norm(qi)
	if b != 0.
	    qi /= b
	end
	return a, b, qiminus1, qi, vm
end

# ****************************************************************
# Assume we are in the i-th iteration. Then the input
# qiminus1 = V[i-1], qi=V[i], bold = beta[i-1],
# the output are alpha[i+1], beta[i], V[i+1], V[i]
# ***************************************************************
function lanczos_make_step(A, qiminus1, qi, bold)
	qiplus1 = A(qi)
	# println(qi)
	# println(qiplus1)
	# println("----------------------------")
	a = dot(qi, qiplus1)

	if abs(imag(a)) > LANCZOS_ORTH_TOL
		@warn "imaginary part the a is very large($(abs(imag(a))))"
	end
	a = real(a)
	qiplus1 -= (qi*a + qiminus1 * bold)
	b = norm(qiplus1)
	if b != 0.
	    qiplus1 /= b
	end
	qiminus1 = qi
	qi = qiplus1
	return a, b, qiminus1, qi
end

function lanczos_linear_transformation_single(singleTvector, Rvectors)
	isempty(Rvectors) && error("Rvectors is empty")
	nR = length(Rvectors)
	evec = Rvectors[end]*singleTvector[end]
	for i=1:(nR-1)
	    evec += Rvectors[i]*singleTvector[i]
	end
	return evec
end

"""
	simple lanczos solver for the lowest eigenpair
"""
function simple_lanczos_solver(A, v, wh::String="SA", kmax::Int=10, btol::Real=1.0e-8; verbosity::Int=0)
	info = 0
	a, b, qiminus1, qi, vm = lanczos_make_first_step(A, v)
	if b < btol
	    (verbosity > 3) && println("lanczos converged after the first iteration")
	    return a, qiminus1, info
	end
	V = Vector{Any}()
	push!(V, qiminus1)
	T = zeros(Float64, kmax, kmax)
	T[1,1] = a
	k = 1
	converged = false
	# @printf("%s, %s.\n", a, b)
	while k < kmax
	    T[k, k+1] = b
	    T[k+1, k] = b
	    a, b, qiminus1, qi = lanczos_make_step(A, qiminus1, qi, b)
	    push!(V, qiminus1)
	    T[k+1, k+1] = a
	    k += 1
	    if b < btol
	    	(verbosity > 3) && println("lanczos converges after $k iterations")
	        converged = true
	        break
	    end
	end
	if !converged
	    info = -1
	    (verbosity > 3) && println("lanczos fail to converge after $kmax iterations")
	end
	# println(T[1:k, 1:k])
	F = eigen(Symmetric(T[1:k, 1:k]))
	eigval = F.values
	eigvec = F.vectors
	pos = find_lanczos(eigval, wh)
	return eigval[pos], lanczos_linear_transformation_single(eigvec[:, pos], V), info
end
