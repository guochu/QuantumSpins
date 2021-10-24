

# initializer
statevector_mps(::Type{T}, states::Vector{Int}) where {T<:Number} = prodmps(T, [2 for i in 1:length(states)], states)
statevector_mps(::Type{T}, n::Int) where {T <: Number} = statevector_mps(T, zeros(Int, n))
statevector_mps(n::Int) = statevector_mps(ComplexF64, n)



# measurement
const MPS_PHYSICAL_PROBABILITY_CHECK_TOL = 1.0e-6

function discrete_sample(l::Vector{Float64})
	isempty(l) && error("no results.")
	s = sum(l)
	L = length(l)
	l1 = Vector{Float64}(undef, L+1)
	l1[1] = 0
	for i=1:L
	    l1[i+1] = l1[i] + l[i]/s
	end
	s = rand(Float64)
	for i = 1:L
	    if (s >= l1[i] && s < l1[i+1])
	        return i
	    end
	end
	return L
end

function _local_measure(svecleft::AbstractArray{Float64, 1}, mpsj::AbstractArray{T, 3}, basis::AbstractMatrix) where {T}
	mpsjnew1 = contract(basis, mpsj, ((1,), (2,)))
	mpsjnew2 = contract(diag(svecleft), mpsjnew1, ((2,), (2,)))
	d = size(mpsjnew2, 2)
	l = Vector{Float64}()
	tol = 1.0e-6
	for i = 1:d
		tmp = mpsjnew2[:, i, :]
		s = dot(tmp, tmp)
		push!(l, real(s))
	end
	l ./= sum(l)
	i = discrete_sample(l)
	return mpsjnew1[i,:,:], l, i
end

function _measure_one_site(mps::AbstractMPS, site::Int, basis::AbstractMatrix) 
	L = length(mps)
	# assert(allclose(basis.T.conj().dot(basis), eye(basis.shape[0])))
	m, l, si = _local_measure(mps.svectors[site], mps[site], basis)
	s = l[si]
	m = m / sqrt(s)
	if L==1
		length(m)==1 || error("somthing went wrong in measurement.")
		return nothing, si, s
	end
	mpsout = typeof(mps)(L-1)
	if site==1
		mpsj = contract(m, mps[2], ((2,), (1,)))
		mpsout[site] = mpsj
		for i = (site+1):(L-1)
			mpsout[i] = mps[i+1]
		end
		maybe_init_boundary_s!(mpsout)
		mpsout.s[2:end] = mps.s[3:end]
	else
		mpsj = contract(mps[site-1], m, ((3,), (1,)))
		for i=1:(site-2)
			mpsout[i] = mps[i]
		end
		mpsout[site-1] = mpsj
		for i = site:(L-1)
			mpsout[i] = mps[i+1]
		end
		maybe_init_boundary_s!(mpsout)
		mpsout.s[2:(site-1)] = mps.s[2:(site-1)]
		mpsout.s[site:end] = mps.s[(site+1):end]
	end
	return mpsout, si, s
end


function measure_and_throw(s::QMeasure, state::AbstractMPS; trunc::TruncationScheme=DefaultTruncation)
	if svectors_uninitialized(state)
	    canonicalize!(state, normalize=false, trunc=trunc)
	end
	mpsout, si, probability = _measure_one_site(state, s.position, _eye(2))
	return mpsout, si, probability
end

function _physical_probability(probability::Real)
	if probability > 1
	    (probability - 1) < MPS_PHYSICAL_PROBABILITY_CHECK_TOL || error("probability $probability larger than 1, result is inaccurate.")
	    probability = 1.
	end
	return probability
end

function apply!(s::QMeasure, state::AbstractMPS; trunc::TruncationScheme=DefaultTruncation)
	pos = s.position
	if s.keep
		op = _eye(2)
		m, l, si = _local_measure(state.s[pos], state[pos], op)
		probability = l[si]
		op[3-si, 3-si] = 0
		if s.auto_reset && (si==2)
			op = X * op
		end
		state[pos] = permute(contract(op, state[pos], ((2,), (2,))), (2,1,3))
		state[pos] /= sqrt(probability)
		# it is really necessary to prepare ? it seems to be...
		canonicalize!(state, normalize=false)
	else
		mpsout, si, probability = measure_and_throw(s, state, trunc=trunc)
		if isnothing(mpsout)
			empty!(raw_data(state))
			empty!(raw_singular_matrices(state))
		else
			pop!(raw_data(state))
			pop!(raw_singular_matrices(state))
			for i = 1:length(mpsout)
			    state[i] = mpsout[i]
			    state.s[:] = mpsout.s[:]
			end
		end
	end
	return si-1, _physical_probability(probability)
end

measure!(state::AbstractMPS, pos::Int; keep::Bool=true, auto_reset::Bool=true, trunc::TruncationScheme=DefaultTruncation) = apply!(
	QMeasure(pos, auto_reset=auto_reset, keep=keep), state, trunc=trunc)

function amplitude(s::AbstractMPS, basis::Vector{Int}; scaling::Real=sqrt(2))
	length(basis)==length(s) || throw(DimensionMismatch())
	for item in basis
	    (item==0 || item==1) || throw(ArgumentError("the qubit should either be 0 or 1."))
	end
	ps = statevector_mps(scalar_type(s), basis)
	if scaling != 1.
	    for i in 1:length(ps)
	        ps[i] = scaling * ps[i]
	    end
	end
	return dot(ps, s)
end
