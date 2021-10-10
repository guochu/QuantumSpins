


struct StateVector{T <: Number} 
	data::Vector{T}
	nqubits::Int

function StateVector{T}(data::AbstractVector{<:Number}, nqubits::Int) where {T <: Number}
	(length(data) == 2^nqubits) || throw(DimensionMismatch())
	new{T}(convert(Vector{T}, data), nqubits)
end

end

StateVector(data::AbstractVector{T}, nqubits::Int) where {T <: Number} = StateVector{T}(data, nqubits)
StateVector{T}(data::AbstractVector{<:Number}) where T = StateVector{T}(data, round(Int, log2(length(data))))
StateVector(data::AbstractVector{T}) where {T <: Number} = StateVector{T}(data)
function StateVector{T}(nqubits::Int) where {T<:Number} 
	v = zeros(T, 2^nqubits)
	v[1] = 1
	return StateVector{T}(v, nqubits)
end
StateVector(::Type{T}, nqubits::Int) where {T<:Number} = StateVector{T}(nqubits)
StateVector(nqubits::Int) = StateVector(ComplexF64, nqubits)


scalar_type(::Type{StateVector{T}}) where T = T
scalar_type(x::StateVector) = scalar_type(typeof(x))

Base.eltype(::Type{StateVector{T}}) where T = T
Base.eltype(x::StateVector) = eltype(typeof(x))

raw_data(x::StateVector) = x.data
nqubits(x::StateVector) = x.nqubits


LinearAlgebra.norm(x::StateVector) = norm(raw_data(x))
LinearAlgebra.dot(x::StateVector, y::StateVector) = dot(raw_data(x), raw_data(y))


distance2(x::StateVector, y::StateVector) = _distance2(x, y)
distance(x::StateVector, y::StateVector) = _distance(x, y)


# probs(x::StateVector) = abs2.(raw_data(x))


function StateVector(psi::MPS{T}) where T
	for item in raw_data(psi)
		(size(item, 2) == 2) || throw(ArgumentError("physical dimension should be 2."))
	end
	L = length(psi)
	isempty(psi) && return StateVector(T, L)
	m = reshape(psi[1], 2, size(psi[1], 3))
	for i in 2:L
		@tensor tmp[-1, -2, -3] := m[-1, 1] * mps[i][1, -2, -3]
		m = tie(tmp, (2, 1))
	end
	return StateVector{T}(reshape(m, length(m)), L)
end
function MPS(psi::StateVector{T}; trunc::TruncationScheme=DefaultTruncation) where T
	r = Vector{Array{T, 3}}(undef, nqubits(psi))
	(nqubits(psi) == 0) && return MPS(r)
	v = raw_data(psi)
	m = reshape(v, Tuple([2 for i in 1:nqubits(psi)]))
	u, s, v, bet = tsvd!(m, (1,), Tuple(2:L), trunc=trunc)
	r[1] = reshape(u, 1, size(u)...)
	v = reshape(Diagonal(s) * tie(v, (1, ndims(v)-1)), size(v))
	for i in 2:L
		u, s, v, bet = tsvd!(v, (1,2), Tuple(3:ndims(v)), trunc=trunc)
		r[i] = u
		v = reshape(Diagonal(s) * tie(v, (1, ndims(v)-1)), size(v))
	end
	r[L] = reshape(v, shape(v)..., 1)
	mp = MPS(r)
	rightorth!(mp, trunc=trunc)
	return mp
end