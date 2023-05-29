
Base.eltype(x::AbstractQuantumGate) = eltype(typeof(x))


function _get_norm_order(key::NTuple{N, Int}, p) where N
	seq = sortperm([key...])
	perm = (seq..., [s + N for s in seq]...)
	return key[seq], permute(p, perm)
end
_shift(key::NTuple{N, Int}, i::Int) where N = NTuple{N, Int}(l+i for l in key)

const MAXIMUM_GATE_RANGE = 4

"""
	struct QuantumGate{N, M<:AbstractArray} <: AbstractQuantumGate{N}
"""
struct QuantumGate{N, M<:AbstractArray} <: AbstractQuantumGate{N}
	positions::NTuple{N, Int}
	op::M

function QuantumGate(positions::NTuple{N, Int}, m::AbstractArray{T, K}) where {N, T, K}
	(K == 2 * N) || throw(ArgumentError("input array rank mismatch with positions."))
	(N <= MAXIMUM_GATE_RANGE) || throw(ArgumentError("only less than $MAXIMUM_GATE_RANGE-body gate supported."))
	positions, m = _get_norm_order(positions, m)
	new{N, typeof(m)}(positions, m)
end

end
QuantumGate(positions::Vector{Int}, m::AbstractArray) = QuantumGate(Tuple(positions), m)
QuantumGate(positions::Int, m::AbstractMatrix) = QuantumGate((positions,), m)


positions(x::AbstractQuantumGate) = x.positions
op(x::QuantumGate) = x.op
Base.eltype(::Type{QuantumGate{N, M}}) where {N, M} = eltype(M)
shift(x::QuantumGate, i::Int) = QuantumGate(_shift(positions(x), i), op(x))



function _get_trans_perm(N::Int)
	v = collect(1:N)
	return vcat(v .+ N, v)
end

struct AdjointQuantumGate{N, G<:AbstractQuantumGate{N}} <: AbstractQuantumGate{N}
	parent::G
end

positions(x::AdjointQuantumGate) = positions(x.parent)
op(s::AdjointQuantumGate{N, G}) where {N, G} = permute(conj(op(s.parent)), _get_trans_perm(N))
Base.eltype(::Type{AdjointQuantumGate{N, G}}) where {N, G} = eltype(G)
shift(x::AdjointQuantumGate) = AdjointQuantumGate(shift(x.parent))

Base.adjoint(x::AbstractQuantumGate) = AdjointQuantumGate(x)
Base.adjoint(x::AdjointQuantumGate) = x.parent

