


function n_qubits_mat_from_external(m::AbstractMatrix)
	(size(m, 1) == size(m, 2)) || error("input matrix is not a square matrix.")
	L = size(m, 1)
	n = floor(Int, log2(L))
	(2^n == L) || error("input matrix size error.")
	v = collect(n:-1:1)
	v = vcat(v, v .+ n)
	return permute(reshape(m, Tuple(2 for i in 1:2*n)), v)
end

_is_unitary(m::AbstractMatrix) = isapprox(m' * m, I) 

"""
	from_external(key::NTuple{N, Int}, m::AbstractMatrix; check_unitary::Bool=true) where N
Initialize a quantum gate from an external matrix built using kron
"""
function from_external(key::NTuple{N, Int}, m::AbstractMatrix; check_unitary::Bool=true) where N
	(size(m, 1) == 2^N) || error("wrong input matrix size.")
	if check_unitary
	    _is_unitary(m) || error("input matrix is not unitary.")
	end
	return QuantumGate(key, n_qubits_mat_from_external(m))
end  
from_external(key::Int, m::AbstractMatrix; kwargs...) = from_external((key,), m; kwargs...)

Gate(key::NTuple{N, Int}, m::AbstractMatrix) where N = QuantumGate(key, reshape(m, ntuple(i->2, 2*N)))
Gate(key::Int, m::AbstractMatrix) = QuantumGate(key, m)
Gate(key::NTuple{N, Int}, m::AbstractArray{T, M}) where {N, T, M} = QuantumGate(key, m)
Gate(key::NTuple{1, Int}, m::AbstractMatrix) = QuantumGate(key[1], m)
