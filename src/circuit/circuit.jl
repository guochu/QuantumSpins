



struct QuantumCircuit <: AbstractQuantumCircuit
	data::Vector{AbstractQuantumGate{N} where N}
end

QuantumCircuit() = QuantumCircuit(Vector{AbstractQuantumGate{N} where N}())

raw_data(x::QuantumCircuit) = x.data

Base.IteratorSize(::QuantumCircuit) = Base.HasLength()
Base.IteratorEltype(::AbstractQuantumCircuit) = Base.HasEltype()
Base.eltype(::Type{QuantumCircuit}) = AbstractQuantumGate{N} where N

Base.getindex(x::QuantumCircuit, i::Int) = getindex(raw_data(x), i)
Base.setindex!(x::QuantumCircuit, v,  i::Int) = setindex!(raw_data(x), v, i)
Base.length(x::QuantumCircuit) = length(raw_data(x))
Base.size(x::QuantumCircuit) = size(raw_data(x))
Base.iterate(x::QuantumCircuit) = iterate(raw_data(x))
Base.iterate(x::QuantumCircuit, state) = iterate(raw_data(x), state)
Base.eltype(x::QuantumCircuit) = eltype(raw_data(x))
Base.empty!(x::QuantumCircuit) = empty!(raw_data(x))

Base.isempty(x::QuantumCircuit) = isempty(raw_data(x))
Base.firstindex(x::QuantumCircuit) = firstindex(raw_data(x))
Base.lastindex(x::QuantumCircuit) = lastindex(raw_data(x))
Base.reverse(x::QuantumCircuit) = QuantumCircuit(reverse(raw_data(x)))
Base.repeat(x::QuantumCircuit, n::Int) = QuantumCircuit(repeat(raw_data(x), n))


Base.push!(x::QuantumCircuit, s::AbstractQuantumGate) = push!(raw_data(x), s)
Base.append!(x::QuantumCircuit, y::QuantumCircuit) = append!(raw_data(x), raw_data(y))
Base.append!(x::QuantumCircuit, y::Vector{<:AbstractQuantumGate}) = append!(raw_data(x), y)

Base.adjoint(x::QuantumCircuit) = QuantumCircuit(Vector{AbstractQuantumGate{N} where N}([adjoint(x[i]) for i in length(x):-1:1]))


Base.similar(x::QuantumCircuit) = QuantumCircuit()
Base.copy(x::QuantumCircuit) = QuantumCircuit(copy(raw_data(x)))

Base.:*(x::QuantumCircuit, y::QuantumCircuit) = QuantumCircuit(vcat(raw_data(y), raw_data(x)))

function scalar_type(x::QuantumCircuit)
	T = Float64
	for gate in x
		T = promote_type(T, scalar_type(gate))
	end
	return T
end