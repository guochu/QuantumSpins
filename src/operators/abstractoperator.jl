
abstract type AbstractOperator end
abstract type AbstractQuantumOperator <: AbstractOperator end
abstract type AbstractSuperOperator <: AbstractOperator end


raw_data(x::AbstractQuantumOperator) = x.data
raw_data(x::AbstractSuperOperator) = raw_data(x.data)

scalar_type(x::AbstractOperator) = scalar_type(typeof(x))

Base.empty!(s::AbstractOperator) = empty!(raw_data(s))
Base.isempty(x::AbstractOperator) = isempty(raw_data(x))
Base.keys(x::AbstractOperator) = keys(raw_data(x))
physical_dimensions(x::AbstractQuantumOperator) = convert(Vector{Int}, x.physpaces)
physical_dimensions(x::AbstractSuperOperator) = physical_dimensions(x.data)


Base.length(x::AbstractQuantumOperator) = length(x.physpaces)
Base.length(x::AbstractSuperOperator) = length(x.data)

function is_constant(x::AbstractOperator)
	for (k, v) in raw_data(x)
		for (m, c) in v
			is_constant(c) || return false
		end
	end
	return true
end

interaction_range(x::AbstractOperator) = maximum([_interaction_range(k) for k in keys(x)])


Base.:*(y::AllowedCoefficient, x::AbstractOperator) = x * y
Base.:/(x::AbstractOperator, y::AllowedCoefficient) = x * (1 / Coefficient(y))
Base.:+(x::AbstractOperator) = x
Base.:-(x::AbstractOperator) = (-1) * x 
