
abstract type AbstractTerm end
abstract type AbstractQuantumTerm <: AbstractTerm end
abstract type AbstractSuperTerm <: AbstractTerm end


positions(x::AbstractQuantumTerm) = x.positions
op(x::AbstractQuantumTerm) = x.op
coeff(x::AbstractQuantumTerm) = x.coeff

positions(x::AbstractSuperTerm) = positions(x.data)
op(x::AbstractSuperTerm) = op(x.data)
coeff(x::AbstractSuperTerm) = coeff(x.data)


space_l(x::AbstractTerm) = size(op(x)[1], 1)
space_r(x::AbstractTerm) = size(op(x)[end], 3)

Base.isempty(x::AbstractTerm) = isempty(op(x))

Base.:*(m::AllowedCoefficient, s::AbstractTerm) = s * m
Base.:/(s::AbstractTerm, m::AllowedCoefficient) = s * (1 / Coefficient(m))
Base.:+(s::AbstractTerm) = s
Base.:-(s::AbstractTerm) = (-1) * s 

nterms(s::AbstractTerm) = length(op(s))
is_constant(s::AbstractTerm) = is_constant(coeff(s))
function scalar_type(x::AbstractTerm)
	T = scalar_type(coeff(x))
	for m in op(x)
		T = promote_type(T, eltype(m))
	end
	return T
end 

function _interaction_range(x::Union{Vector{Int}, Tuple})::Int
	(length(x) == 0) && return 0
	(length(x)==1) && return 1
	return x[end] - x[1] + 1
end
interaction_range(x::AbstractTerm) = _interaction_range(positions(x))

function is_zero(x::AbstractTerm) 
	is_zero(coeff(x)) && return true
	isempty(x) && return true
	for item in op(x)
	    is_zero(item) && return true
	end
	return false
end

isstrict(t::AbstractTerm) = (space_l(t) == space_r(t) == 1)

bond_dimension(h::AbstractTerm, bond::Int) = begin
	((bond >= 1) && (bond <= nterms(h))) || throw(BoundsError())
	size(op(h)[bond], 3)
end 
bond_dimensions(h::AbstractTerm) = [bond_dimension(h, i) for i in 1:nterms(h)]
bond_dimension(h::AbstractTerm) = maximum(bond_dimensions(h))

ophysical_dimensions(psi::AbstractTerm) = [size(item, 2) for item in op(psi)]
iphysical_dimensions(psi::AbstractTerm) = [size(item, 4) for item in op(psi)]

function physical_dimensions(psi::AbstractTerm)
	xs = ophysical_dimensions(psi)
	(xs == iphysical_dimensions(psi)) || error("i and o physical dimension mismatch.")
	return xs
end


