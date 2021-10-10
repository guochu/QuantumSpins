# I support time-dependent quantum operator by using this class
abstract type AbstractCoefficient end

struct Coefficient{T} <: AbstractCoefficient
	value::T

	Coefficient(x::Function) = begin
	    m = x(1.)
	    isa(m, Number) || error("wrong Coefficient Function.")
	    new{typeof(x)}(x)
	end
	Coefficient(x::Number) = new{typeof(x)}(x)

	Coefficient{T}(x::Union{Number, Function}) where T = new{T}(x)
	Coefficient{T}(x::Coefficient) where T = new{T}(value(x))
end

# value(x::Number) = x
value(x::Coefficient) = x.value
Coefficient(x::Coefficient) = Coefficient(value(x))
Base.copy(x::Coefficient) = Coefficient(value(x))
coeff(x::Union{Number, Function, Coefficient}) = Coefficient(x)

Base.:*(x::Coefficient{<:Number}, y::Number) = Coefficient(value(x) * y)
Base.:*(x::Coefficient{<:Function}, y::Number) = Coefficient(z->value(x)(z)*y)
Base.:*(y::Number, x::Coefficient) = x * y
Base.:*(x::Coefficient{<:Number}, y::Coefficient{<:Number}) = Coefficient(value(x) * value(y))
Base.:*(x::Coefficient{<:Function}, y::Coefficient{<:Number}) = x * value(y)
Base.:*(y::Coefficient{<:Number}, x::Coefficient{<:Function}) = x * y
Base.:*(x::Coefficient{<:Function}, y::Coefficient{<:Function}) = Coefficient(z->value(x)(z) * value(y)(z))

Base.:/(x::Coefficient{<:Number}, y::Number) = Coefficient(value(x) / y)
Base.:/(x::Coefficient{<:Function}, y::Number) = Coefficient(z->value(x)(z)/y)
Base.:/(x::Number, y::Coefficient{<:Number}) = Coefficient(x / value(y))
Base.:/(x::Number, y::Coefficient{<:Function}) = Coefficient(z->x/(value(y)(z)))

Base.:-(x::Coefficient{<:Number}) = Coefficient(-value(x))
Base.:-(x::Coefficient{<:Function}) = Coefficient(z->-value(x)(z))

# only implemented those I need, but this can be far general
Base.conj(x::Coefficient{<:Number}) = Coefficient(conj(value(x)))
Base.conj(x::Coefficient{<:Function}) = Coefficient(z->conj(value(x)(z)))
Base.sqrt(x::Coefficient{<:Number}) = Coefficient(sqrt(value(x)))
Base.sqrt(x::Coefficient{<:Function}) = Coefficient(z->sqrt(value(x)(z)))

(op::Coefficient{<:Number})(t::Number) = value(op)
(op::Coefficient{<:Function})(t::Number) = value(op)(t)

"""
	is_constant(x)
	check if the object x is a pure constant or contrans function 
"""
is_constant(x::Coefficient{<:Number}) = true
is_constant(x::Coefficient{<:Function}) = false


Base.:(==)(x::Coefficient, y::Coefficient) = false
Base.:(==)(x::Coefficient{T}, y::Coefficient{T}) where {T <: Union{Number, Function}} = value(x) == value(y)
Base.isapprox(x::Coefficient, y::Coefficient) = false
Base.isapprox(x::Coefficient{<:Function}, y::Coefficient{<:Function}) = (x == y)
Base.isapprox(x::Coefficient{<:Number}, y::Coefficient{<:Number}; kwargs...) = isapprox(value(x), value(y); kwargs...)

"""
	scalar_type(x)
	return the scale type of x, which must be a subtype of Number.
"""
scalar_type(x::Coefficient{T}) where {T<:Number} = T
scalar_type(x::Coefficient{<:Function}) = begin
    m = value(x)(1.)
    return typeof(m)
end

"""
	is_zero(x) 
	check if x is zero.
"""
is_zero(x::Coefficient{T}) where {T<:Number} = value(x) == zero(T)
is_zero(x::Coefficient{<:Function}) = false


