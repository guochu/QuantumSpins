# I support time-dependent quantum operator by using this class
abstract type AbstractCoefficient end

"""
	struct Coefficient{T} <: AbstractCoefficient

T is either a scalar or an unary function
"""
struct Coefficient{T} <: AbstractCoefficient
	value::T

function Coefficient{T}(f) where {T<:Function}
	m = f(1.)
	isa(m, Number) || throw(ArgumentError("f should be either a scalar or a unary function"))
	new{T}(f)
end
Coefficient{T}(f) where {T<:Number} = new{T}(f)
end

Coefficient(f::Union{Number, Function}) = Coefficient{typeof(f)}(f)
Coefficient(x::Coefficient) = Coefficient(value(x))
const AllowedCoefficient = Union{Number, Function, Coefficient}

# value(x::Number) = x
value(x::Coefficient) = x.value
scalar(x::Coefficient{<:Number}) = value(x)
scalar(x::Coefficient{<:Function}) = throw(ArgumentError("cannot convert a function into a scalar"))
Base.copy(x::Coefficient) = Coefficient(value(x))
coeff(x::AllowedCoefficient) = Coefficient(x)

Base.:+(x::Coefficient{<:Number}, y::Coefficient{<:Number}) = Coefficient(value(x) + value(y))
Base.:+(x::Coefficient{<:Function}, y::Coefficient{<:Number}) = x + value(y)
Base.:+(y::Coefficient{<:Number}, x::Coefficient{<:Function}) = x + y
Base.:+(x::Coefficient{<:Function}, y::Coefficient{<:Function}) = Coefficient(z->value(x)(z) + value(y)(z))

Base.:-(x::Coefficient{<:Number}, y::Coefficient{<:Number}) = Coefficient(value(x) - value(y))
Base.:-(x::Coefficient{<:Function}, y::Coefficient{<:Number}) = x - value(y)
Base.:-(y::Coefficient{<:Number}, x::Coefficient{<:Function}) = -x + y
Base.:-(x::Coefficient{<:Function}, y::Coefficient{<:Function}) = Coefficient(z->value(x)(z) - value(y)(z))

Base.:*(x::Coefficient{<:Number}, y::Coefficient{<:Number}) = Coefficient(value(x) * value(y))
Base.:*(x::Coefficient{<:Function}, y::Coefficient{<:Number}) = x * value(y)
Base.:*(y::Coefficient{<:Number}, x::Coefficient{<:Function}) = x * y
Base.:*(x::Coefficient{<:Function}, y::Coefficient{<:Function}) = Coefficient(z->value(x)(z) * value(y)(z))

Base.:/(x::Coefficient{<:Number}, y::Coefficient{<:Number}) = Coefficient(value(x) / value(y))
Base.:/(x::Coefficient{<:Function}, y::Coefficient{<:Number}) = x / value(y)
Base.:/(y::Coefficient{<:Number}, x::Coefficient{<:Function}) = Coefficient(z -> y(value(x)(z)))
Base.:/(x::Coefficient{<:Function}, y::Coefficient{<:Function}) = Coefficient(z->value(x)(z) / value(y)(z))


Base.:+(x::Coefficient{<:Number}, y::Number) = Coefficient(value(x) + y)
Base.:+(x::Coefficient{<:Function}, y::Number) = Coefficient(z->value(x)(z+y))
Base.:+(y::Number, x::Coefficient) = x + y

Base.:-(x::Coefficient{<:Number}, y::Number) = Coefficient(value(x) - y)
Base.:-(x::Coefficient{<:Function}, y::Number) = Coefficient(z->value(x)(z-y))
Base.:-(y::Number, x::Coefficient) = -x + y

Base.:*(x::Coefficient{<:Number}, y::Number) = Coefficient(value(x) * y)
Base.:*(x::Coefficient{<:Function}, y::Number) = Coefficient(z->value(x)(z)*y)
Base.:*(y::Number, x::Coefficient) = x * y

Base.:/(x::Coefficient{<:Number}, y::Number) = Coefficient(value(x) / y)
Base.:/(x::Coefficient{<:Function}, y::Number) = Coefficient(z->value(x)(z)/y)
Base.:/(x::Number, y::Coefficient{<:Number}) = Coefficient(x / value(y))
Base.:/(x::Number, y::Coefficient{<:Function}) = Coefficient(z->x/(value(y)(z)))

Base.:-(x::Coefficient{<:Number}) = Coefficient(-value(x))
Base.:-(x::Coefficient{<:Function}) = Coefficient(z->-value(x)(z))

# only implemented those I need, not aimed for general unary operations
Base.conj(x::Coefficient{<:Number}) = Coefficient(conj(value(x)))
Base.conj(x::Coefficient{<:Function}) = Coefficient(z->conj(value(x)(z)))
Base.sqrt(x::Coefficient{<:Number}) = Coefficient(sqrt(value(x)))
Base.sqrt(x::Coefficient{<:Function}) = Coefficient(z->sqrt(value(x)(z)))

(op::Coefficient{<:Number})(t::Number) = value(op)
(op::Coefficient{<:Function})(t::Number) = value(op)(t)

"""
	isconstant(x)
	check if the object x is a pure constant or contrans function 
"""
isconstant(x::Coefficient{<:Number}) = true
isconstant(x::Coefficient{<:Function}) = false


Base.:(==)(x::Coefficient, y::Coefficient) = false
Base.:(==)(x::Coefficient{<:Number}, y::Coefficient{<:Number}) = value(x) == value(y)
Base.:(==)(x::Coefficient{<:Function}, y::Coefficient{<:Function}) = value(x) === value(y)

Base.isapprox(x::Coefficient, y::Coefficient; kwargs...) = false
Base.isapprox(x::Coefficient{<:Number}, y::Coefficient{<:Number}; kwargs...) = isapprox(x, y; kwargs...)
Base.isapprox(x::Coefficient{<:Function}, y::Coefficient{<:Function}; kwargs...) = x == y

"""
	Base.eltype(x)

Return the scale type of x, which must be a subtype of Number.
"""
Base.eltype(x::Coefficient{T}) where {T<:Number} = T
Base.eltype(x::Coefficient{<:Function}) = begin
    m = value(x)(1.)
    return typeof(m)
end

"""
	iszero(x) 
	check if x is zero.
"""
Base.iszero(x::Coefficient{T}) where {T<:Number} = iszero(value(x))
Base.iszero(x::Coefficient{<:Function}) = false
