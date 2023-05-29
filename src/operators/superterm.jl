

struct SuperTerm{T} <: AbstractSuperTerm
	data::QTerm{T}

function SuperTerm{T}(data::QTerm) where {T<:Number}
	new{T}(data)
end
end

SuperTerm(data::QTerm{T}) where {T} = SuperTerm{T}(data)
Base.eltype(::Type{SuperTerm{T}}) where T = T

data(x::SuperTerm) = x.data

Base.:*(s::SuperTerm, m::AllowedCoefficient) = SuperTerm(data(s) * m)


"""
	superoperator(x::Vector, y::Vector)
	utility function to return x ⊗ conj(y) |ρ⟩ = x ρ y^†
	basically a copy from ⊗(x, y) or ⊠(x, y), 
"""
function superterm(x::QTerm, y::QTerm) 
    pos, opx, opy = _coerce_qterms(x, y)
    v = [rkron(a, conj(b)) for (a, b) in zip(opx, opy)]
	return SuperTerm(QTerm(pos, v, coeff=coeff(x) * conj(coeff(y))))
end

Base.:*(x::SuperTerm, y::SuperTerm) = SuperTerm(x.data * y.data)