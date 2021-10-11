

# The default constructor is never to be explicited by the user.
# superoperator has essentially the same data structure as a QuantumOperator
struct SuperOperatorBase{T<:Number}
	data::QuantumOperator{T}
end

scalar_type(::Type{SuperOperatorBase{T}}) where {T} = T
scalar_type(x::SuperOperatorBase) = scalar_type(typeof(x))

physical_dimensions(x::SuperOperatorBase) = physical_dimensions(x.data)

Base.copy(x::SuperOperatorBase) = SuperOperatorBase(copy(x.data))

Base.empty!(s::SuperOperatorBase) = empty!(s.data)
Base.isempty(s::SuperOperatorBase) = isempty(s.data)

Base.similar(s::SuperOperatorBase{T}) where {T} = SuperOperatorBase{T}()
Base.length(x::SuperOperatorBase) = length(x.data)
Base.keys(x::SuperOperatorBase) = keys(x.data)


is_constant(x::SuperOperatorBase) = is_constant(x.data)
interaction_range(x::SuperOperatorBase)  = interaction_range(x.data)

(x::SuperOperatorBase)(t::Number) = SuperOperatorBase(x.data(t))
Base.:*(x::SuperOperatorBase, y::Number) = SuperOperatorBase(x.data * y)
Base.:*(y::Number, x::SuperOperatorBase) = x * y
Base.:/(x::SuperOperatorBase, y::Number) = x * (1/y)
Base.:+(x::SuperOperatorBase) = x
Base.:-(x::SuperOperatorBase) = SuperOperatorBase(-x.data)
Base.:+(x::SuperOperatorBase, y::SuperOperatorBase) = SuperOperatorBase(x.data + y.data)
Base.:-(x::SuperOperatorBase, y::SuperOperatorBase) = SuperOperatorBase(x.data - y.data)
shift(x::SuperOperatorBase, n::Int) = SuperOperatorBase(shift(x.data, n))
qterms(x::SuperOperatorBase, args...) = qterms(x.data, args...)
_expm(x::SuperOperatorBase, dt::Number) = _expm(x.data, dt)
absorb_one_bodies(h::SuperOperatorBase) = SuperOperatorBase(absorb_one_bodies(h.data))



"""
	superoperator(x::Vector, y::Vector)
	utility function to return x ⊗ conj(y) |ρ⟩ = x ρ y^†
	basically a copy from ⊗(x, y) or ⊠(x, y), 
"""
function superoperator(x::QTerm, y::QTerm) 
    pos, opx, opy = _coerce_qterms(x, y)
    v = [rkron(a, conj(b)) for (a, b) in zip(opx, opy)]
	return QTerm(pos, v, coeff=coeff(x) * conj(coeff(y)))
end

id(x::QTerm) = QTerm(positions(x), [_eye(scalar_type(x), size(m, 1)) for m in op(x)], coeff=1.)

# add new terms into SuperOperatorBase

"""
	add_unitary!(m::SuperOperatorBase, x::QTerm)
	add a term -i x ρ + i ρ x', namely -i x ⊗ conj(I) + I ⊗ conj(x)
"""
function add_unitary!(m::SuperOperatorBase, x::QTerm)
	x2 = -im * x
	iden = id(x)
	add!(m.data, superoperator(x2, iden) )
	add!(m.data, superoperator(iden, x2) )
end 

function add_unitary!(m::SuperOperatorBase, x::QuantumOperator)
	for t in qterms(x)
		add_unitary!(m, t)
	end
end

function add_dissipation!(m::SuperOperatorBase, x::QTerm)
	add!(m.data, 2*superoperator(x, x))
	x2 = x' * x
	iden = -id(x)
	add!(m.data, superoperator(x2, iden) )
	add!(m.data, superoperator(iden, x2))
end

function add_dissipation!(m::SuperOperatorBase, x::QuantumOperator)
	terms = qterms(x)
	for t1 in terms
		for t2 in terms
			add!(m.data, 2*superoperator(t1, t2))
			x2 = t2' * t1
			iden = -id(x2)
			add!(m.data, superoperator(x2, iden))
			add!(m.data, superoperator(iden, x2))
		end
	end
end

"""
	superoperator(h::QuantumOperator, f)
	hρ + ρ conj(h)
"""
function superoperator(h::QuantumOperator)
	terms = []
	for t in qterms(h)
		x2 = t
		iden = id(t)
		push!(terms, superoperator(x2, iden))
		push!(terms, superoperator(iden, x2))
	end
	return SuperOperatorBase(QuantumOperator([terms...]))
end


