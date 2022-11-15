

# The default constructor is never to be explicited by the user.
# superoperator has essentially the same data structure as a QuantumOperator
struct SuperOperator{T<:Number} <: AbstractSuperOperator
	data::QuantumOperator{T}
end

scalar_type(::Type{SuperOperator{T}}) where {T} = T

Base.copy(x::SuperOperator) = SuperOperator(copy(x.data))

Base.similar(s::SuperOperator{T}) where {T} = SuperOperator{T}()

(x::SuperOperator)(t::Number) = SuperOperator(x.data(t))

Base.:*(x::SuperOperator, y::AllowedCoefficient) = SuperOperator(x.data * y)
Base.:+(x::SuperOperator, y::SuperOperator) = SuperOperator(x.data + y.data)
Base.:-(x::SuperOperator, y::SuperOperator) = SuperOperator(x.data - y.data)
shift(x::SuperOperator, n::Int) = SuperOperator(shift(x.data, n))
qterms(x::SuperOperator, args...) = [SuperTerm(item) for item in qterms(x.data, args...)]
_expm(x::SuperOperator, dt::Number) = _expm(x.data, dt)
absorb_one_bodies(h::SuperOperator) = SuperOperator(absorb_one_bodies(h.data))

SuperOperator(terms::Vector{<:SuperTerm}) = SuperOperator(QuantumOperator(data.(terms)))


# add new terms into SuperOperator
function add!(m::SuperOperator, x::SuperTerm)
	add!(m.data, data(x))
end

"""
	add_unitary!(m::SuperOperator, x::QTerm)
	add a term -i x ρ + i ρ x', namely -i x ⊗ conj(I) + I ⊗ conj(x)
"""
function add_unitary!(m::SuperOperator, x::QTerm)
	x2 = -im * x
	iden = id(x)
	add!(m, superoperator(x2, iden) )
	add!(m, superoperator(iden, x2) )
end 

function add_unitary!(m::SuperOperator, x::QuantumOperator)
	for t in qterms(x)
		add_unitary!(m, t)
	end
end

function add_dissipation!(m::SuperOperator, x::QTerm)
	add!(m, 2*superoperator(x, x))
	x2 = x' * x
	iden = -id(x)
	add!(m, superoperator(x2, iden) )
	add!(m, superoperator(iden, x2))
end

function add_dissipation!(m::SuperOperator, x::QuantumOperator)
	terms = qterms(x)
	for t1 in terms
		for t2 in terms
			add!(m, 2*superoperator(t1, t2))
			x2 = t2' * t1
			iden = -id(x2)
			add!(m, superoperator(x2, iden))
			add!(m, superoperator(iden, x2))
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
	return SuperOperator([terms...])
end


