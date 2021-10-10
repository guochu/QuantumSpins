abstract type AbstractQuantumTerm end


function _get_normal_term(key::Vector{Int}, op)
	seq = sortperm(key)
	return key[seq], op[seq]
end

struct QTerm{T <: Number} <: AbstractQuantumTerm 
	positions::Vector{Int}
	op::Vector{Matrix{T}}
	coeff::AbstractCoefficient


function QTerm{T}(pos::Vector{Int}, m::Vector{<:AbstractMatrix}, v::Coefficient) where {T<:Number}
	# checks
	(length(pos) == length(m)) || throw(DimensionMismatch())
	isempty(m) && throw(ArgumentError("no input."))
	pos, m = _get_normal_term(pos, m)
	for item in m
		(size(item, 1) == size(item, 2)) || throw(DimensionMismatch("square matrix required."))
	end
	return new{T}(pos, convert(Vector{Matrix{T}}, m), v)
end

end

positions(x::QTerm) = x.positions
op(x::QTerm) = x.op
coeff(x::QTerm) = x.coeff

function QTerm(pos::Vector{Int}, m::Vector{<:AbstractMatrix}, v::Union{Coefficient, Number, Function})
	v = Coefficient(v)
	T = scalar_type(v)
	for item in m
		T = promote_type(T, eltype(item))
	end
	return QTerm{T}(pos, m, v)
end 
QTerm(pos::Vector{Int}, m::Vector{<:AbstractMatrix}; coeff::Union{Coefficient, Number, Function}=1) = QTerm(pos, m, coeff)
QTerm(pos::Tuple, m::Vector{<:AbstractMatrix}; coeff::Union{Coefficient, Number, Function}=1) = QTerm([pos...], m, coeff=coeff)
QTerm(pos::Int, m::AbstractMatrix; coeff::Union{Number, Function, Coefficient}=1.) = QTerm([pos], [m], coeff=coeff)
function QTerm(x::Pair{Int, <:AbstractMatrix}...; coeff::Union{Number, Function, Coefficient}=1.) 
	pos, ms = _parse_pairs(x...)
	return QTerm(pos, ms; coeff=coeff)
end 


function _parse_pairs(x::Pair...)
	pos = Int[]
	ms = []
	for (a, b) in x
		push!(pos, a)
		push!(ms, b)
	end
	return pos, [ms...]
end

(x::QTerm)(t::Number) = QTerm(positions(x), op(x), coeff(x)(t))



Base.copy(x::QTerm) = QTerm(copy(positions(x)), copy(op(x)), copy(coeff(x)))
Base.isempty(x::QTerm) = isempty(op(x))
Base.adjoint(x::QTerm) = QTerm(copy(positions(x)), [item' for item in op(x)], conj(coeff(x)))

Base.:*(s::QTerm, m::Union{Number, Coefficient}) = QTerm(positions(s), op(s), coeff(s) * m)
Base.:*(m::Union{Number, Coefficient}, s::QTerm) = s * m
Base.:/(s::QTerm, m::Union{Number, Coefficient}) = QTerm(positions(s), op(s), coeff(s) / m)
Base.:+(s::QTerm) = s
Base.:-(s::QTerm) = QTerm(positions(s), op(s), -coeff(s))


nterms(s::QTerm) = length(op(s))
is_constant(s::QTerm) = is_constant(coeff(s))
function scalar_type(x::QTerm)
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
interaction_range(x::QTerm) = _interaction_range(positions(x))

shift(m::QTerm, i::Int) = QTerm(positions(m) .+ i, op(m), coeff(m))

function is_zero(x::QTerm) 
	is_zero(coeff(x)) && return true
	isempty(x) && return true
	for item in op(x)
	    is_zero(item) && return true
	end
	return false
end

_join_ops(s::QTerm) = begin
    is_constant(s) || error("can not join non-constant hamiltonian term.")
    return _join_chain_ops(op(s))*value(coeff(s))
end 

function _coerce_qterms(x::QTerm, y::QTerm)
	T = promote_type(scalar_type(x), scalar_type(y))
    opx = op(x)
    opy = op(y)
    pos = positions(x)
    if !(positions(x) == positions(y))
    	new_pos = sort([Set(vcat(positions(x), positions(y)))...])
    	new_opx = Matrix{T}[]
    	new_opy = Matrix{T}[]
    	for pos in new_pos
    		pos_x = findfirst(a->a==pos, positions(x))
    		pos_y = findfirst(a->a==pos, positions(y))
    		if isnothing(pos_x) && !(isnothing(pos_y))
    			push!(new_opx, _eye(T, size(opy[pos_y], 1) ) )
    			push!(new_opy, opy[pos_y])
    		elseif !(isnothing(pos_x)) && isnothing(pos_y)
    			push!(new_opx, opx[pos_x])
    			push!(new_opy, _eye(T, size(opx[pos_x], 1)))
    		elseif !(isnothing(pos_x)) && !(isnothing(pos_y))
    			push!(new_opx, opx[pos_x])
    			push!(new_opy, opy[pos_y])
    		else
    			throw(ArgumentError("why here?"))
    		end
    	end
    	opx = new_opx
    	opy = new_opy
    	pos = new_pos
    end
    return pos, opx, opy
end


"""
	Base.:*(x::QTerm, y::QTerm)
multiplication between two QTerms
"""
function Base.:*(x::QTerm, y::QTerm)
	pos, opx, opy = _coerce_qterms(x, y)
	v = [a * b for (a, b) in zip(opx, opy)]
	return QTerm(pos, v, coeff=coeff(x)*coeff(y))
end









