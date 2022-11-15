struct QTerm{T <: Number} <: AbstractQuantumTerm 
	positions::Vector{Int}
	op::Vector{Array{T, 4}}
	coeff::AbstractCoefficient


function QTerm{T}(pos::Vector{Int}, m::Vector{<:MPOTensor}, v::Coefficient) where {T<:Number}
	# checks
	(length(pos) == length(m)) || throw(DimensionMismatch())
	isempty(m) && throw(ArgumentError("no input."))
	check_qterm_positions(pos)
	((size(m[1], 1) == 1) && size(m[end], 3)==1) || throw(ArgumentError("strict MPO tensors expected."))
	return new{T}(pos, convert(Vector{Array{T, 4}}, m), v)
end

end

function QTerm(pos::Vector{Int}, m::Vector{<:MPOTensor}, v::AllowedCoefficient)
	v = Coefficient(v)
	T = scalar_type(v)
	for item in m
		T = promote_type(T, eltype(item))
	end
	return QTerm{T}(pos, m, v)
end 
QTerm(pos::Vector{Int}, m::Vector{<:AbstractMatrix}, v::AllowedCoefficient) = QTerm(
	pos, [reshape(item, (1, size(item, 1), 1, size(item, 2))) for item in m], v)
QTerm(pos::Vector{Int}, m::Vector{M}; coeff::AllowedCoefficient=1) where {M <: Union{AbstractMatrix, MPOTensor}} = QTerm(pos, m, coeff)

QTerm(pos::Tuple, m::Vector{M}; coeff::AllowedCoefficient=1) where {M <: Union{AbstractMatrix, MPOTensor}} = QTerm([pos...], m, coeff=coeff)
QTerm(pos::Int, m::M; coeff::AllowedCoefficient=1.) where {M <: Union{AbstractMatrix, MPOTensor}} = QTerm([pos], [m], coeff=coeff)
function QTerm(x::Pair{Int, <:AbstractArray}...; coeff::AllowedCoefficient=1.) 
	pos, ms = _parse_pairs(x...)
	return QTerm(pos, ms; coeff=coeff)
end 

(x::QTerm)(t::Number) = QTerm(positions(x), op(x), coeff(x)(t))


Base.copy(x::QTerm) = QTerm(copy(positions(x)), copy(op(x)), copy(coeff(x)))
Base.adjoint(x::QTerm) = QTerm(copy(positions(x)), mpo_tensor_adjoint.(op(x)), conj(coeff(x)))

Base.:*(s::QTerm, m::AllowedCoefficient) = QTerm(positions(s), op(s), coeff(s) * m)

shift(m::QTerm, i::Int) = QTerm(positions(m) .+ i, op(m), coeff(m))

id(x::QTerm) = QTerm(positions(x), [_eye(scalar_type(x), d) for d in physical_dimensions(x)], coeff=1.)


"""
	Base.:*(x::QTerm, y::QTerm)
multiplication between two QTerms
"""
function Base.:*(x::QTerm, y::QTerm)
	pos, opx, opy = _coerce_qterms(x, y)
	v = _mult_n_n(opx, opy)
	return QTerm(pos, v, coeff=coeff(x)*coeff(y))
end

function _coerce_qterms(x::QTerm, y::QTerm)
	T = promote_type(scalar_type(x), scalar_type(y))
    opx = op(x)
    opy = op(y)
    pos = positions(x)
    if !(positions(x) == positions(y))
    	x_left = 1
    	y_left = 1
    	new_pos = sort([Set(vcat(positions(x), positions(y)))...])
    	new_opx = Matrix{T}[]
    	new_opy = Matrix{T}[]
    	for pos in new_pos
    		pos_x = findfirst(a->a==pos, positions(x))
    		pos_y = findfirst(a->a==pos, positions(y))
    		if isnothing(pos_x) && !(isnothing(pos_y))
    			@assert y_left == size(opy[pos_y], 1)
    			push!(new_opx, _id4(T, x_left, size(opy[pos_y], 2) ) )
    			push!(new_opy, opy[pos_y])
    			y_left = size(opy[pos_y], 3)
    		elseif !(isnothing(pos_x)) && isnothing(pos_y)
    			@assert x_left == size(opx[pos_x], 1)
    			push!(new_opx, opx[pos_x])
    			push!(new_opy, _id4(T, y_left, size(opx[pos_x], 2)) )
    			x_left = size(opx[pos_x], 3)
    		elseif !(isnothing(pos_x)) && !(isnothing(pos_y))
    			@assert x_left == size(opx[pos_x], 1)
    			@assert y_left == size(opy[pos_y], 1)
    			push!(new_opx, opx[pos_x])
    			push!(new_opy, opy[pos_y])
    			x_left = size(opx[pos_x], 3)
    			y_left = size(opy[pos_y], 3)
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


function check_qterm_positions(pos::Vector{Int})
	(length(Set(pos)) == length(pos)) || throw(ArgumentError("duplicate positions not allowed"))
	(sort(pos) == pos) || throw(ArgumentError("QTerm positions should be strictly ordered."))
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

