"""
	prodmpo(::Type{T}, physpaces::Vector{Int}, ms::AbstractDict{Int, M}) where {T <: Number, M <: AbstractMatrix}
construct product mpo with chain of 4-dimensional tensors, missing points are interpretted as identity.
"""
function prodmpo(::Type{T}, physpaces::Vector{Int}, ms::AbstractDict{Int, M}) where {T <: Number, M <: MPOTensor}
	L = length(physpaces)
	for (k, v) in ms
		((k>= 1) && (k <= L)) || throw(BoundsError())
		(physpaces[k] == size(v, 2) == size(v, 4)) || throw(DimensionMismatch("space mismatch on site $k."))
	end
	mpotensors = Vector{Array{T, 4}}(undef, L)
	left = 1
	for i in 1:L
		mpotensors[i] = get(ms, i, _id4(T, left, physpaces[i]))
		left = size(mpotensors[i], 3)
	end
	return MPO(mpotensors)
end
prodmpo(::Type{T}, physpaces::Vector{Int}, ms::AbstractDict{Int, AbstractMatrix}) where T = prodmpo(
	T, physpaces, Dict(k=>reshape(v, (1, size(v, 1), 1, size(v, 2)) ) for (k, v) in ms))

function _site_ops_to_dict(pos::Vector{Int}, ms::Vector)
	(length(pos) == length(ms)) || throw(DimensionMismatch())
	(length(Set(pos)) == length(pos)) || throw(ArgumentError("duplicate positions not allowed."))
	return Dict(k=>v for (k, v) in zip(pos, ms))
end

prodmpo(::Type{T}, physpaces::Vector{Int}, pos::Vector{Int}, ms::Vector{M}) where {T <: Number, M <: Union{AbstractMatrix, MPOTensor}} = prodmpo(
	T, physpaces, _site_ops_to_dict(pos, ms))

prodmpo(physpaces::Vector{Int}, ms::AbstractDict{Int, M}) where {M <: Union{AbstractMatrix, MPOTensor}} = prodmpo(eltype(M), physpaces, ms)
prodmpo(physpaces::Vector{Int}, pos::Vector{Int}, ms::Vector{M}) where {M <: Union{AbstractMatrix, MPOTensor}} = prodmpo(eltype(M), physpaces, pos, ms)


"""
	randommpo(::Type{T}, dy::Vector{Int}, dx::Vector{Int}; D::Int) where {T<:Number}
	dy are the input dimensions, dx are the output dimensions
"""
function randommpo(::Type{T}, dx::Vector{Int}, dy::Vector{Int}; D::Int) where {T<:Number}
	(length(dx) == length(dy)) || throw(DimensionMismatch())
	L = length(dx)
	r = Vector{Array{T, 4}}(undef, L)
	r[1] = randn(T, 1, dx[1], D, dy[1])
	r[L] = randn(T, D, dx[L], 1, dy[L])
	for i in 2:L-1
		r[i] = randn(T, D, dx[i], D, dy[i])
	end
	return MPO(r)
end 
randommpo(::Type{T}, physpaces::Vector{Int}; kwargs...) where {T<:Number} = randommpo(T, physpaces, physpaces; kwargs...)
randommpo(::Type{T}, L::Int; d::Int, D::Int) where {T<:Number} = randommpo(T, [d for i in 1:L], D=D)
randommpo(L::Int; kwargs...) = randommpo(Float64, L; kwargs...)


function _id4(::Type{T}, a::Int, b::Int) where {T <: Number}
	m1 = _eye(T, a)
	m2 = _eye(T, b)
	@tensor tmp[1,3,2,4] := m1[1,2] * m2[3,4]
	return tmp
end
_id4(a::Int, b::Int) = _id4(Float64, a, b)
