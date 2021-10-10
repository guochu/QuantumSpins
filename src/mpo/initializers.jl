"""
	prodmpo(::Type{T}, physpaces::Vector{Int}, ms::AbstractDict{Int, M}) where {T <: Number, M <: AbstractMatrix}
construct product mpo with chain of 4-dimensional tensors, missing points are interpretted as identity.
"""
function prodmpo(::Type{T}, physpaces::Vector{Int}, ms::AbstractDict{Int, M}) where {T <: Number, M <: AbstractMatrix}
	L = length(physpaces)
	for (k, v) in ms
		((k>= 1) && (k <= L)) || throw(BoundsError())
		(physpaces[k] == size(v, 1) == size(v, 2)) || throw(DimensionMismatch("space mismatch on site $k."))
	end
	mpotensors = Vector{Array{T, 4}}(undef, L)
	for i in 1:L
		tmp = get(ms, i, _eye(T, physpaces[i]))
		mpotensors[i] = reshape(tmp, 1, size(tmp, 1), 1, size(tmp, 2))
	end
	return MPO(mpotensors)
end

function _site_ops_to_dict(pos::Vector{Int}, ms::Vector)
	(length(pos) == length(ms)) || throw(DimensionMismatch())
	(length(Set(pos)) == length(pos)) || throw(ArgumentError("duplicate positions not allowed."))
	return Dict(k=>v for (k, v) in zip(pos, ms))
end

prodmpo(::Type{T}, physpaces::Vector{Int}, pos::Vector{Int}, ms::Vector{M}) where {T <: Number, M <: AbstractMatrix} = prodmpo(
	T, physpaces, _site_ops_to_dict(pos, ms))

prodmpo(physpaces::Vector{Int}, ms::AbstractDict{Int, M}) where {M <: AbstractMatrix} = prodmps(eltype(M), physpaces, ms)
prodmpo(physpaces::Vector{Int}, pos::Vector{Int}, ms::Vector{M}) where {M <: AbstractMatrix} = prodmps(eltype(M), physpaces, pos, ms)


function randommpo(::Type{T}, dy::Vector{Int}, dx::Vector{Int}; D::Int) where {T<:Number}
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