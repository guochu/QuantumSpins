

function MPS(f, ::Type{T}, physpaces::Vector{Int}, virtualpaces::Vector{Int}) where {T <: Number}
	L = length(physpaces)
	(length(virtualpaces) == L+1) || throw(DimensionMismatch())
	any(virtualpaces .== 0) &&  @warn "auxiliary space is empty."
	mpstensors = [f(T, virtualpaces[i], physpaces[i], virtualpaces[i+1]) for i in 1:L]
	return MPS(mpstensors)
end


function prodmps(::Type{T}, physpaces::Vector{Int}, physectors::Vector{Vector{T2}}) where {T <: Number, T2 <:Number}
	(length(physpaces) == length(physectors)) || throw(DimensionMismatch())
	for (a, b) in zip(physpaces, physectors)
		(a == length(b)) || throw(DimensionMismatch())
	end
	return MPS([reshape(convert(Vector{T}, b), 1, a, 1) for (a, b) in zip(physpaces, physectors)])
end
prodmps(physpaces::Vector{Int}, physectors::Vector{Vector{T}}) where {T <:Number} = prodmps(T, physpaces, physectors)
prodmps(::Type{T}, physectors::Vector{Vector{T2}}) where {T <: Number, T2 <:Number} = prodmps(T, length.(physectors), physectors )
prodmps(physectors::Vector{Vector{T}}) where {T <: Number} = prodmps(T, physectors)

function onehot(::Type{T}, d::Int, i::Int) where {T <: Number}
	((i >= 0) && (i < d)) || throw(BoundsError())
	r = zeros(T, d)
	r[i+1] = 1
	return r
end
function prodmps(::Type{T}, ds::Vector{Int}, physectors::Vector{Int}) where {T <: Number}
	(length(ds) == length(physectors)) || throw(DimensionMismatch())
	return prodmps([onehot(T, d, i) for (d, i) in zip(ds, physectors)])
end
prodmps(ds::Vector{Int}, physectors::Vector{Int}) = prodmps(Float64, ds, physectors)


function randommps(::Type{T}, physpaces::Vector{Int}; D::Int) where {T <: Number}
	virtualpaces = max_bond_dimensions(physpaces, D)
	return MPS(randn, T, physpaces, virtualpaces)
end
randommps(physpaces::Vector{Int}; D::Int) = randommps(Float64, physpaces; D=D)
randommps(::Type{T}, L::Int; d::Int, D::Int) where {T <: Number} = randommps(T, [d for i in 1:L]; D=D)
randommps(L::Int; d::Int, D::Int) = randommps(Float64, L; d=d, D=D)

