


struct DensityMatrix{T <: Number}
	data::Vector{T}
	nqubits::Int

function DensityMatrix{T}(data::Vector{<:Number}, nqubits::Int) where {T <: Number}
	(length(data) == 2^(2*nqubits)) || throw(DimensionMismatch())
	new{T}(data, nqubits)
end

end


function DensityMatrix{T}(m::AbstractMatrix{<:Number}, nqubits::Int) where {T <: Number}
	(size(m, 1) == size(m, 2) == 2^nqubits) || throw(DimensionMismatch())
	

end


nqubits(x::DensityMatrix) = x.nqubits