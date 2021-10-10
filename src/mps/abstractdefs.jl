
const MPSTensor{T} = AbstractArray{T, 3} where {T<:Number}
const SingularVector{T} = AbstractVector{T} where {T <: Real}

abstract type AbstractMPS end