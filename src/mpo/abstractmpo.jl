

abstract type AbstractMPO end

space_l(a::AbstractMPO) = space_l(a[1])
space_r(a::AbstractMPO) = space_r(a[end])