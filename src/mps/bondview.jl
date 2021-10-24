




struct MPSBondView{M}
	parent::M
end


Base.length(m::MPSBondView) = length(raw_singular_matrices(m.parent))
Base.firstindex(m::MPSBondView) = firstindex(raw_singular_matrices(m.parent))
Base.lastindex(m::MPSBondView) = lastindex(raw_singular_matrices(m.parent))
# Base.getindex(m::MPSBondView, i::Integer) = getindex(raw_singular_matrices(m.parent), i)
Base.getindex(m::MPSBondView,r::Union{Integer,AbstractRange{Int64}, Colon}) = raw_singular_matrices(m.parent)[r] 
function Base.setindex!(m::MPSBondView, v, i::Integer)
	return setindex!(raw_singular_matrices(m.parent), v, i)
end
function Base.setindex!(m::MPSBondView, v, r::AbstractRange{Int64})
	for (vj, rj) in zip(v, r)
		setindex!(m, vj, rj)
	end
	return v
end
Base.setindex!(m::MPSBondView, v, r::Colon) = setindex!(m, v, 1:length(m.parent)+1)

