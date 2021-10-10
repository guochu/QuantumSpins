




struct MPSBondView{M}
	parent::M
end


Base.length(m::MPSBondView) = length(raw_singular_matrices(m.parent))
Base.getindex(m::MPSBondView, i::Int) = getindex(raw_singular_matrices(m.parent), i)
Base.firstindex(m::MPSBondView) = firstindex(raw_singular_matrices(m.parent))
Base.lastindex(m::MPSBondView) = lastindex(raw_singular_matrices(m.parent))
function Base.setindex!(m::MPSBondView, v, i::Int)
	return setindex!(raw_singular_matrices(m.parent), v, i)
end

Base.getindex(psi::MPSBondView,r::AbstractRange{Int64}) = [psi[ri] for ri in r]
