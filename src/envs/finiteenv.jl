
function init_hstorage!(hstorage::Vector, mpo::MPO, mps::MPS, center::Int)
	(length(mpo) == length(mps)) || throw(DimensionMismatch())
	(length(mps)+1 == length(hstorage)) || throw(DimensionMismatch())
	right = r_RR(mps, mpo, mps)
	L = length(mps)
	# hstorage = Vector{typeof(right)}(undef, L+1)
	hstorage[L+1] = right
	hstorage[1] = l_LL(mps, mpo, mps)
	for i in L:-1:center+1
		hstorage[i] = updateright(hstorage[i+1], mps[i], mpo[i], mps[i])
	end
	for i in 1:center-1
		hstorage[i+1] = updateleft(hstorage[i], mps[i], mpo[i], mps[i])
	end
	return hstorage	
end

function init_hstorage(mpo::MPO, mps::MPS, center::Int)
	T = promote_type(scalar_type(mpo), scalar_type(mps))
	hstorage = Vector{Array{T, 3}}(undef, length(mps)+1)
	return init_hstorage!(hstorage, mpo, mps, center)
end

init_hstorage_right(mpo::MPO, mps::MPS) = init_hstorage(mpo, mps, 1)


struct ExpectationCache{M<:MPO, V<:MPS, H} <: AbstractCache
	mpo::M
	mps::V
	hstorage::H
end

environments(mpo::MPO, mps::MPS) = ExpectationCache(mpo, mps, init_hstorage_right(mpo, mps))

function Base.getproperty(m::ExpectationCache, s::Symbol)
	if s == :state
		return m.mps
	elseif s == :h
		return m.mpo
	elseif s == :env 
		return m.hstorage
	else
		return getfield(m, s)
	end
end

function recalculate!(m::ExpectationCache, mps::MPS, center::Int)
	if mps !== m.state
		init_hstorage!(m.env, m.h, mps, center)
	end
end

function increase_bond!(m::ExpectationCache; D::Int)
	if bond_dimension(m.state) < D
		increase_bond!(m.state, D=D)
		canonicalize!(m.state, normalize=false)
		init_hstorage!(m.env, m.h, m.state, 1)
	end
end


function updateleft!(env::ExpectationCache, site::Int)
	env.hstorage[site+1] = updateleft(env.hstorage[site], env.mps[site], env.h[site], env.mps[site])
end

function updateright!(env::ExpectationCache, site::Int)
	env.hstorage[site] = updateright(env.hstorage[site+1], env.mps[site], env.h[site], env.mps[site])
end




# for excited states
struct ProjectedExpectationCache{M<:MPO, V<:MPS, H, C} <: AbstractCache
	mpo::M
	mps::V
	projectors::Vector{V}
	hstorage::H
	cstorages::Vector{C}
end


function init_cstorage_right(psiA::MPS, psiB::MPS)
	(length(psiA) == length(psiB)) || throw(DimensionMismatch())
	(space_r(psiA) == space_r(psiB)) || throw(DimensionMismatch())
	L = length(psiA)
	hold = r_RR(psiA, psiB)
	cstorage = Vector{typeof(hold)}(undef, L+1)
	cstorage[1] = l_LL(psiA)
	cstorage[L+1] = hold
	for i in L:-1:2
		cstorage[i] = updateright(cstorage[i+1], psiA[i], psiB[i])
	end
	return cstorage
end

environments(mpo::MPO, mps::M, projectors::Vector{M}) where {M<:MPS} = ProjectedExpectationCache(
	mpo, mps, projectors, init_hstorage_right(mpo, mps), [init_cstorage_right(mps, item) for item in projectors])

function Base.getproperty(m::ProjectedExpectationCache, s::Symbol)
	if s == :state
		return m.mps
	elseif s == :h
		return m.mpo
	elseif s == :env 
		return m.hstorage
	elseif s == :cenvs
		return m.cstorages
	else
		return getfield(m, s)
	end
end


function updateleft!(env::ProjectedExpectationCache, site::Int)
	env.hstorage[site+1] = updateleft(env.hstorage[site], env.mps[site], env.h[site], env.mps[site])
	for l in 1:length(env.cstorages)
	    env.cstorages[l][site+1] = updateleft(env.cstorages[l][site], env.mps[site], env.projectors[l][site])
	end
end


function updateright!(env::ProjectedExpectationCache, site::Int)
	env.hstorage[site] = updateright(env.hstorage[site+1], env.mps[site], env.h[site], env.mps[site])
	for l in 1:length(env.cstorages)
	    env.cstorages[l][site] = updateright(env.cstorages[l][site+1], env.mps[site], env.projectors[l][site])
	end
end
