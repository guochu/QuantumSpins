

# two body gates
struct SWAPGate <: AbstractTwoBodyGate
	positions::Tuple{Int, Int}

	function SWAPGate(key::Tuple{Int, Int})
		new(Tuple(sort([key...])))
	end
end
SWAPGate(c::Int, t::Int) = SWAPGate((c, t))

op(s::SWAPGate) = reshape(SWAP, 2,2,2,2)
shift(s::SWAPGate, i::Int) = SWAPGate(_shift(positions(s), i))
scalar_type(::Type{SWAPGate}) = Float64

Base.adjoint(x::SWAPGate) = x
Base.transpose(x::SWAPGate) = x
Base.conj(x::SWAPGate) = x

struct iSWAPGate <: AbstractTwoBodyGate
	positions::Tuple{Int, Int}

	function iSWAPGate(key::Tuple{Int, Int})
		new(Tuple(sort([key...])))
	end
end
iSWAPGate(c::Int, t::Int) = iSWAPGate((c, t))

op(s::iSWAPGate) = reshape(iSWAP, 2,2,2,2)
shift(s::iSWAPGate, i::Int) = iSWAPGate(_shift(positions(s), i))
scalar_type(::Type{iSWAPGate}) = ComplexF64

Base.transpose(x::iSWAPGate) = x

# two body controled gates
function _get_norm_order(key::NTuple{N, Int}) where N
	seq = sortperm([key...])
	perm = [seq; [s + N for s in seq]]
	return key[seq], perm
end


abstract type AbstractControlGate <: AbstractTwoBodyGate end
original_key(x::AbstractControlGate) = x.original_key
target_op(x::AbstractControlGate) = error("target_op not implement for gate $(typeof(x)).")
# control(x::AbstractControlGate) = original_key(x)[1]
# target(x::AbstractControlGate) = original_key(x)[2]

struct CZGate <: AbstractControlGate
	positions::Tuple{Int, Int}

	function CZGate(key::Tuple{Int, Int})
		new(Tuple(sort([key...])))
	end
end
original_key(x::CZGate) = x.positions
target_op(x::CZGate) = Z
CZGate(c::Int, t::Int) = CZGate((c, t))
CZGate(;control::Int, target::Int) = CZGate(control, target)

op(s::CZGate) = reshape(CZ, 2,2,2,2)
shift(s::CZGate, i::Int) = CZGate(_shift(positions(s), i))
scalar_type(::Type{CZGate}) = Float64

Base.adjoint(x::CZGate) = x
Base.transpose(x::CZGate) = x
Base.conj(x::CZGate) = x

struct CNOTGate <: AbstractControlGate
	original_key::Tuple{Int, Int}
	positions::Tuple{Int, Int}
	perm::Vector{Int}
end
# original_key(x::CNOTGate) = x.original_key
target_op(x::CNOTGate) = X
function CNOTGate(key::Tuple{Int, Int})
	key_new, perm = _get_norm_order(key)
	CNOTGate(key, key_new, perm)
end
CNOTGate(c::Int, t::Int) = CNOTGate((c, t))
CNOTGate(;control::Int, target::Int) = CNOTGate(control, target)

op(s::CNOTGate) = permute(reshape(CNOT, 2,2,2,2), s.perm)
shift(s::CNOTGate, i::Int) = CNOTGate(_shift(original_key(s), i), _shift(positions(s), i), s.perm)
scalar_type(::Type{CNOTGate}) = Float64

Base.adjoint(x::CNOTGate) = x
Base.transpose(x::CNOTGate) = x
Base.conj(x::CNOTGate) = x

# parameteric two body gates
struct CRxGate{T} <:AbstractControlGate
	original_key::Tuple{Int, Int}
	positions::Tuple{Int, Int}
	parameter::T
	perm::Vector{Int}
end
# original_key(x::CRxGate) = x.original_key
target_op(x::CRxGate) = Rx(value(x.parameter))
function CRxGate(key::Tuple{Int, Int}, parameter::T) where T
	key_new, perm = _get_norm_order(key)
	CRxGate(key, key_new, parameter, perm)
end
CRxGate(c::Int, t::Int, parameter::T) where T = CRxGate((c, t), parameter)
CRxGate(;control::Int, target::Int, theta) = CRxGate(control, target, theta)

op(s::CRxGate) = permute(reshape(CONTROL(Rx(value(s.parameter))),2,2,2,2), s.perm)
shift(s::CRxGate, i::Int) = CRxGate(_shift(original_key(s), i), _shift(positions(s), i), s.parameter, s.perm)
scalar_type(::Type{CRxGate{T}}) where {T<:Real} = ComplexF64

struct CRyGate{T} <: AbstractControlGate
	original_key::Tuple{Int, Int}
	positions::Tuple{Int, Int}
	parameter::T
	perm::Vector{Int}
end
# original_key(x::CRyGate) = x.original_key
target_op(x::CRyGate) = Ry(value(x.parameter))
function CRyGate(key::Tuple{Int, Int}, parameter::T) where T
	key_new, perm = _get_norm_order(key)
	CRyGate(key, key_new, parameter, perm)
end
CRyGate(c::Int, t::Int, parameter::T) where T = CRyGate((c, t), parameter)
CRyGate(;control::Int, target::Int, theta) = CRyGate(control, target, theta)

op(s::CRyGate) = permute(reshape(CONTROL(Ry(value(s.parameter))),2,2,2,2), s.perm)
shift(s::CRyGate, i::Int) = CRyGate(_shift(original_key(s), i), _shift(positions(s), i), s.parameter, s.perm)
scalar_type(::Type{CRyGate{T}}) where {T<:Real} = Float64

struct CRzGate{T} <: AbstractControlGate
	original_key::Tuple{Int, Int}
	positions::Tuple{Int, Int}
	parameter::T
	perm::Vector{Int}
end
# original_key(x::CRzGate) = x.original_key
target_op(x::CRzGate) = Rz(value(x.parameter))
function CRzGate(key::Tuple{Int, Int}, parameter::T) where T
	key_new, perm = _get_norm_order(key)
	CRzGate(key, key_new, parameter, perm)
end
CRzGate(c::Int, t::Int, parameter::T) where T = CRzGate((c, t), parameter)
CRzGate(;control::Int, target::Int, theta) = CRzGate(control, target, theta)

op(s::CRzGate) = permute(reshape(CONTROL(Rz(value(s.parameter))),2,2,2,2), s.perm)
shift(s::CRzGate, i::Int) = CRzGate(_shift(original_key(s), i), _shift(positions(s), i), s.parameter, s.perm)
scalar_type(::Type{CRzGate{T}}) where {T<:Real} = ComplexF64

struct CPHASEGate{T} <: AbstractControlGate
	original_key::Tuple{Int, Int}
	positions::Tuple{Int, Int}
	parameter::T
	perm::Vector{Int}
end
# original_key(x::CRzGate) = x.original_key
target_op(x::CPHASEGate) = PHASE(value(x.parameter))
function CPHASEGate(key::Tuple{Int, Int}, parameter::T) where T
	key_new, perm = _get_norm_order(key)
	CPHASEGate(key, key_new, parameter, perm)
end
CPHASEGate(c::Int, t::Int, parameter::Real) = CPHASEGate((c, t), parameter)
CPHASEGate(;control::Int, target::Int, theta::Real) = CPHASEGate(control, target, theta)

op(s::CPHASEGate) = permute(reshape(CONTROL(PHASE(value(s.parameter))),2,2,2,2), s.perm)
shift(s::CPHASEGate, i::Int) = CPHASEGate(_shift(original_key(s), i), _shift(positions(s), i), s.parameter, s.perm)
scalar_type(::Type{CPHASEGate{T}}) where {T<:Real} = ComplexF64

Base.adjoint(x::CPHASEGate) = CPHASEGate(original_key(x), positions(x), -x.parameter, x.perm)
Base.transpose(x::CPHASEGate) = x
Base.conj(x::CPHASEGate) = CPHASEGate(original_key(x), positions(x), -x.parameter, x.perm)


struct FSIMGate <: AbstractTwoBodyGate
	original_key::Tuple{Int, Int}
	positions::Tuple{Int, Int}
	theta::Float64
	phi::Float64
	perm::Vector{Int}
end
original_key(x::FSIMGate) = x.original_key
function FSIMGate(key::Tuple{Int, Int}, theta::Real, phi::Real)
	key_new, perm = _get_norm_order(key)
	FSIMGate(key, key_new, theta, phi, perm)
end
FSIMGate(c::Int, t::Int, theta::Real, phi::Real) = FSIMGate((c, t), theta, phi)
FSIMGate(;control::Int, target::Int, theta::Real, phi::Real) = FSIMGate(control, target, theta, phi)
op(s::FSIMGate) = permute(reshape(FSIM(s.theta, s.phi),2,2,2,2), s.perm)
shift(s::FSIMGate, i::Int) = FSIMGate(_shift(original_key(s), i), _shift(positions(s), i), s.theta, s.phi, s.perm)

Base.transpose(x::FSIMGate) = x


struct GFSIMGate{T1, T2, T3, T4, T5} <: AbstractTwoBodyGate
	original_key::Tuple{Int, Int}
	positions::Tuple{Int, Int}
	theta::T1
	phi::T2
	deltap::T3
	deltam::T4
	deltamoff::T5
	perm::Vector{Int}
end
original_key(x::GFSIMGate) = x.original_key
function GFSIMGate(key::Tuple{Int, Int}, theta, phi, deltap=0., deltam=0., deltamoff=0.)
	key_new, perm = _get_norm_order(key)
	GFSIMGate(key, key_new, theta, phi, deltap, deltam, deltamoff, perm)
end
GFSIMGate(c::Int, t::Int, theta, phi, deltap=0., deltam=0., deltamoff=0.) = GFSIMGate(
	(c, t), theta, phi, deltap, deltam, deltamoff)
GFSIMGate(;control::Int, target::Int, theta, phi, deltap=0., deltam=0., deltamoff=0.) = GFSIMGate(
	control, target, theta, phi, deltap, deltam, deltamoff) 


op(s::GFSIMGate) = permute(reshape(GFSIM(value(s.theta), value(s.phi), 
	value(s.deltap), value(s.deltam), value(s.deltamoff)),2,2,2,2), s.perm)
shift(s::GFSIMGate, i::Int) = GFSIMGate(_shift(original_key(s), i), _shift(positions(s), i), s.theta, 
	s.phi, s.deltap, s.deltam, s.deltamoff, s.perm)


# struct PFSIMGate <: AbstractTwoBodyGate
# 	original_key::Tuple{Int, Int}
# 	key::Tuple{Int, Int}
# 	phi::Float64
# 	perm::Vector{Int}
# end
# original_key(x::PFSIMGate) = x.original_key
# function PFSIMGate(key::Tuple{Int, Int}, phi::Real)
# 	key_new, perm = _get_norm_order(key)
# 	PFSIMGate(key, key_new, phi, perm)
# end
# PFSIMGate(c::Int, t::Int, phi::Real) = PFSIMGate((c, t), phi)
# PFSIMGate(;control::Int, target::Int, phi::Real) = PFSIMGate(control, target, phi)
# op(s::PFSIMGate) = permute(reshape(FSIM(pi/2, s.phi),2,2,2,2), s.perm)
# shift(s::PFSIMGate, i::Int) = PFSIMGate(_shift(original_key(s), i), _shift(key(s), i), s.phi, s.perm)

# FSIMGate(c::Int, t::Int, phi::Real) = PFSIMGate((c, t), phi)
# FSIMGate(;control::Int, target::Int, phi::Real) = PFSIMGate(control, target, phi)

# control gate
struct CONTROLGate{T} <: AbstractControlGate
	original_key::Tuple{Int, Int}
	positions::Tuple{Int, Int}
	target::T
	perm::Vector{Int}
end
# original_key(x::CONTROLGate) = x.original_key
target_op(x::CONTROLGate) = x.target
function CONTROLGate(key::Tuple{Int, Int}, m::AbstractMatrix)
	key_new, perm = _get_norm_order(key)
	return CONTROLGate(key, key_new, m, perm)
end
CONTROLGate(c::Int, t::Int, m::AbstractMatrix) = CONTROLGate((c, t), m)
CONTROLGate(m::AbstractMatrix; control::Int, target::Int) = CONTROLGate(control, target, m)
op(s::CONTROLGate) = permute(reshape(CONTROL(s.target),2,2,2,2), s.perm)
shift(s::CONTROLGate, i::Int) = CONTROLGate(_shift(original_key(s), i), _shift(positions(s), i), s.target, s.perm)

Base.adjoint(x::CONTROLGate) = CONTROLGate(original_key(x), positions(x), target_op(x)', x.perm)
Base.transpose(x::CONTROLGate) = CONTROLGate(original_key(x), positions(x), transpose(target_op(x)), x.perm)
Base.conj(x::CONTROLGate) = CONTROLGate(original_key(x), positions(x), conj(target_op(x)), x.perm)
