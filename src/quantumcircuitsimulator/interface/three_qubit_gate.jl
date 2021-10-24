
abstract type AbstractControlControlGate <: AbstractThreeBodyGate end
original_key(x::AbstractControlControlGate) = x.original_key
target_op(x::AbstractControlControlGate) = error("target_op not implement for gate $(typeof(x)).")
# target(x::AbstractControlControlGate) = original_key(x)[3]

struct TOFFOLIGate <: AbstractControlControlGate
	original_key::Tuple{Int, Int, Int}
	positions::Tuple{Int, Int, Int}
	perm::Vector{Int}
end
target_op(x::TOFFOLIGate) = X
function TOFFOLIGate(key::Tuple{Int, Int, Int})
	key_new, perm = _get_norm_order(key)
	TOFFOLIGate(key,key_new, perm)
end
TOFFOLIGate(a::Int, b::Int, c::Int) = TOFFOLIGate((a, b, c))

op(s::TOFFOLIGate) = permute(reshape(TOFFOLI, 2,2,2,2,2,2), s.perm)
shift(s::TOFFOLIGate, i::Int) = TOFFOLIGate(_shift(original_key(s), i), _shift(positions(s), i), s.perm)

Base.adjoint(x::TOFFOLIGate) = x
Base.transpose(x::TOFFOLIGate) = x
Base.conj(x::TOFFOLIGate) = x

struct FREDKINGate <: AbstractThreeBodyGate
	original_key::Tuple{Int, Int, Int}
	positions::Tuple{Int, Int, Int}
	perm::Vector{Int}
end
original_key(x::FREDKINGate) = x.original_key
function FREDKINGate(key::Tuple{Int, Int, Int})
	key_new, perm = _get_norm_order(key)
	FREDKINGate(key, key_new, perm)
end
FREDKINGate(a::Int, b::Int, c::Int) = FREDKINGate((a, b, c))

op(s::FREDKINGate) = permute(reshape(FREDKIN, 2,2,2,2,2,2), s.perm)
shift(s::FREDKINGate, i::Int) = FREDKINGate(_shift(original_key(s), i), _shift(positions(s), i), s.perm)

Base.adjoint(x::FREDKINGate) = x
Base.transpose(x::FREDKINGate) = x
Base.conj(x::FREDKINGate) = x


struct CCPHASEGate <: AbstractControlControlGate
	original_key::Tuple{Int, Int, Int}
	positions::Tuple{Int, Int, Int}
	parameter::Float64
	perm::Vector{Int}
end
# original_key(x::CRzGate) = x.original_key
target_op(x::CCPHASEGate) = PHASE(x.parameter)
function CCPHASEGate(key::Tuple{Int, Int, Int}, parameter::Real)
	key_new, perm = _get_norm_order(key)
	CCPHASEGate(key, key_new, parameter, perm)
end
CCPHASEGate(a::Int, b::Int, c::Int, parameter::Real) = CCPHASEGate((a, b, c), parameter)

op(s::CCPHASEGate) = permute(reshape(CONTROLCONTROL(PHASE(s.parameter)),2,2,2,2,2,2), s.perm)
shift(s::CCPHASEGate, i::Int) = CCPHASEGate(_shift(original_key(s), i), _shift(positions(s), i), s.parameter, s.perm)

Base.adjoint(x::CCPHASEGate) = CCPHASEGate(original_key(x), positions(x), -x.parameter, x.perm)
Base.transpose(x::CCPHASEGate) = x
Base.conj(x::CCPHASEGate) = CCPHASEGate(original_key(x), positions(x), -x.parameter, x.perm)

# control control gate
struct CONTROLCONTROLGate{T} <: AbstractControlControlGate
	original_key::Tuple{Int, Int, Int}
	positions::Tuple{Int, Int, Int}
	target::T
	perm::Vector{Int}
end
# original_key(x::CONTROLCONTROLGate) = x.original_key
target_op(x::CONTROLCONTROLGate) = x.target
function CONTROLCONTROLGate(key::Tuple{Int, Int, Int}, m::AbstractMatrix)
	key_new, perm = _get_norm_order(key)
	return CONTROLCONTROLGate(key, key_new, m, perm)
end
CONTROLCONTROLGate(a::Int, b::Int, c::Int, m::AbstractMatrix) = CONTROLCONTROLGate((a, b, c), m)
op(s::CONTROLCONTROLGate) = permute(reshape(CONTROLCONTROL(s.target),2,2,2,2,2,2), s.perm)
shift(s::CONTROLCONTROLGate, i::Int) = CONTROLCONTROLGate(_shift(original_key(s), i), _shift(positions(s), i), s.target, s.perm)

Base.adjoint(x::CONTROLCONTROLGate) = CONTROLCONTROLGate(original_key(x), positions(x), target_op(x)', x.perm)
Base.transpose(x::CONTROLCONTROLGate) = CONTROLCONTROLGate(original_key(x), positions(x), transpose(target_op(x)), x.perm)
Base.conj(x::CONTROLCONTROLGate) = CONTROLCONTROLGate(original_key(x), positions(x), conj(target_op(x)), x.perm)
