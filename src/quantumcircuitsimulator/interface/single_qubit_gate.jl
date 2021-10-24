value(x::Number) = x

# one body gates
struct XGate <: AbstractOneBodyGate
	positions::Tuple{Int}
end
XGate(pos::Int) = XGate((pos,))
op(s::XGate) = X
shift(s::XGate, i::Int) = XGate(_shift(positions(s), i))
scalar_type(::Type{XGate}) = Float64

Base.adjoint(x::XGate) = x
Base.transpose(x::XGate) = x
Base.conj(x::XGate) = x

struct YGate <: AbstractOneBodyGate
	positions::Tuple{Int}
end
YGate(pos::Int) = YGate((pos,))
op(s::YGate) = Y
shift(s::YGate, i::Int) = YGate(_shift(positions(s), i))
scalar_type(::Type{YGate}) = Float64

struct ZGate <: AbstractOneBodyGate
	positions::Tuple{Int}
end
ZGate(pos::Int) = ZGate((pos,))
op(s::ZGate) = Z
shift(s::ZGate, i::Int) = ZGate(_shift(positions(s), i))
scalar_type(::Type{ZGate}) = Float64

struct SGate <: AbstractOneBodyGate
	positions::Tuple{Int}
end
SGate(pos::Int) = SGate((pos,))
op(s::SGate) = S
shift(s::SGate, i::Int) = SGate(_shift(positions(s), i))
scalar_type(::Type{SGate}) = ComplexF64

struct SqrtXGate <: AbstractOneBodyGate
	positions::Tuple{Int}
end
SqrtXGate(pos::Int) = SqrtXGate((pos,))
op(s::SqrtXGate) = Xh
shift(s::SqrtXGate, i::Int) = SqrtXGate(_shift(positions(s), i))
scalar_type(::Type{SqrtXGate}) = ComplexF64

struct SqrtYGate <: AbstractOneBodyGate
	positions::Tuple{Int}
end
SqrtYGate(pos::Int) = SqrtYGate((pos,))
op(s::SqrtYGate) = Yh
shift(s::SqrtYGate, i::Int) = SqrtYGate(_shift(positions(s), i))
scalar_type(::Type{SqrtYGate}) = ComplexF64

struct HGate <: AbstractOneBodyGate
	positions::Tuple{Int}
end
HGate(pos::Int) = HGate((pos,))
op(s::HGate) = H
shift(s::HGate, i::Int) = HGate(_shift(positions(s), i))
scalar_type(::Type{HGate}) = Float64

struct TGate <: AbstractOneBodyGate
	positions::Tuple{Int}
end
TGate(pos::Int) = TGate((pos,))
op(s::TGate) = T
shift(s::TGate, i::Int) = TGate(_shift(positions(s), i))
scalar_type(::Type{TGate}) = ComplexF64

# parameteric one body gate
struct RxGate{T} <: AbstractOneBodyGate
	positions::Tuple{Int}
	parameter::T
end
RxGate(pos::Int, p) = RxGate((pos,), p)
op(s::RxGate) = Rx(value(s.parameter))
shift(s::RxGate, i::Int) = RxGate(_shift(positions(s), i), s.parameter)
scalar_type(::Type{RxGate{T}}) where {T<:Real} = ComplexF64

struct RyGate{T} <: AbstractOneBodyGate
	positions::Tuple{Int}
	parameter::T
end
RyGate(pos::Int, p) = RyGate((pos,), p)
op(s::RyGate) = Ry(value(s.parameter))
shift(s::RyGate, i::Int) = RyGate(_shift(positions(s), i), s.parameter)
scalar_type(::Type{RyGate{T}}) where {T<:Real} = Float64

struct RzGate{T} <: AbstractOneBodyGate
	positions::Tuple{Int}
	parameter::T
end
RzGate(pos::Int, p) = RzGate((pos,), p)
op(s::RzGate) = Rz(value(s.parameter))
shift(s::RzGate, i::Int) = RzGate(_shift(positions(s), i), s.parameter)
scalar_type(::Type{RzGate{T}}) where {T<:Real} = ComplexF64

struct PHASEGate{T} <: AbstractOneBodyGate
	positions::Tuple{Int}
	parameter::T
end
PHASEGate(pos::Int, p) = PHASEGate((pos,), p)
op(s::PHASEGate) = PHASE(value(s.parameter))
shift(s::PHASEGate, i::Int) = PHASEGate(_shift(positions(s), i), s.parameter)
scalar_type(::Type{PHASEGate{T}}) where {T<:Real} = ComplexF64

Base.adjoint(x::PHASEGate) = PHASEGate(_shift(positions(s), i), -x.parameter)
Base.transpose(x::PHASEGate) = x
Base.conj(x::PHASEGate) = PHASEGate(_shift(positions(s), i), -x.parameter)
