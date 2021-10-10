

function compute_step_size(t::Number, dt::Number)
	n = round(Int, abs(t / dt)) 
	if n == 0
		n = 1
	end
	return n, t / n
end


abstract type AbstractStepper end

struct TEBDStepper{T<:Number, C <: TruncationScheme} <: AbstractStepper
	tspan::Tuple{T, T}
	stepsize::T
	n::Int
	order::Int
	trunc::C
end

function TEBDStepper(;stepsize::Number, tspan::Tuple{<:Number, <:Number}=(0., stepsize), order::Int=2, trunc::TruncationScheme=Default_Truncation)
	ti, tf = tspan
	δ = tf - ti
	n, stepsize = compute_step_size(δ, stepsize)
	T = promote_type(typeof(ti), typeof(tf), typeof(stepsize))
	return TEBDStepper((convert(T, ti), convert(T, tf)), convert(T, stepsize), n, order, trunc)
end

function Base.getproperty(x::TEBDStepper, s::Symbol)
	if s == :δ
		return x.tspan[2] - x.tspan[1]
	else
		getfield(x, s)
	end
end

change_tspan_dt(x::TEBDStepper; tspan::Tuple{<:Number, <:Number}, stepsize::Number=x.stepsize) = TEBDStepper(tspan=tspan, stepsize=stepsize, order=x.order, trunc=x.trunc)


struct TDVPStepper{T<:Number} <: AbstractStepper
	tspan::Tuple{T, T}
	n::Int
	alg::TDVP{T}
end

function Base.getproperty(x::TDVPStepper, s::Symbol)
	if s == :δ
		return x.tspan[2] - x.tspan[1]
	elseif s == :order
		return 2
	elseif s == :stepsize
		return x.alg.stepsize
	else
		getfield(x, s)
	end
end

function _change_stepsize(x::TDVP, stepsize::Number)
	if !(x.stepsize ≈ stepsize)
		x = TDVP(stepsize=stepsize, ishermitian=x.ishermitian, verbosity=x.verbosity)
	end
	return x
end

function TDVPStepper(;stepsize::Number, tspan::Tuple{<:Number, <:Number}=(0., stepsize), ishermitian::Bool=false, verbosity::Int=0)
	ti, tf = tspan
	δ = tf - ti	
	n, stepsize = compute_step_size(δ, stepsize)
	T = promote_type(typeof(ti), typeof(tf), typeof(stepsize))
	return TDVPStepper((convert(T, ti), convert(T, tf)), n, TDVP(stepsize=convert(T, stepsize), ishermitian=ishermitian, verbosity=verbosity))
end

change_tspan_dt(x::TDVPStepper; tspan::Tuple{<:Number, <:Number}, stepsize::Number=x.stepsize) = TDVPStepper(
	tspan=tspan, stepsize=stepsize, ishermitian=x.alg.ishermitian, verbosity=x.alg.verbosity)


mutable struct HomogeousTEBDCache{H<:QuantumOperator, C<:QuantumCircuit, S<:TEBDStepper} <: AbstractCache
	h::H
	circuit::C
	stepper::S
end

function HomogeousTEBDCache(h::QuantumOperator, stepper::TEBDStepper)
	is_constant(h) || throw(ArgumentError("const hamiltonian expected."))
	circuit = fuse_gates(repeat(trotter_propagator(h, (0., stepper.stepsize), order=stepper.order, stepsize=stepper.stepsize), stepper.n))
	return HomogeousTEBDCache(h, circuit, stepper)
end

function recalculate!(x::HomogeousTEBDCache, h::QuantumOperator, stepper::TEBDStepper)
	if !((x.h === h) && (stepper.δ==x.stepper.δ) && (stepper.stepsize==x.stepper.stepsize) && (stepper.order==x.stepper.order) )
		is_constant(h) || throw(ArgumentError("const hamiltonian expected."))
		return HomogeousTEBDCache(h, fuse_gates(repeat(trotter_propagator(h, (0., stepper.stepsize), order=stepper.order, stepsize=stepper.stepsize), stepper.n)), stepper)
	else
		return x
	end
end

# function recalculate!(x::HomogeousTEBDCache, stepper::TEBDStepper)
# 	if !((stepper.δ==x.stepper.δ) && (stepper.stepsize==x.stepper.stepsize) && (stepper.order==x.stepper.order) )
# 		x.circuit = fuse_gates(repeat(trotter_propagator(x.h, (0., stepper.stepsize), order=stepper.order, stepsize=stepper.stepsize), stepper.n))
# 		x.stepper = stepper
# 	end
# end

function make_step!(h::QuantumOperator, stepper::TEBDStepper, state::MPS, x::HomogeousTEBDCache)
	recalculate!(x, h, stepper)
	apply!(x.circuit, state, trunc=x.stepper.trunc)
	return state, x
end


mutable struct InhomogenousTEBDCache{H<:QuantumOperator, C<:QuantumCircuit, S<:TEBDStepper} <: AbstractCache
	h::H
	circuit::C
	stepper::S	
end

function InhomogenousTEBDCache(h::QuantumOperator, stepper::TEBDStepper)
	circuit = fuse_gates(trotter_propagator(h, stepper.tspan, order=stepper.order, stepsize=stepper.stepsize))
	return InhomogenousTEBDCache(h, circuit, stepper)
end


function recalculate!(x::InhomogenousTEBDCache, h::QuantumOperator, stepper::TEBDStepper)
	if !((x.h === h) && (stepper.tspan==x.stepper.tspan) && (stepper.stepsize==x.stepper.stepsize) && (stepper.order==x.stepper.order) )
		return InhomogenousTEBDCache(h, fuse_gates(trotter_propagator(h, stepper.tspan, order=stepper.order, stepsize=stepper.stepsize)), stepper)
	else
		return x
	end
end

# function recalculate!(x::InhomogenousTEBDCache, stepper::TEBDStepper)
# 	if !((stepper.tspan==x.stepper.tspan) && (stepper.stepsize==x.stepper.stepsize) && (stepper.order==x.stepper.order) )
# 		x.circuit = fuse_gates(trotter_propagator(x.h, stepper.tspan, order=stepper.order, stepsize=stepper.stepsize))
# 		x.stepper = stepper
# 	end
# end

function make_step!(h::QuantumOperator, stepper::TEBDStepper, state::MPS, x::InhomogenousTEBDCache)
	x = recalculate!(x, h, stepper)
	apply!(x.circuit, state, trunc=x.stepper.trunc)
	return state, x
end

TEBDCache(h::QuantumOperator, stepper::TEBDStepper) = is_constant(h) ? HomogeousTEBDCache(h, stepper) : InhomogenousTEBDCache(h, stepper)


mutable struct HomogeousTDVPCache{E<:ExpectationCache, S<:TDVPStepper}
	env::E
	stepper::S
end

HomogeousTDVPCache(h::MPO, stepper::TDVPStepper, state::MPS) = HomogeousTDVPCache(environments(h, state), stepper)
TDVPCache(h::MPO, stepper::TDVPStepper, state::MPS) = HomogeousTDVPCache(h, stepper, state)

function recalculate!(x::HomogeousTDVPCache, h::MPO, stepper::TDVPStepper, state::MPS)
	if !((x.env.state === state) && (x.env.h === h))
		return HomogeousTDVPCache(environments(h, state), stepper)
	else
		return HomogeousTDVPCache(x.env, stepper)
	end
end

function make_step!(h::MPO, stepper::TDVPStepper, state::MPS, x::HomogeousTDVPCache)
	x = recalculate!(x, h, stepper, state)
	for i in 1:stepper.n
		sweep!(x.env, stepper.alg)
	end
	return state, x
end

mutable struct HomogeousHamTDVPCache{H<:QuantumOperator, E<:ExpectationCache, S<:TDVPStepper}
	h::H
	env::E
	stepper::S
end
function HomogeousTDVPCache(h::QuantumOperator, stepper::TDVPStepper, state::MPS) 
	mpo = MPO(h)
	env = environments(mpo, state)
	return HomogeousHamTDVPCache(h, env, stepper)
end

function recalculate!(x::HomogeousHamTDVPCache, h::QuantumOperator, stepper::TDVPStepper, state::MPS)
	if !((x.env.state === state) && (x.h === h))
		return HomogeousTDVPCache(h, stepper, state)
	else
		return HomogeousHamTDVPCache(x.h, x.env, stepper)
	end	
end

function make_step!(h::QuantumOperator, stepper::TDVPStepper, state::MPS, x::HomogeousHamTDVPCache)
	x = recalculate!(x, h, stepper, state)
	for i in 1:stepper.n
		sweep!(x.env, stepper.alg)
	end
	return state, x
end


mutable struct InhomogenousHamTDVPCache{H<:QuantumOperator, S<:TDVPStepper}
	h::H
	stepper::S
end

InhomogenousTDVPCache(h::QuantumOperator, stepper::TDVPStepper, state::MPS) = InhomogenousHamTDVPCache(h, stepper)

function recalculate!(x::InhomogenousHamTDVPCache, h::QuantumOperator, stepper::TDVPStepper, state::MPS)
	return InhomogenousHamTDVPCache(h, stepper)
end

function make_step!(h::QuantumOperator, stepper::TDVPStepper, state::MPS, x::InhomogenousHamTDVPCache)
	x = recalculate!(x, h, stepper, state)
	t_start = stepper.tspan[1]
	for i in 1:stepper.n
		t = t_start + (i-1) * stepper.stepsize + stepper.stepsize/2
		mpo = MPO(x.h(t), alg=SVDCompression())
		env = environments(mpo, state)
		sweep!(env, stepper.alg)
	end
	return state, x
end

function TDVPCache(h::QuantumOperator, stepper::TDVPStepper, state::MPS) 
	is_constant(h) ? HomogeousTDVPCache(h, stepper, state) : InhomogenousTDVPCache(h, stepper, state)
end
TDVPCache(h::SuperOperatorBase, stepper::TDVPStepper, state::DensityOperatorMPS) = TDVPCache(h.data, stepper, state.data)

timeevo_cache(h::QuantumOperator, stepper::TEBDStepper, state::MPS) = TEBDCache(h, stepper)
timeevo_cache(h::SuperOperatorBase, stepper::TEBDStepper, state::DensityOperatorMPS) = TEBDCache(h.data, stepper)
timeevo_cache(h::MPO, stepper::TDVPStepper, state::MPS) = TDVPCache(h, stepper, state)
timeevo_cache(h::MPO, stepper::TDVPStepper, state::DensityOperatorMPS) = TDVPCache(h, stepper, state.data)
timeevo_cache(h::QuantumOperator, stepper::TDVPStepper, state::MPS) = TDVPCache(h, stepper, state)
timeevo_cache(h::SuperOperatorBase, stepper::TDVPStepper, state::DensityOperatorMPS) = TDVPCache(h, stepper, state)

timeevo!(state::MPS, h::QuantumOperator, stepper::TEBDStepper, cache=TEBDCache(h, stepper)) = make_step!(h, stepper, state, cache)
function timeevo!(state::DensityOperatorMPS, h::SuperOperatorBase, stepper::TEBDStepper, cache=TEBDCache(h.data, stepper))
	make_step!(h.data, stepper, state.data, cache)
	return state, cache
end 
timeevo!(state::MPS, h::Union{MPO, QuantumOperator}, stepper::TDVPStepper, cache=TDVPCache(h, stepper, state)) = make_step!(h, stepper, state, cache)
function timeevo!(state::DensityOperatorMPS, h::MPO, stepper::TDVPStepper, cache=TDVPCache(h, stepper, state.data))
	make_step!(h, stepper, state.data, cache)
	return state, cache
end
function timeevo!(state::DensityOperatorMPS, h::SuperOperatorBase, stepper::TDVPStepper, cache=TDVPCache(h, stepper, state.data))
	make_step!(h.data, stepper, state.data, cache)
	return state, cache
end


