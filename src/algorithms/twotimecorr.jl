# two time correlation for pure state and density matrices

_time_reversal(t::Number) = -conj(t)
_time_reversal(a::Tuple{<:Number, <:Number}) = (_time_reversal(a[1]), _time_reversal(a[2]))

function _unitary_tt_corr_at_b(h, A::MPO, B::MPO, state, times, stepper)
	state_right = B * state
	canonicalize!(state_right)
	state_left = copy(state)

	result = scalar_type(state)[]
	local cache_left, cache_right	
	for i in 1:length(times)	
		# println("state norm $(norm(state_left)), $(norm(state_right)).")
		tspan = (i == 1) ? (0., times[1]) : (times[i-1], times[i])
		if abs(tspan[2] - tspan[1]) > 0.
			stepper_right = change_tspan_dt(stepper, tspan=tspan)
			stepper_left = change_tspan_dt(stepper, tspan=_time_reversal(tspan))
			(@isdefined cache_left) || (cache_left = timeevo_cache(h, stepper_left, state_left))
			(@isdefined cache_right) || (cache_right = timeevo_cache(h, stepper_right, state_right))
			state_left, cache_left = timeevo!(state_left, h, stepper_left, cache_left)
			state_right, cache_right = timeevo!(state_right, h, stepper_right, cache_right)
		end
		push!(result, expectation(state_left, A, state_right))
	end
	return result
end

function _unitary_tt_corr_a_bt(h, A::MPO, B::MPO, state, times, stepper)
	state_right = copy(state)
	state_left = A' * state
	canonicalize!(state_left)

	result = scalar_type(state)[]
	local cache_left, cache_right	
	for i in 1:length(times)	
		tspan = (i == 1) ? (0., times[1]) : (times[i-1], times[i])
		if abs(tspan[2] - tspan[1]) > 0.
			stepper_right = change_tspan_dt(stepper, tspan=tspan)
			stepper_left = change_tspan_dt(stepper, tspan=_time_reversal(tspan))
			(@isdefined cache_left) || (cache_left = timeevo_cache(h, stepper_left, state_left))
			(@isdefined cache_right) || (cache_right = timeevo_cache(h, stepper_right, state_right))
			state_left, cache_left = timeevo!(state_left, h, stepper_left, cache_left)
			state_right, cache_right = timeevo!(state_right, h, stepper_right, cache_right)

		end
		push!(result, expectation(state_left, B, state_right))
	end
	return result
end


"""
	correlation_2op_1t(h::QuantumOperator, a::QuantumOperator, b::QuantumOperator, state::MPS, times::Vector{<:Real}, stepper::AbstractStepper; 
	reverse::Bool=false) 
	for a unitary system with hamiltonian h, compute <a(t)b> if revere=false and <a b(t)> if reverse=true
	for an open system with superoperator h, and a, b to be normal operators, compute <a(t)b> if revere=false and <a b(t)> if reverse=true.
	For open system see definitions of <a(t)b> or <a b(t)> on Page 146 of Gardiner and Zoller (Quantum Noise)
"""
function correlation_2op_1t(h::Union{QuantumOperator, AbstractMPO}, a::AbstractMPO, b::AbstractMPO, state::MPS, times::Vector{<:Real};
	stepper::AbstractStepper=TEBDStepper(tspan=(0., 0.01), stepsize=0.01), reverse::Bool=false)
	if scalar_type(state) <: Real
		state = complex(state)
	end
	times = -im .* times
	reverse ? _unitary_tt_corr_a_bt(h, a, b, state, times, stepper) : _unitary_tt_corr_at_b(h, a, b, state, times, stepper)
end

"""
	correlation_2op_1τ(h::QuantumOperator, a::QuantumOperator, b::QuantumOperator, state::MPS, times::Vector{<:Real}, stepper::AbstractStepper; 
	reverse::Bool=false) 
	for a unitary system with hamiltonian h, compute <a(τ)b> if revere=false and <a b(τ)> if reverse=true
"""
function correlation_2op_1τ(h::Union{QuantumOperator, AbstractMPO}, a::AbstractMPO, b::AbstractMPO, state::MPS, times::Vector{<:Real};
	stepper::AbstractStepper=TEBDStepper(tspan=(0., 0.01), stepsize=0.01), reverse::Bool=false)
	times = -times
	reverse ? _unitary_tt_corr_a_bt(h, a, b, state, times, stepper) : _unitary_tt_corr_at_b(h, a, b, state, times, stepper)
end


function _open_tt_corr_at_b(h, A::MPO, B::MPO, state, times, stepper)
	state_right = B * state
	canonicalize!(state_right)

	result = scalar_type(state_right)[]
	local cache_right
	for i in 1:length(times)	
		tspan = (i == 1) ? (0., times[1]) : (times[i-1], times[i])
		if tspan[2] - tspan[1] > 0.
			stepper = change_tspan_dt(stepper, tspan=tspan)
			(@isdefined cache_right) || (cache_right = timeevo_cache(h, stepper, state_right))
			state_right, cache_right = timeevo!(state_right, h, stepper, cache_right)
		end
		push!(result, expectation(A, state_right))
	end
	return result
end 

function _open_tt_corr_a_bt(h, A::MPO, B::MPO, state, times, stepper)
	state_right = A * state
	canonicalize!(state_right)

	result = scalar_type(state_right)[]
	local cache_right
	for i in 1:length(times)	
		tspan = (i == 1) ? (0., times[1]) : (times[i-1], times[i])
		if tspan[2] - tspan[1] > 0.
			stepper = change_tspan_dt(stepper, tspan=tspan)
			(@isdefined cache_right) || (cache_right = timeevo_cache(h, stepper, state_right))
			state_right, cache_right = timeevo!(state_right, h, stepper, cache_right)
		end
		push!(result, expectation(B, state_right) )
	end
	return result
end 


function correlation_2op_1t(h::Union{SuperOperatorBase, AbstractMPO}, a::MPO, b::MPO, state::DensityOperatorMPS, times::Vector{<:Real}; 
	stepper::AbstractStepper=TEBDStepper(tspan=(0., 0.01), stepsize=0.01), reverse::Bool=false)
	# (h.fuser === ⊠) || throw(ArgumentError("only fuser ⊠ is supported here."))
	iden = id(b)
	mpo_b = kron(b, iden) 
	if reverse
		return _open_tt_corr_a_bt(h, kron(iden, transpose(a)), mpo_b, state, times, stepper)
	else
		return _open_tt_corr_at_b(h, kron(a, iden), mpo_b, state, times, stepper)
	end
end









