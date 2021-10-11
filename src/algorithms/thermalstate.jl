

_mixed_thermalize(h::QuantumOperator) = SuperOperatorBase(QuantumOperator([superoperator(item, id(item)) for item in qterms(h)]))
_mixed_thermalize(h::MPO) = kron(h, id(h))

function thermal_state(h::Union{QuantumOperator, MPO}; β::Real, stepper::AbstractStepper=TEBDStepper(stepsize=0.05, order=4), D::Union{Int, Nothing}=nothing) 
	(isa(stepper, TEBDStepper) && isa(h, MPO)) && throw(ArgumentError("TEBD can not be used with MPO."))
	beta = convert(Float64, β)
	((beta >= 0.) && (beta != Inf)) || throw(ArgumentError("β expected to be finite."))
	state = infinite_temperature_state(scalar_type(h), physical_dimensions(h))
	(beta == 0.) && return state
	if isa(stepper, TDVPStepper)
		isa(D, Int) || throw(ArgumentError("D must be given in case of TDVPStepper."))
		state = increase_bond!(state, D=D)
	end
	canonicalize!(state, normalize=true)

	# superh = superoperator(h)
	superh = _mixed_thermalize(h)
	delta = 0.1


	nsteps, delta = compute_step_size(beta, delta)
	local cache
	for i in 1:nsteps
		tspan = ( -(i-1) * delta, -i * delta )
		stepper = change_tspan_dt(stepper, tspan=tspan)
		(@isdefined cache) || (cache = timeevo_cache(superh, stepper, state))
		state, cache = timeevo!(state, superh, stepper, cache)
		canonicalize!(state, normalize=true)
	end
	return normalize!(state)
end

# function evolve_and_normalize(state, h, stepper, cache)
# end

const ITIME_EVO_TOL = 1.0e-6

function itimeevo!(state::MPS, ham::QuantumOperator; stepper::AbstractStepper=TDVPStepper(stepsize=0.1, ishermitian=true), atol::Real=ITIME_EVO_TOL)
	# preparation evolution
	stepper = change_tspan_dt(stepper, tspan=(0., -1), stepsize=1)
	state, cache = timeevo!(state, ham, stepper)
	canonicalize!(state, normalize=true)
	# warmup_sweeps = 2
	# for i in 1:warmup_sweeps - 1
	# 	state, cache = timeevo!(state, ham, stepper, cache)
	# 	canonicalize!(state, normalize=true)
	# end

	# imaginary time evolution
	energy = expectation(ham, state, iscanonical=true)
	new_energy = energy - 2 * atol
	delta_t = 0.1
	beta = 0.5
	maxitr = 1000
	itr = 1
	vtol = 10^(stepper.order) * atol
	while ( energy - new_energy >= atol) && (itr <= maxitr)
		if (energy - new_energy < vtol) && (delta_t > vtol)
			# scale down dt if energy is too close
			delta_t *= (0.9 * energy / new_energy)
		end
		energy = new_energy
		stepper = change_tspan_dt(stepper, tspan=(0., -beta), stepsize=delta_t)
		state, cache = timeevo!(state, ham, stepper, cache)
		canonicalize!(state, normalize=true)
		new_energy = expectation(ham, state, iscanonical=true)
		println("current energy is $(new_energy) in the $itr-th iteration, current δ=$delta_t")
		itr += 1
	end
	(itr > maxitr) && (@warn "can not converge to precision $atol in $maxitr till $(β=beta*maxitr).")
	return new_energy, state
end
