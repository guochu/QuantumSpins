

mixed_thermalize(m::QTerm) = superoperator(m, id(m))
mixed_thermalize(h::QuantumOperator) = SuperOperator(QuantumOperator([mixed_thermalize(item) for item in qterms(h)]))
mixed_thermalize(h::MPO) = kron(h, id(h))

function thermal_state(h::Union{QuantumOperator, MPO}; β::Real, stepper::AbstractStepper=TEBDStepper(stepsize=0.05, order=4)) 
	(isa(stepper, TEBDStepper) && isa(h, MPO)) && throw(ArgumentError("TEBD can not be used with MPO."))
	beta = convert(Float64, β)
	((beta >= 0.) && (beta != Inf)) || throw(ArgumentError("β expected to be finite."))
	state = infinite_temperature_state(scalar_type(h), physical_dimensions(h))
	canonicalize!(state, normalize=true)
	(beta == 0.) && return state
	

	# superh = superoperator(h)
	superh = mixed_thermalize(h)
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


const ITIME_EVO_TOL = 1.0e-5

_expec(h::QuantumOperator, state) = expectation(h, state, iscanonical=true)
_expec(h::MPO, state) = expectation(h, state)

function itimeevo!(state::MPS, ham::Union{QuantumOperator, MPO}; stepper::AbstractStepper=TDVPStepper(stepsize=0.1, ishermitian=true), atol::Real=ITIME_EVO_TOL)
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
	energy = _expec(ham, state)
	delta_t = 0.1
	beta = 0.5
	maxitr = 1000
	itr = 1
	vtol = 10 * atol
	err = 1.
	# scaling_down = 0.5
	energies = [energy]
	while (( err >= atol) || (delta_t^(stepper.order) > atol)) && (itr <= maxitr)
		stepper = change_tspan_dt(stepper, tspan=(0., -beta), stepsize=delta_t)
		state, cache = timeevo!(state, ham, stepper, cache)
		canonicalize!(state, normalize=true)
		new_energy = _expec(ham, state)
		# println("current energy is $(new_energy) in the $itr-th iteration, current δ=$delta_t")
		push!(energies, new_energy)
		if new_energy >= energy
			err = 1.
			delta_t *= 0.5
		else
			err = energy - new_energy
			if (err < vtol) && (delta_t^(stepper.order) > atol)
				delta_t *= 0.9
			end
		end
		# if (err < vtol) && (delta_t > vtol)
		# 	# scale down dt if energy is too close
		# 	delta_t *= (scaling_down * energy / new_energy)
		# end
		energy = new_energy
		itr += 1
	end
	(itr > maxitr) && (@warn "can not converge to precision $atol in $maxitr till $(β=beta*maxitr).")
	return energy, state
end
