
function is_lower_than(ks, L::Int)
	for key in ks
		if length(key) > L
			return false
		end
	end
	return true
end

function _is_nn_single(key)
	(length(key) != 2) && error("input should be a tuple of 2.")
	return key[1]+1 == key[2]
end

function is_nn(ks)
	for key in ks
		if !_is_nn_single(key)
			return false
		end
	end
	return true
end

function split_nn_ham(ham::QuantumOperator)
	is_nn(Base.keys(ham)) || error("splt nn requires a nearest neighbour hamiltonian.")
	ham_even = typeof(ham)(physical_dimensions(ham))
	ham_odd = typeof(ham)(physical_dimensions(ham))
	for (key, value) in raw_data(ham)
		i, j = key
		(j == i+1) || error("hamiltonian contains non-nearest-neighbour term.")
		if i%2==0
			raw_data(ham_even)[key] = value
		else
			raw_data(ham_odd)[key] = value
		end
	end
	return ham_even, ham_odd
end

function _expm(ham, t::Number, dt::Number)
	if is_constant(ham)
		return _expm(ham, dt)
	else
		return _expm(ham(t), dt)
	end
end 

_td_tebd1order(ham, t, dt) = _expm(ham, t+dt, dt)

function _td_generic_tebd2order(ham, t, dt)
	tmp = _expm(ham, t+dt/2, dt/2)
	circuit = similar(tmp)
	append!(circuit, tmp)
	append!(circuit, reverse(tmp))
	return circuit
end


"""
	used for nearest neighbour two body hamiltonian
"""
function _td_nn_tebd2order(ham_A, ham_B, t, dt)
	mpoevenhalf = _expm(ham_A, t+dt/2, dt/2)
	mpooddone = _expm(ham_B, t+dt/2, dt)
	circuit = similar(mpoevenhalf)
	append!(circuit, mpoevenhalf)
	append!(circuit, mpooddone)
	append!(circuit, mpoevenhalf)
	return circuit
end


_propagator_AB_2order(ham_A, ham_B, tf::Number, ti::Number) = _td_nn_tebd2order(
	ham_A, ham_B, ti, tf-ti)
# _propagator_AB_2order(ham_A, ham_B, dt::Number) = _nn_tebd2order(ham_A, ham_B, dt)
_propagator_generic_2order(ham, tf::Number, ti::Number) = _td_generic_tebd2order(ham, ti, tf-ti)
# _propagator_generic_2order(ham, dt::Number) = _generic_tebd2order(ham, dt)

function propagator_2order(ham::QuantumOperator, tf::Number, ti::Number)
	ks = keys(ham)
	if is_lower_than(ks, 2)
		if is_nn(ks)
		    ham_A, ham_B = split_nn_ham(ham)
		    return _propagator_AB_2order(ham_A, ham_B, tf, ti)
		end
	end
	return _propagator_generic_2order(ham, tf, ti)
end

function propagator_2order(ham::QuantumOperator, dt::Number)
	ks = keys(ham)
	if is_lower_than(ks, 2)
		if is_nn(ks)
		    ham_A, ham_B = split_nn_ham(ham)
		    return _propagator_AB_2order(ham_A, ham_B, dt)
		end
	end
	return _propagator_generic_2order(ham, dt)
end


function _propagator_impl(ham::QuantumOperator, tf::Number, ti::Number, order::Int=1)
	(order == 1) && return propagator_2order(ham, tf, ti)
	# circuit = GenericCircuit1D()
	p = order - 1	
	sp = 1/(4 - 4^(1/(2*p+1)))
	dt = tf - ti
	circuit = _propagator_impl(ham, ti + sp * dt, ti, p)
	append!(circuit, _propagator_impl(ham, ti + 2 * sp * dt, ti + sp * dt, p) )
	append!(circuit, _propagator_impl(ham, ti + (1 - 2*sp) * dt, ti + 2 * sp * dt, p) ) 
	append!(circuit, _propagator_impl(ham, ti + (1-sp) * dt, ti + (1-2*sp) * dt, p) ) 
	append!(circuit, _propagator_impl(ham, ti + dt, ti + (1-sp) * dt, p) ) 
	# return fuse_gates(circuit)
	return circuit
end

"""
	trotter_propagator(ham::QuantumOperator, tspan::Tuple{<:Number, <:Number}; order::Int=2, dt::Number=0.01)
	algorithm reference: "Higher order decompositions of ordered operator exponentials"
"""
function trotter_propagator(ham::QuantumOperator, tspan::Tuple{<:Number, <:Number}; order::Int=2, stepsize::Number=0.01)
	ham = absorb_one_bodies(ham)
	p = div(order, 2)
	(p * 2 == order) || throw(ArgumentError("only even order supported.")) 
	dt = stepsize
	ti, tf = tspan
	tdiff = tf - ti
	nsteps = round(Int, abs(tdiff / dt)) 	
	if nsteps == 0
		nsteps = 1
	end
	dt = tdiff / nsteps
	local circuit
	for i in 1:nsteps
		if @isdefined circuit
			append!(circuit, _propagator_impl(ham, ti + i * dt, ti + (i-1)*dt, p))
		else
			circuit = _propagator_impl(ham, ti + i * dt, ti + (i-1)*dt, p)
		end
	end
	# return fuse_gates(circuit)
	return circuit
end

trotter_propagator(ham::SuperOperatorBase, tspan::Tuple{<:Number, <:Number}; kwargs...) = trotter_propagator(ham.data, tspan; kwargs...)

# """
# 	infinite tebd circuit
# 	period is the period of the operator, len is the period of the state.
# """
# function infinite_trotter_propagator(ham::QuantumOperator, tspan::Tuple{<:Number, <:Number}; period::Int=1, len::Int=length(ham), kwargs...)
# 	(mod(len, period) == 0) || throw(DimensionMismatch())
# 	(len >= interaction_range(ham)) || throw(ArgumentError("period state must be greater equal than the interaction range."))
# 	nperiod = len ÷ period
# 	ham2 = QuantumOperator([shift(item, (i-1) * period) for item in qterms(ham) if !(all(positions(item) .> period)) for i in 1:nperiod])
# 	return trotter_propagator(ham2, tspan; kwargs...)
# end 

