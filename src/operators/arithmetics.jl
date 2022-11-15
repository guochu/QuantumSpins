

"""
	simplify(m::QTerm; atol::Real=1.0e-12) 

Simplifying QTerm by removing the identities
"""
simplify(m::QTerm; atol::Real=1.0e-12) = remove_identities(m; atol=atol)
function remove_identities(m::QTerm; kwargs...)
	_pos = positions(m)
	_ops = op(m)

	new_pos = Int[]
	new_ops = []
	scale = 1.
	for (a, b) in zip(_pos, _ops)
		_is_id, scal = isid(b; kwargs...)
		if _is_id
			scale *= scal
		else
			push!(new_pos, a)
			push!(new_ops, b)
		end
	end
	if isempty(new_pos)
		@warn "QTerm is identity"
		m1 = _ops[1]
		return QTerm(_ops[1], _eye(eltype(m1), size(m1, 1)), coeff=scale * coeff(m)) 
	else
		return QTerm(new_pos, [new_ops...], coeff=scale * coeff(m))
	end
end
simplify(x::SuperTerm; kwargs...) = SuperTerm(simplify(data(x); kwargs...))

function isid(x::MPOTensor; kwargs...)
	s1, s2, s3, s4 = size(x)

	((s1 == s3) && (s2 == s4)) || return false, 0.0

    id = _id4(eltype(x), s1, s2 ) 

    return _is_prop_util(x, id; kwargs...)
end

function _is_prop_util(x::AbstractArray, a::AbstractArray; atol::Real=1.0e-14) 
	scal = dot(a,x)/dot(a,a)
	diff = x-scal*a
	scal = (scal ≈ 0.0) ? 0.0 : scal #shouldn't be necessary (and I don't think it is)
	return norm(diff)<atol,scal
end

function _expm(x::QuantumOperator, dt::Number) 
	is_constant(x) || throw(ArgumentError("input operator should be constant."))
	r = QuantumCircuit()
	for k in keys(x.data)
	    m = qterms(x, k)
	    v = nothing
	    for item in m
	    	tmp = _join_ops(item)
	    	if !isnothing(tmp)
	    		if isnothing(v)
	    			v = tmp
	    		else
	    			v += tmp
	    		end
	    	end
	    end
	    if !isnothing(v)
	    	n = length(k)
	    	push!(r, QuantumGate(k, texp(v * dt, Tuple(1:n), Tuple(n+1:2*n) ) ))
	    end
	end
	return r
end


function matrix(ds::Vector{Int}, m::QTerm)
	is_constant(m) || throw(ArgumentError("QTerm must be constant."))
	opstr = Dict(k=>v for (k, v) in zip(positions(m), op(m)))
	return _prodham_util(ds, opstr) * value(coeff(m))
end

function matrix(x::QuantumOperator)
	is_constant(x) || error("input must be a constant operator.")
	ds = physical_dimensions(x)
	n = prod(ds)
	h = zeros(scalar_type(x), n, n)
	for item in qterms(x)
		h += matrix(ds, item)
	end
	return h
end


function _prodham_util(ds::Vector{Int}, opstr::Dict{Int, <:MPOTensor}) 
	L = length(ds)
	(max(keys(opstr)...) <= L) || error("op str out of bounds")
	i = 1
	ops = []
	left = 1
	for i in 1:L
	    v = get(opstr, i, nothing)
	    if v === nothing
	    	tmp = _eye(left)
	    	Id = _eye(ds[i])
	        v = @tensor tmp4[1,2,3,4] := tmp[1,3] * Id[2,4]
	    else
	    	(size(v, 2) == size(v, 4)==ds[i]) && (size(v, 1)==left) || error("dimension mismatch with dim.")
	    	left = size(v, 3)
	    end
	    push!(ops, v)
	end
	return _kron_chain_ops(ops)
end

_join_ops(s::QTerm) = begin
    is_constant(s) || error("can not join non-constant hamiltonian term.")
    ds = physical_dimensions(s)
    m = _kron_chain_ops(op(s))
    return reshape(m * value(coeff(s)) , Tuple(repeat(ds, 2)))
end 


function _kron_chain_ops(op)
	isempty(op) && error("ops is empty.")
	nb = length(op)
	@assert (size(op[1], 1) == 1) && (size(op[end], 3) == 1)
	m = permute(reshape(op[1], size(op[1])[2:4]), (1,3,2))
	for i in 2:length(op)
		@tensor tmp[1,4,2,6,5] := m[1,2,3] * op[i][3,4,5,6]
		s1, s2, s3, s4, s5 = size(tmp)
		m = reshape(tmp, (s1 * s2, s3 * s4, s5) )
	end
	return reshape(m, size(m)[1:2])
end

