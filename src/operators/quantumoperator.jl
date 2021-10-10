abstract type AbstractQuantumOperator end

const QOP_DATA_TYPE{T} = Dict{Tuple{Int, Vararg{Int, N} where N}, Vector{Tuple{Vector{Matrix{T}}, AbstractCoefficient}}}

struct QuantumOperator{T<:Number} <: AbstractQuantumOperator
	physpaces::Vector{Union{Int, Missing}}
	data::QOP_DATA_TYPE{T}
end

QuantumOperator{T}(ds::Vector{Union{Int, Missing}}) where {T <: Number} = QuantumOperator{T}(ds, 
	Dict{Tuple{Int, Vararg{Int, N} where N}, Vector{Tuple{Vector{Matrix{T}}, AbstractCoefficient}}}() )
QuantumOperator{T}(ds::Vector{Int}) where {T <: Number} = QuantumOperator{T}(convert(Vector{Union{Int, Missing}}, ds))
QuantumOperator{T}() where {T <: Number} = QuantumOperator{T}(Vector{Union{Int, Missing}}())

function QuantumOperator(ms::Vector{<:QTerm})
	T = Float64
	for item in ms
		T = promote_type(T, scalar_type(item))
	end
	r = QuantumOperator{T}()
	for item in ms
		add!(r, item)
	end
	return r
end

raw_data(x::QuantumOperator) = x.data

scalar_type(::Type{QuantumOperator{T}}) where T = T
scalar_type(x::QuantumOperator) = scalar_type(typeof(x))

Base.copy(x::QuantumOperator) = QuantumOperator(copy(x.physpaces), x.data)
Base.isempty(s::QuantumOperator) = isempty(s.data)
Base.length(x::QuantumOperator) = length(x.physpaces)
Base.similar(s::QuantumOperator{T}) where T = QuantumOperator{T}()
Base.keys(x::QuantumOperator) = keys(x.data)
physical_dimensions(x::QuantumOperator) = convert(Vector{Int}, x.physpaces)

function is_constant(x::QuantumOperator)
	for (k, v) in x.data
		for (m, c) in v
			is_constant(c) || return false
		end
	end
	return true
end

interaction_range(x::QuantumOperator) = maximum([_interaction_range(k) for k in keys(x.data)])

data_type(x::QuantumOperator{T}) where T = QOP_DATA_TYPE{T}
function (x::QuantumOperator)(t::Number)
	r = data_type(x)()
	for (k, v) in x.data
		vr =  typeof(v)()
		for (m, c) in v
			push!(vr, (m, coeff(c(t)) ))
		end
		r[k] = vr
	end
	return QuantumOperator(copy(x.physpaces), r)
end


function Base.:*(x::QuantumOperator, y::Union{Number, Function, Coefficient}) 
	y = Coefficient(y)
	T = promote_type(scalar_type(x), scalar_type(y))
	r = QOP_DATA_TYPE{T}()
	for (k, v) in x.data
		vr =  typeof(v)()
		for (t, c) in v
			push!(vr, (t, c * y))
		end
		r[k] = vr
	end
	return QuantumOperator(copy(x.physpaces), r)
end

Base.:*(y::Union{Number, Function, Coefficient}, x::QuantumOperator) = x * y
Base.:/(x::QuantumOperator, y::Union{Number, Function, Coefficient}) = x * (1 / Coefficient(y))
Base.:+(x::QuantumOperator) = x
Base.:-(x::QuantumOperator) = x * (-1)

function _merge_spaces(x, y)
	L = max(length(x), length(y))
	r = copy(x)
	resize!(r, L)
	for i in 1:L
		if i <= length(y)
			if (!isassigned(r, i)) || ismissing(r[i])
				if !ismissing(y[i])
					r[i] = y[i]
				end
			else
				if !ismissing(y[i])
					(r[i] == y[i]) || throw(DimensionMismatch())
				end
			end
		end
	end
	return r
end

function Base.:+(x::QuantumOperator, y::QuantumOperator)
	T = promote_type(scalar_type(x), scalar_type(y)) 
	z = QuantumOperator{T}(_merge_spaces(x.physpaces, y.physpaces), convert(QOP_DATA_TYPE{T}, copy(x.data)))
	for (k, v) in y.data
	   	tmp = get!(z.data, k, Vector{Tuple{Vector{Matrix{T}}, AbstractCoefficient}}())
	   	append!(tmp, convert(Vector{Tuple{Vector{Matrix{T}}, AbstractCoefficient}}, v))
	end	
	return z
end
Base.:-(x::QuantumOperator, y::QuantumOperator) = x + (-y)

function shift(x::QuantumOperator, n::Int)
	s = similar(x.physpaces, length(x) + n)
	s[1:n] .= missing
	s[(n+1):end] .= x.physpaces
	return QuantumOperator(s, data_type(x)( k .+ n =>v for (k, v) in x.data) )
end 


"""
	add!(x::QuantumOperator{M}, m::QTerm)
	adding a new term into the quantum operator
"""
function add!(x::QuantumOperator, m::QTerm) 
	is_zero(m) && return
	pos = Tuple(positions(m))
	L = length(x)
	if pos[end] > L
		resize!(x.physpaces, pos[end])
		for i in L+1:pos[end]
			x.physpaces[i] = missing
		end
	end
	for i in 1:length(pos)
		if ismissing(x.physpaces[pos[i]])
			x.physpaces[pos[i]] = size(op(m)[i], 1) 
		else
			(x.physpaces[pos[i]] == size(op(m)[i], 1)) || throw(DimensionMismatch())
		end
	end
	x_data = x.data
	v = get!(x_data, pos, valtype(x_data)())
	push!(v, (op(m), coeff(m)))
	return x
end  

function qterms(x::QuantumOperator) 
	r = []
	for (k, v) in x.data
		for (m, c) in v
			a = QTerm(k, m, coeff=c)
			if !is_zero(a)
				push!(r, a)
			end
		end
	end
	return r
end

function qterms(x::QuantumOperator, k::Tuple)
	r = []
	v = get(x.data, k, nothing)
	if isnothing(v)
		return r
	else
		for (m, c) in v
			a = QTerm(k, m, coeff=c)		
			if !is_zero(a)
				push!(r, a)
			end
		end
	end
	return r
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

function _absorb_one_bodies(physpaces::Vector, h::Dict)
	r = typeof(h)()
	L = length(physpaces)
	(L >= 2) || throw(ArgumentError("operator should at least have two sites."))
	any(ismissing.(physpaces)) && throw(ArgumentError("hamiltonian has missing spaces."))
	for (key, value) in h
		if length(key)==1
			i = key[1]
			if i < L
				iden = _eye(physpaces[i+1]) 
				m = typeof(value)( [([ item[1], iden], c) for (item, c) in value ] )
				hj = get(r, (i, i+1), nothing)
				if hj == nothing
					r[(i, i+1)] = m
				else
					append!(r[(i, i+1)], m)
				end
			else
				iden = _eye(physpaces[i-1]) 
				m = typeof(value)( [([ iden, item[1]], c) for (item, c) in value ] )
				hj = get(r, (i-1, i), nothing)
				if hj == nothing
					r[(i-1, i)] = m
				else
					append!(r[(i-1, i)], m)
				end
			end
		else
			hj = get(r, key, nothing)
			if hj == nothing
				r[key] = copy(value)
			else
				append!(r[key], value)
			end
		end
	end
	return r
end

function absorb_one_bodies(h::QuantumOperator) 
	r = _absorb_one_bodies(h.physpaces, h.data)
	return QuantumOperator(copy(h.physpaces), r)
end

function _prodham_util(ds::Vector{Int}, opstr::Dict{Int, <:AbstractMatrix}) 
	L = length(ds)
	(max(keys(opstr)...) <= L) || error("op str out of bounds")
	i = 1
	ops = []
	for i in 1:L
	    v = get(opstr, i, nothing)
	    if v === nothing
	        v = _eye(ds[i])
	    else
	    	(size(v, 1)==2 && size(v, 2)==2) || error("dimension mismatch with dim.")
	    end
	    push!(ops, sparse(v))
	end
	return _kron_chain_ops(reverse(ops))
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
	h = spzeros(scalar_type(x), n, n)
	for item in qterms(x)
		h += matrix(ds, item)
	end
	return h
end
