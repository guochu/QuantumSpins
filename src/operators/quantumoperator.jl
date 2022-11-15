const QOP_DATA_TYPE{T} = Dict{Tuple{Int, Vararg{Int, N} where N}, Vector{Tuple{Vector{MPOTensor{T}}, AbstractCoefficient}}}

struct QuantumOperator{T<:Number} <: AbstractQuantumOperator
	physpaces::Vector{Union{Int, Missing}}
	data::QOP_DATA_TYPE{T}
end

QuantumOperator{T}(ds::Vector{Union{Int, Missing}}) where {T <: Number} = QuantumOperator{T}(ds, QOP_DATA_TYPE{T}() )
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

scalar_type(::Type{QuantumOperator{T}}) where T = T

Base.copy(x::QuantumOperator) = QuantumOperator(copy(x.physpaces), deepcopy(x.data))
Base.similar(s::QuantumOperator{T}) where T = QuantumOperator{T}()

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


function Base.:*(x::QuantumOperator, y::AllowedCoefficient) 
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

Base.adjoint(x::QuantumOperator) = QuantumOperator(x.physpaces, data_type(x)(k=>[(adjoint.(a), conj(b)) for (a, b) in v] for (k, v) in x.data))

function Base.:+(x::QuantumOperator, y::QuantumOperator)
	T = promote_type(scalar_type(x), scalar_type(y)) 
	z = QuantumOperator{T}(_merge_spaces(x.physpaces, y.physpaces), convert(QOP_DATA_TYPE{T}, copy(x.data)))
	for (k, v) in y.data
	   	tmp = get!(z.data, k, Vector{Tuple{Vector{MPOTensor{T}}, AbstractCoefficient}}())
	   	append!(tmp, convert(Vector{Tuple{Vector{MPOTensor{T}}, AbstractCoefficient}}, v))
	end	
	return z
end
Base.:-(x::QuantumOperator, y::QuantumOperator) = x + (-y)

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
			x.physpaces[pos[i]] = size(op(m)[i], 2) 
		else
			(x.physpaces[pos[i]] == size(op(m)[i], 2) == size(op(m)[i], 4) ) || throw(DimensionMismatch())
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


function _absorb_one_bodies(physpaces::Vector, h::Dict)
	r = typeof(h)()
	L = length(physpaces)
	(L >= 2) || throw(ArgumentError("operator should at least have two sites."))
	any(ismissing.(physpaces)) && throw(ArgumentError("hamiltonian has missing spaces."))
	for (key, value) in h
		if length(key)==1
			i = key[1]
			if i < L
				iden = _id4(1, physpaces[i+1]) 
				m = typeof(value)( [([ item[1], iden], c) for (item, c) in value ] )
				hj = get(r, (i, i+1), nothing)
				if hj == nothing
					r[(i, i+1)] = m
				else
					append!(r[(i, i+1)], m)
				end
			else
				iden = _id4(1, physpaces[i-1]) 
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



