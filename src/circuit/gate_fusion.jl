

_is_commutative(a::Int, b::Int) = (a != b)
function _is_commutative(a::Int, b::NTuple{N, Int}) where N
    for item in b
        _is_commutative(a, item) || return false
    end
    return true
end
_is_commutative(b::NTuple{N, Int}, a::Int) where N = _is_commutative(a, b)

function _is_commutative(a::NTuple{N1, Int}, b::NTuple{N2, Int}) where {N1, N2}
    for item in a
        _is_commutative(item, b) || return false
    end
    return true
end
_is_commutative(a::AbstractQuantumGate, b::AbstractQuantumGate) = _is_commutative(positions(a), positions(b))
function _is_commutative(a::Vector{<:AbstractQuantumGate}, b::AbstractQuantumGate)
    for item in a
        _is_commutative(item, b) || return false
    end
    return true
end
_is_commutative(a::QuantumCircuit, b::AbstractQuantumGate) = _is_commutative(raw_data(a), b)

function commutative_blocks(s::QuantumCircuit)
    t = Vector{typeof(s)}()
    length(s)==0 && return t
    tmp = similar(s)
    for gate in s
        if _is_commutative(tmp, gate)
            push!(tmp, gate)
        else
            push!(t, tmp)
            tmp = similar(s)
            push!(tmp, gate)
        end
    end
    isempty(tmp) && error("something wrong.")
    push!(t, tmp)
    return t
end




# 1 1
function _try_mul_two_ops_impl(ka::Int, ma, kb::Int, mb)
    if ka == kb
        # return contract(ma, mb, ((2,), (1,)))
        @tensor r[-1; -2] := ma[-1, 1] * mb[1, -2]
        return r
    end
    return nothing
end
_try_mul_two_ops_impl(ka::Tuple{Int}, ma, kb::Tuple{Int}, mb) = _try_mul_two_ops_impl(ka[1], ma, kb[1], mb)

# 1, 2
function _try_mul_two_ops_impl(ka::Int, ma, kb::Tuple{Int, Int}, mb)
    i, j = kb
    if ka == i
        # return contract(ma, mb, ((2,), (1,)))
        @tensor r[-1 -2; -3 -4] := ma[-1, 1] * mb[1, -2,-3,-4]
        return r
    elseif ka == j
        # return permute(contract(ma, mb, ((2,), (2,))), (2,1,3,4))
        @tensor r[-1 -2; -3 -4] := ma[-2, 1] * mb[-1, 1,-3,-4]
        return r
    end
    return nothing
end
_try_mul_two_ops_impl(ka::Tuple{Int}, ma, kb::Tuple{Int, Int}, mb) = _try_mul_two_ops_impl(ka[1], ma, kb, mb)

# 1, 3
function _try_mul_two_ops_impl(ka::Int, ma, kb::Tuple{Int, Int, Int}, mb)
    i,j,k = kb
    if ka == i
        # return contract(ma, mb, ((2,), (1,)))
        @tensor r[-1 -2 -3; -4 -5 -6] := ma[-1, 1] * mb[1, -2, -3, -4, -5, -6]
        return r
    elseif ka == j
        # return permute(contract(ma, mb, ((2,), (2,))), (2,1,3,4,5,6))
        @tensor r[-1 -2 -3; -4 -5 -6] := ma[-2, 1] * mb[-1, 1, -3, -4, -5, -6]
        return r
    elseif ka == k
        # return permute(contract(ma, mb, ((2,), (3,))), (2,3,1,4,5,6))
         @tensor r[-1 -2 -3; -4 -5 -6] := ma[-3, 1] * mb[-1, -2, 1, -4, -5, -6]
         return r
    end
    return nothing
end
_try_mul_two_ops_impl(ka::Tuple{Int}, ma, kb::Tuple{Int, Int, Int}, mb) = _try_mul_two_ops_impl(ka[1], ma, kb, mb)

# 1, 4
function _try_mul_two_ops_impl(ka::Int, ma, kb::Tuple{Int, Int, Int, Int}, mb)
    i,j,k,l = kb
    if ka == i
        # return contract(ma, mb, ((2,), (1,)))
        @tensor r[-1 -2 -3 -4; -5 -6 -7 -8] := ma[-1, 1] * mb[1, -2, -3, -4, -5, -6, -7, -8]
        return r
    elseif ka == j
        # return permute(contract(ma, mb, ((2,), (2,))), (2,1,3,4,5,6,7,8))
        @tensor r[-1 -2 -3 -4; -5 -6 -7 -8] := ma[-2, 1] * mb[-1, 1, -3, -4, -5, -6, -7, -8]
        return r
    elseif ka == k
        # return permute(contract(ma, mb, ((2,), (3,))), (2,3,1,4,5,6,7,8))
        @tensor r[-1 -2 -3 -4; -5 -6 -7 -8] := ma[-3, 1] * mb[-1, -2, 1, -4, -5, -6, -7, -8]
        return r
    elseif ka == l
        # return permute(contract(ma, mb, ((2,), (4,))), (2,3,4,1,5,6,7,8))
        @tensor r[-1 -2 -3 -4; -5 -6 -7 -8] := ma[-4, 1] * mb[-1, -2, -3, 1, -5, -6, -7, -8]
        return r
    end
    return nothing
end
_try_mul_two_ops_impl(ka::Tuple{Int}, ma, kb::Tuple{Int, Int, Int, Int}, mb) = _try_mul_two_ops_impl(ka[1], ma, kb, mb)

# 2, 1
function _try_mul_two_ops_impl(ka::Tuple{Int, Int}, ma, kb::Int, mb)
    i, j = ka
    if i == kb
        # return permute(contract(ma, mb, ((3,), (1,))), (1,2,4,3))
        @tensor r[-1 -2; -3 -4] := ma[-1, -2, 1, -4] * mb[1, -3]
        return r
    elseif j == kb
        # return contract(ma, mb, ((4,), (1,)))
        @tensor r[-1 -2; -3 -4] := ma[-1, -2, -3, 1] * mb[1, -4]
        return r
    end
    return nothing
end
_try_mul_two_ops_impl(ka::Tuple{Int, Int}, ma, kb::Tuple{Int}, mb) = _try_mul_two_ops_impl(ka, ma, kb[1], mb)

# 2, 2
function _try_mul_two_ops_impl(ka::Tuple{Int, Int}, ma, kb::Tuple{Int, Int}, mb)
    if ka == kb
        # return contract(ma, mb, ((3,4), (1,2)))
        @tensor r[-1 -2; -3 -4] := ma[-1, -2, 1, 2] * mb[1, 2, -3, -4]
        return r
    end
    return nothing
end

# 2, 3
function _try_mul_two_ops_impl(ka::Tuple{Int, Int}, ma, kb::Tuple{Int, Int, Int}, mb)
    i, j, k = kb
    if ka == (i, j)
        # return contract(ma, mb, ((3,4), (1,2)))
        @tensor r[-1 -2 -3; -4 -5 -6] := ma[-1, -2, 1, 2] * mb[1, 2, -3, -4, -5, -6]
        return r
    elseif ka == (i, k)
        # return permute(contract(ma, mb, ((3,4), (1,3))), (1,3,2,4,5,6))
        @tensor r[-1 -2 -3; -4 -5 -6] := ma[-1, -3, 1, 2] * mb[1, -2, 2, -4, -5, -6]
        return r
    elseif ka == (j, k)
        # return permute(contract(ma, mb, ((3,4), (2,3))), (3,1,2,4,5,6))
        @tensor r[-1 -2 -3; -4 -5 -6] := ma[-2, -3, 1, 2] * mb[-1, 1, 2, -4, -5, -6]
        return r
    end
    return nothing
end

# 2, 4
function _try_mul_two_ops_impl(ka::Tuple{Int, Int}, ma, kb::Tuple{Int, Int, Int, Int}, mb)
    i, j, k, l = kb
    if ka == (i, j)
        # return contract(ma, mb, ((3,4), (1,2)))
        @tensor r[-1 -2 -3 -4; -5 -6 -7 -8] := ma[-1, -2, 1, 2] * mb[1, 2, -3, -4, -5, -6, -7, -8]
        return r
    elseif ka == (i, k)
        # return permute(contract(ma, mb, ((3,4), (1,3))), (1,3,2,4,5,6,7,8))
        @tensor r[-1 -2 -3 -4; -5 -6 -7 -8] := ma[-1, -3, 1, 2] * mb[1, -2, 2, -4, -5, -6, -7, -8]
        return r
    elseif ka == (i, l)
        # return permute(contract(ma, mb, ((3,4), (1,4))), (1,3,4,2,5,6,7,8))
        @tensor r[-1 -2 -3 -4; -5 -6 -7 -8] := ma[-1, -4, 1, 2] * mb[1, -2, -3, 2, -5, -6, -7, -8]
        return r
    elseif ka == (j, k)
        # return permute(contract(ma, mb, ((3,4), (2,3))), (3,1,2,4,5,6,7,8))
        @tensor r[-1 -2 -3 -4; -5 -6 -7 -8] := ma[-2, -3, 1, 2] * mb[-1, 1, 2, -4, -5, -6, -7, -8]
        return r
    elseif ka == (j, l)
        # return permute(contract(ma, mb, ((3,4), (2,4))), (3,1,4,2,5,6,7,8))
        @tensor r[-1 -2 -3 -4; -5 -6 -7 -8] := ma[-2, -4, 1, 2] * mb[-1, 1, -3, 2, -5, -6, -7, -8]
        return r
    elseif ka == (k, l)
        # return permute(contract(ma, mb, ((3,4), (3,4))), (3,4,1,2,5,6,7,8))
        @tensor r[-1 -2 -3 -4; -5 -6 -7 -8] := ma[-3, -4, 1, 2] * mb[-1, -2, 1, 2, -5, -6, -7, -8]
        return r
    end
    return nothing
end

# 3, 1
function _try_mul_two_ops_impl(ka::Tuple{Int, Int, Int}, ma, kb::Int, mb)
    i,j,k = ka
    if i == kb
        # return permute(contract(ma, mb, ((4,), (1,))), (1,2,3,6,4,5))
        @tensor r[-1 -2 -3; -4 -5 -6] := ma[-1, -2, -3, 1, -5, -6] * mb[1, -4]
        return r
    elseif j == kb
        # return permute(contract(ma, mb, ((5,), (1,))), (1,2,3,4,6,5))
        @tensor r[-1 -2 -3; -4 -5 -6] := ma[-1, -2, -3, -4, 1, -6] * mb[1, -5]
        return r
    elseif k == kb
        # return contract(ma, mb, ((6,), (1,)))
        @tensor r[-1 -2 -3; -4 -5 -6] := ma[-1, -2, -3, -4, -5, 1] * mb[1, -6]
        return r
    end
    return nothing
end
_try_mul_two_ops_impl(ka::Tuple{Int, Int, Int}, ma, kb::Tuple{Int}, mb) = _try_mul_two_ops_impl(ka, ma, kb[1], mb)

# 3, 2
function _try_mul_two_ops_impl(ka::Tuple{Int, Int, Int}, ma, kb::Tuple{Int, Int}, mb)
    i, j, k = ka
    if (i, j) == kb
        # return permute(contract(ma, mb, ((4,5), (1,2))), (1,2,3,5,6,4))
        @tensor r[-1 -2 -3; -4 -5 -6] := ma[-1, -2, -3, 1, 2, -6] * mb[1,2,-4,-5]
        return r
    elseif (i, k) == kb
        # return permute(contract(ma, mb, ((4,6), (1,2))), (1,2,3,5,4,6))
        @tensor r[-1 -2 -3; -4 -5 -6] := ma[-1, -2, -3, 1, -5, 2] * mb[1,2,-4,-6]
        return r
    elseif (j, k) == kb
        # return contract(ma, mb, ((5,6), (1,2)))
        @tensor r[-1 -2 -3; -4 -5 -6] := ma[-1, -2, -3, -4, 1, 2] * mb[1,2,-5,-6]
        return r
    end
    return nothing
end

# 3, 3
function _try_mul_two_ops_impl(ka::Tuple{Int, Int, Int}, ma, kb::Tuple{Int, Int, Int}, mb)
    if ka == kb
         # return contract(ma, mb, ((4,5,6), (1,2,3)))
        @tensor r[-1 -2 -3; -4 -5 -6] := ma[-1, -2, -3, 1, 2, 3] * mb[1,2,3,-4,-5,-6]
        return r
    end
    return nothing
end

# 3, 4
function _try_mul_two_ops_impl(ka::Tuple{Int, Int, Int}, ma, kb::Tuple{Int, Int, Int, Int}, mb)
    i, j, k, l = kb
    if ka == (i, j, k)
        # return contract(ma, mb, ((4,5,6), (1,2,3)))
        @tensor r[-1 -2 -3 -4; -5 -6 -7 -8] := ma[-1, -2, -3, 1, 2, 3] * mb[1, 2, 3, -4, -5, -6, -7, -8]
        return r
    elseif ka == (i, j, l)
        # return permute(contract(ma, mb, ((4,5,6), (1,2,4))), (1,2,4,3,5,6,7,8))
        @tensor r[-1 -2 -3 -4; -5 -6 -7 -8] := ma[-1, -2, -4, 1, 2, 3] * mb[1, 2, -3, 3, -5, -6, -7, -8]
        return r
    elseif ka == (i, k, l)
        # return permute(contract(ma, mb, ((4,5,6), (1,3,4))), (1,4,2,3,5,6,7,8))
        @tensor r[-1 -2 -3 -4; -5 -6 -7 -8] := ma[-1, -3, -4, 1, 2, 3] * mb[1, -2, 2, 3, -5, -6, -7, -8]
        return r
    elseif ka == (j, k, l)
        # return permute(contract(ma, mb, ((4,5,6), (2,3,4))), (4,1,2,3,5,6,7,8))
        @tensor r[-1 -2 -3 -4; -5 -6 -7 -8] := ma[-2, -3, -4, 1, 2, 3] * mb[-1, 1, 2, 3, -5, -6, -7, -8]
        return r
    end
    return nothing
end


# 4, 1
function _try_mul_two_ops_impl(ka::Tuple{Int, Int, Int, Int}, ma, kb::Int, mb)
    i,j,k,l = ka
    if i == kb
        # return permute(contract(ma, mb, ((5,), (1,))), (1,2,3,4,8,5,6,7))
        @tensor r[-1 -2 -3 -4; -5 -6 -7 -8] := ma[-1, -2, -3, -4, 1, -6, -7, -8] * mb[1, -5]
        return r
    elseif j == kb
        # return permute(contract(ma, mb, ((6,), (1,))), (1,2,3,4,5,8,6,7))
        @tensor r[-1 -2 -3 -4; -5 -6 -7 -8] := ma[-1, -2, -3, -4, -5, 1, -7, -8] * mb[1, -6]
        return r
    elseif k == kb
        # return permute(contract(ma, mb, ((7,), (1,))), (1,2,3,4,5,6,8,7))
        @tensor r[-1 -2 -3 -4; -5 -6 -7 -8] := ma[-1, -2, -3, -4, -5, -6, 1, -8] * mb[1, -7]
        return r
    elseif l == kb
        # return contract(ma, mb, ((8,), (1,)))
        @tensor r[-1 -2 -3 -4; -5 -6 -7 -8] := ma[-1, -2, -3, -4, -5, -6, -7, 1] * mb[1, -8]
        return r
    end
    return nothing
end
_try_mul_two_ops_impl(ka::Tuple{Int, Int, Int, Int}, ma, kb::Tuple{Int}, mb) = _try_mul_two_ops_impl(ka, ma, kb[1], mb)

# 4, 2
function _try_mul_two_ops_impl(ka::Tuple{Int, Int, Int, Int}, ma, kb::Tuple{Int, Int}, mb)
    i, j, k, l = ka
    if (i, j) == kb
        # return permute(contract(ma, mb, ((5,6), (1,2))), (1,2,3,4,7,8,5,6))
        @tensor r[-1 -2 -3 -4; -5 -6 -7 -8] := ma[-1, -2, -3, -4, 1, 2, -7, -8] * mb[1, 2,-5,-6]
        return r
    elseif (i, k) == kb
        # return permute(contract(ma, mb, ((5,7), (1,2))), (1,2,3,4,7,5,8,6))
        @tensor r[-1 -2 -3 -4; -5 -6 -7 -8] := ma[-1, -2, -3, -4, 1, -6, 2, -8] * mb[1, 2,-5,-7]
        return r
    elseif (i, l) == kb
        # return permute(contract(ma, mb, ((5,8), (1,2))), (1,2,3,4,7,5,6,8))
        @tensor r[-1 -2 -3 -4; -5 -6 -7 -8] := ma[-1, -2, -3, -4, 1, -5, -7, 2] * mb[1, 2,-5,-8]
        return r
    elseif (j, k) == kb
        # return permute(contract(ma, mb, ((6,7), (1,2))), (1,2,3,4,5,7,8,6))
        @tensor r[-1 -2 -3 -4; -5 -6 -7 -8] := ma[-1, -2, -3, -4, -5, 1, 2, -8] * mb[1, 2,-6,-7]
        return r
    elseif (j, l) == kb
        # return permute(contract(ma, mb, ((6,8), (1,2))), (1,2,3,4,5,7,6,8))
        @tensor r[-1 -2 -3 -4; -5 -6 -7 -8] := ma[-1, -2, -3, -4, -5, 1, -7, 2] * mb[1, 2,-6,-8]
        return r
    elseif (k, l) == kb
        # return contract(ma, mb, ((7,8), (1,2)))
        @tensor r[-1 -2 -3 -4; -5 -6 -7 -8] := ma[-1, -2, -3, -4, -5, -6, 1, 2] * mb[1, 2,-7,-8]
        return r
    end
    return nothing
end

# 4, 3
function _try_mul_two_ops_impl(ka::Tuple{Int, Int, Int, Int}, ma, kb::Tuple{Int, Int, Int}, mb)
    i, j, k, l = ka
    if (i, j, k) == kb
        # return permute(contract(ma, mb, ((5,6,7), (1,2,3))), (1,2,3,4,6,7,8,5))
        @tensor r[-1 -2 -3 -4; -5 -6 -7 -8] := ma[-1, -2, -3, -4, 1, 2, 3, -8] * mb[1, 2,3,-5,-6,-7]
        return r
    elseif (i, j, l) == kb
        # return permute(contract(ma, mb, ((5,6,8), (1,2,3))), (1,2,3,4,6,7,5,8))
        @tensor r[-1 -2 -3 -4; -5 -6 -7 -8] := ma[-1, -2, -3, -4, 1, 2, -7, 3] * mb[1, 2,3,-5,-6,-8]
        return r
    elseif (i, k, l) == kb
        # return permute(contract(ma, mb, ((5,7,8), (1,2,3))), (1,2,3,4,6,5,7,8))
        @tensor r[-1 -2 -3 -4; -5 -6 -7 -8] := ma[-1, -2, -3, -4, 1, -6, 2, 3] * mb[1, 2,3,-5,-7,-8]
        return r
    elseif (j, k, l) == kb
        # return contract(ma, mb, ((6,7,8), (1,2,3)))
        @tensor r[-1 -2 -3 -4; -5 -6 -7 -8] := ma[-1, -2, -3, -4, -5, 1, 2, 3] * mb[1, 2,3,-6,-7,-8]
        return r
    end
    return nothing
end

# 4, 4
function _try_mul_two_ops_impl(ka::Tuple{Int, Int, Int, Int}, ma, kb::Tuple{Int, Int, Int, Int}, mb)
    if ka == kb
        # return contract(ma, mb, ((5,6,7,8), (1,2,3,4)))
        @tensor r[-1 -2 -3 -4; -5 -6 -7 -8] := ma[-1, -2, -3, -4, 1, 2, 3, 4] * mb[1, 2,3,4,-5,-6,-7,-8]
        return r
    end
    return nothing
end


function try_mul_two_ops(a::AbstractQuantumGate, b::AbstractQuantumGate, workspace::AbstractVector)
    a_op = op(a)
    b_op = op(b)
    L = length(a_op) + length(b_op)
    if length(workspace) < L
        resize!(workspace, L)
    end
    if !isa(a_op, StridedArray)
        a_op = copyto!(reshape(view(workspace, 1:length(a_op)), size(a_op)), a_op)
    end
    if !isa(b_op, StridedArray)
       b_op = copyto!(reshape(view(workspace, length(a_op)+1:L), size(b_op)), b_op) 
    end
    r = _try_mul_two_ops_impl(positions(a), a_op, positions(b), b_op)
    (r === nothing) && return nothing
    k = (length(positions(a)) >= length(positions(b))) ? positions(a) : positions(b)
    return QuantumGate(k, r)
end

function try_absorb_one_gate_by_mul!(a::Vector{<:AbstractQuantumGate}, gate::AbstractQuantumGate, workspace::AbstractVector)
    for i in length(a):-1:1
        r = try_mul_two_ops(gate, a[i], workspace)
        if r !== nothing
            a[i] = r
            return true
        end
        _is_commutative(a[i], gate) || return false
    end
    return false
end

function try_absorb_by_mul_impl(a::Vector{<:AbstractQuantumGate}, workspace::AbstractVector)
    isempty(a) && return a
    L = length(a)
    b = copy(a)
    for i in L:-1:2
        item = b[i]
        br = b[1:(i-1)]
        if try_absorb_one_gate_by_mul!(br, item, workspace)
            b = [br; b[(i+1):end]]
        end
    end
    return b
end

function _fuse_gate_impl(a::QuantumCircuit)
    workspace = scalar_type(a)[]
    b = try_absorb_by_mul_impl(raw_data(a), workspace)
    r = similar(a)
    append!(r, b)
    return r
end

function fuse_gates(a::QuantumCircuit)
    b = _fuse_gate_impl(a')
    return _fuse_gate_impl(b')
end

