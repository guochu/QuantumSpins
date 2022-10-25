


# function _swap_gate(svectorj1, mpsj1, svectorj2, mpsj2, trunc::TruncationScheme)
# 	# @tensor twositemps[-1 -3; -2 -4] := mpsj1[-1, -2, 1] * mpsj2[1, -3, -4]
# 	a1, b1, c1 = size(mpsj1)
# 	a2, b2, c2 = size(mpsj2)
# 	twositemps = permute(reshape(reshape(mpsj1, a1 * b1, c1) * reshape(mpsj2, a2, b2 * c2), a1, b1, b2, c2), (1,3,2,4))

# 	twositemps1 = reshape(Diagonal(svectorj1) * tie(twositemps, (1, 3)), size(twositemps))
# 	u, s, v, err = tsvd!(twositemps1, (1,2), (3,4), trunc=trunc)

# 	# @tensor u[-1 -2; -3] = twositemps[-1,-2,1,2] * conj(v[-3,1,2])
# 	size_m = length(s)
# 	# we can further save a copy of u by reuse the previous u here.
# 	u = reshape(reshape(twositemps, a1 * b2, b1 * c2) * reshape(v, size_m, b1 * c2)', a1, b2, size_m)
# 	return u, s, v, err
# end

# # multiply the second index of the first matrix with the sencond index of the rank-3 tensor
# function apply_2_2_3_2!(m::StridedMatrix{T}, vi::StridedArray{T, 3}, vo::StridedArray{T, 3}) where {T <: Number}
#     beta = zero(T)
#     alpha = one(T)
#     for i in 1:size(vi, 3)
#         vij = view(vi, :, :, i)
#         voj = view(vo, :, :, i)
#         gemm!('N', 'T', alpha, vij, m, beta, voj)
#     end
#     return vo
# end


# apply_4_34_4_23!(m::StridedArray{T, 4}, vi::StridedArray{T, 4}, vo::StridedArray{T, 4}) where T = apply_2_2_3_2!(tie(m, (2,2)), tie(vi, (1,2,1)), tie(vo, (1,2,1)))
# apply_4_34_4_23!(m::StridedArray{T1, 4}, vi::StridedArray{T, 4}, vo::StridedArray{T, 4}) where {T1, T} = apply_4_34_4_23!(convert(Array{T, 4}, m), vi, vo)

# function bond_evolution(bondmpo, svectorj1, mpsj1, svectorj2, mpsj2, trunc::TruncationScheme)
# 	# @tensor twositemps[-1 -2; -3 -4] :=  mpsj1[-1, 1, 3] * mpsj2[3, 2, -4] * bondmpo[-2,-3, 1,2]
# 	a1, b1, c1 = size(mpsj1)
# 	a2, b2, c2 = size(mpsj2)
# 	twositemps = reshape(reshape(mpsj1, a1 * b1, c1) * reshape(mpsj2, a2, b2 * c2), a1, b1, b2, c2)
# 	twositemps1 = similar(twositemps)
# 	apply_4_34_4_23!(bondmpo, twositemps, twositemps1)
# 	copyto!(twositemps, twositemps1)

# 	# we can save a copy of twositemps1 by reuse the previous memory of twositemps1 here.
# 	twositemps1 = reshape(Diagonal(svectorj1) * tie(twositemps, (1, 3)), size(twositemps))
# 	# to remove very small numbers
# 	# u, s, v, err = tsvd!(twositemps1, (1,2), (3,4), trunc=trunc)
# 	u, s, v, err = tsvd!(tie(twositemps1,(2,2)), trunc=trunc)
# 	u = reshape(u, size(twositemps1, 1), size(twositemps1, 2), length(s))
# 	v = reshape(v, length(s), size(twositemps1, 3), size(twositemps1, 4))

# 	# @tensor u[-1 -2; -3] = twositemps[-1,-2,1,2] * conj(v[-3,1,2])
# 	size_m = length(s)
# 	# we can further save a copy of u by reuse the previous u here.
# 	u = reshape(reshape(twositemps, a1 * b2, b1 * c2) * reshape(v, size_m, b1 * c2)', a1, b2, size_m)
# 	return u, s, v, err	
# end