
function ac_prime(x::MPSTensor, m::MPOTensor, hleft::MPSTensor, hright::MPSTensor) 
	@tensor tmp[-1 -2; -3] := ((hleft[-1, 1, 2] * x[2,3,4]) * m[1,-2,5,3]) * hright[-3,5,4]
end

function c_prime(x::AbstractMatrix, hleft::MPSTensor, hright::MPSTensor)
    @tensor tmp[-1; -2] := (hleft[-1, 1, 2] * x[2, 3]) * hright[-2, 1, 3]
end

struct CentralHeff{T}
	mpojhleft::Array{T, 5}
	hright::Array{T, 3}
end

function Heff(mpoj::MPOTensor, hleft::MPSTensor, hright::MPSTensor)
	@tensor tmp[1,4,5,3,6] := hleft[1,2,3] * mpoj[2,4,5,6]
	return CentralHeff(tmp, hright)
end

function ac_prime(x::MPSTensor, heff::CentralHeff)
	@tensor tmp[1,2,7] := (heff.mpojhleft[1,2,3,4,5] * x[4,5,6]) * heff.hright[7,3,6]
end

# # the convention of mpotensor is assumed to be permute(m, (2,3,1,4))
# function ac_prime_fast(x::MPSTensor, m::MPOTensor, hleft::MPSTensor, hright::MPSTensor, workspace::AbstractVector) 
# 	# @tensor tmp[-1 -2; -3] := ((hleft[-1, 1, 2] * x[2,3,4]) * m[1,-2,5,3]) * hright[-3,5,4]
# 	L = size(hleft, 1) * size(hleft, 2) * size(x, 2) * size(x, 3)
# 	L2 = size(hleft, 1) * size(m, 1) * size(m, 2) * size(x, 3)
# 	if length(workspace) < L + L2
# 		resize!(workspace, L + L2)
# 	end
# 	tmp = reshape(view(workspace, 1:L), (size(hleft, 1), size(hleft, 2), size(x, 2), size(x, 3)) )
# 	tmp2 = reshape(view(workspace, L+1:L+L2), (size(hleft, 1), size(m, 1), size(m, 2), size(x, 3)) )
# 	# mul!(tie(tmp, (2, 2)), tie(hleft, (2, 1)), tie(x, (1, 2)))
# 	gemm!('N', 'N', true, tie(hleft, (2, 1)), tie(x, (1, 2)), false, tie(tmp, (2, 2)))
# 	apply_physical!(m, tmp, tmp2)
# 	r = gemm('N', 'T', tie(tmp2, (2,2)), tie(hright, (1, 2)))
# 	return reshape(r, size(hleft, 1), size(m, 1), size(hright, 1) )
# end

# function ac2_prime(x::MPOTensor, h1::MPOTensor, h2::MPOTensor, hleft::MPSTensor, hright::MPSTensor) 
# 	@tensor tmp[-1 -2; -3 -4] := hleft[-1, 1, 2] * x[2, 3, 4, 5] * h1[1, -2, 6, 3] * h2[6, -3, 7, 4] * hright[-4, 7, 5]
# end




# function c_prime_fast(x::AbstractMatrix, hleft::MPSTensor, hright::MPSTensor, workspace::AbstractVector)
# 	L = size(hleft, 1) * size(hleft, 2) * size(x, 2)
# 	if length(workspace) < L
# 		resize!(workspace, L)
# 	end
# 	tmp = reshape(view(workspace, 1:L), size(hleft, 1), size(hleft, 2), size(x, 2) )
# 	gemm!('N', 'N', true, tie(hleft, (2, 1)), x, false, tie(tmp, (2, 1)))
# 	return gemm('N', 'T', tie(tmp, (1, 2)), tie(hright, (1, 2)) )
# end

