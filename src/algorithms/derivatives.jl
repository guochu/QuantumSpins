
function ac_prime(x::MPSTensor, m::MPOTensor, hleft::MPSTensor, hright::MPSTensor) 
	@tensor tmp[-1 -2; -3] := hleft[-1, 1, 2] * x[2,3,4] * m[1,-2,5,3] * hright[-3,5,4]
end


function ac2_prime(x::MPOTensor, h1::MPOTensor, h2::MPOTensor, hleft::MPSTensor, hright::MPSTensor) 
	@tensor tmp[-1 -2; -3 -4] := hleft[-1, 1, 2] * x[2, 3, 4, 5] * h1[1, -2, 6, 3] * h2[6, -3, 7, 4] * hright[-4, 7, 5]
end


function c_prime(x::AbstractMatrix, hleft::MPSTensor, hright::MPSTensor)
    @tensor tmp[-1; -2] := hleft[-1, 1, 2] * x[2, 3] * hright[-2, 1, 3]
end

