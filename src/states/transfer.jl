

"""
	updateright(hold::AbstractArray{T, 2}, mpsAj::MPSTensor, mpsBj::MPSTensor) where {T<:Number}
update storage from right to left for overlap of mps
"""
function updateright(hold::AbstractArray{T, 2}, mpsAj::MPSTensor, mpsBj::MPSTensor) where {T<:Number}
	@tensor m2[-1 -2;-3] := conj(mpsAj[-1, -2, 1]) * hold[1, -3]
	@tensor hnew[-1;-2] := m2[-1,1,2] * mpsBj[-2,1,2]
	return hnew
end

"""
	updateleft(hold::AbstractArray{T, 2}, mpsAj::MPSTensor, mpsBj::MPSTensor) where {T<:Number}
update storage from left to right for overlap of mps
"""
function updateleft(hold::AbstractArray{T, 2}, mpsAj::MPSTensor, mpsBj::MPSTensor) where {T<:Number}
	@tensor m2[-1 -2; -3] := conj(mpsAj[1, -2, -3]) * hold[1, -1]
	@tensor hnew[-1; -2] := m2[1,2,-1] * mpsBj[1,2,-2]
	return hnew
end