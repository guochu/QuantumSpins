


"""
	updateright(hold::AbstractArray{T, 2}, hAj::MPOTensor, hBj::MPOTensor) where {T <: Number}
update storage from right to left for overlap of mps
"""
function updateright(hold::AbstractArray{T, 2}, hAj::MPOTensor, hBj::MPOTensor) where {T <: Number}
	@tensor m2[-1 -2;-3 -4] := conj(hAj[-1, -2, 1, -4]) * hold[1, -3]
	@tensor hnew[-1;-2] := m2[-1,1,2,3] * hBj[-2,1,2,3]
	return hnew
end



"""
	updateleft(hold::AbstractArray{T, 2}, hAj::MPOTensor, hBj::MPOTensor) where {T <: Number}
update storage from left to right for overlap of mps
"""
function updateleft(hold::AbstractArray{T, 2}, hAj::MPOTensor, hBj::MPOTensor) where {T <: Number}
	@tensor m2[-1 -2 ; -3 -4] := conj(hAj[1, -2, -3, -4]) * hold[1, -1]
	@tensor hnew[-1; -2] := m2[1,2,-1,3] * hBj[1,2,-2,3]
	return hnew
end

function updateright(hold::AbstractArray{T, 3}, psiAj::MPSTensor, hj::MPOTensor, psiBj::MPSTensor) where {T <: Number}
	@tensor hnew[-1; -2 -3] := hold[1,2,3] * psiBj[-3,4,3] * hj[-2,5,2,4] * conj(psiAj[-1,5,1])
	return hnew
end

function updateleft(hold::AbstractArray{T, 3}, psiAj::MPSTensor, hj::MPOTensor, psiBj::MPSTensor) where {T <: Number}
	@tensor hnew[-1; -2 -3] := hold[1,2,3] * psiBj[3,4,-3] * hj[2,5,-2,4] * conj(psiAj[1,5,-1])
	return hnew
end


function updateright(hold::AbstractArray{T, 3}, psiAj::MPSTensor, hj::Nothing, psiBj::MPSTensor) where {T <: Number}
	@tensor hnew[-1; -2 -3] := hold[1,-2,2] * psiBj[-3,3,2] * conj(psiAj[-1,3,1])
	return hnew
end

function updateleft(hold::AbstractArray{T, 3}, psiAj::MPSTensor, hj::Nothing, psiBj::MPSTensor) where {T <: Number}
	@tensor hnew[-1; -2 -3] := hold[1,-2,2] * psiBj[2,3,-3] * conj(psiAj[1,3,-1])
	return hnew
end

function updateright(hold::AbstractMatrix, psiAj::MPSTensor, hj::Nothing, psiBj::MPSTensor) where {T <: Number}
	@tensor hnew[-1; -3] := hold[1,2] * psiBj[-3,3,2] * conj(psiAj[-1,3,1])
	return hnew
end

function updateleft(hold::AbstractMatrix, psiAj::MPSTensor, hj::Nothing, psiBj::MPSTensor) where {T <: Number}
	@tensor hnew[-1; -3] := hold[1,2] * psiBj[2,3,-3] * conj(psiAj[1,3,-1])
	return hnew
end

function updateright(hold::AbstractArray{T, 3}, psiAj::MPSTensor, hj::Number, psiBj::MPSTensor) where {T <: Number}
	@tensor hnew[-1; -2 -3] := hj * hold[1,-2,2] * psiBj[-3,3,2] * conj(psiAj[-1,3,1])
	return hnew
end

function updateleft(hold::AbstractArray{T, 3}, psiAj::MPSTensor, hj::Number, psiBj::MPSTensor) where {T <: Number}
	@tensor hnew[-1; -2 -3] := hj * hold[1,-2,2] * psiBj[2,3,-3] * conj(psiAj[1,3,-1])
	return hnew
end

function updateright(hold::AbstractArray{T, 2}, psiAj::MPSTensor, hj::AbstractMatrix, psiBj::MPSTensor) where {T <: Number}
	@tensor hnew[-1; -3] := hold[1,3] * psiBj[-3,4,3] * hj[5,4] * conj(psiAj[-1,5,1])
	return hnew
end

function updateleft(hold::AbstractArray{T, 2}, psiAj::MPSTensor, hj::AbstractMatrix, psiBj::MPSTensor) where {T <: Number}
	@tensor hnew[-1; -3] := hold[1,3] * psiBj[3,4,-3] * hj[5,4] * conj(psiAj[1,5,-1])
	return hnew
end

function updatetraceleft(hold::AbstractArray{T, 1}, hj::MPOTensor) where {T <: Number}
	@tensor hnew[-1] := hold[1] * hj[1,2,-1,2]
	return hnew
end

function updatetraceleft(hold::AbstractArray{T, 2}, hj::MPOTensor, psij::MPSTensor, fuser::MPSTensor) where {T <: Number}
	@tensor hnew[-1 -2] := hold[1, 2] * hj[1, 3, -1, 4] * psij[2, 4, -2] * fuser[5, 5, 3]
	return hnew
end

function updatetraceleft(hold::AbstractArray{T, 2}, hj::Nothing, psij::MPSTensor, fuser::MPSTensor) where {T <: Number}
	@tensor hnew[-1 -2] := hold[-1, 2] * psij[2, 3, -2] * fuser[5, 5, 3]
	return hnew
end


