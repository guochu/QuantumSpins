

ZERO = [1., 0.]

ONE = [0., 1.]

PHASE(phi::Real) = [1. 0. ; 0. exp(im*phi)]

const X = [0. 1. ; 1. 0.]

const Y = [0. -im; im 0.]

const Z = [1. 0. ; 0. -1.]

const S = [1. 0. ;  0. im]

const H = (X + Z) / sqrt(2)

const UP = [1. 0. ; 0. 0.]

const DOWN = [0. 0.; 0. 1.]

const Xh = [1+im 1-im; 1-im 1+im]/2

const sqrtX = Xh

const Yh = [im -im; im im]/sqrt(2*im)

const sqrtY = Yh

const T = [1. 0.; 0. exp(im*pi/4)]

R(k::Int) = [1 0; 0 exp(pi*im/(2^(k-1)))]

_Rx(theta::Number) = [cos(theta) -im*sin(theta); -im*sin(theta) cos(theta)]
Rx(theta::Number) = _Rx(theta/2)

_Ry(theta::Number) = [cos(theta) -sin(theta); sin(theta) cos(theta)]
Ry(theta::Number) = _Ry(theta/2)

_Rz(theta::Number) = [exp(-im*theta) 0; 0 exp(im*theta)]
Rz(theta::Number) = _Rz(theta/2)

function CONTROL(u::AbstractMatrix)
	(size(u, 1) == size(u, 2)) || error("must be a square matrix.")
	(size(u, 1) == 2) || error("input matrix must be 2 by 2")
	return rkron(UP, _eye(2)) + rkron(DOWN, u)
end

const CZ = CONTROL(Z)

const CNOT = CONTROL(X)

const CX = CNOT

const SWAP = [1. 0. 0. 0.; 0. 0. 1. 0.; 0. 1. 0. 0.; 0. 0. 0. 1.]

const iSWAP = [1. 0. 0. 0.; 0. 0. im 0.; 0. im 0. 0.; 0. 0. 0. 1.]

# three body gates
function CONTROLCONTROL(u::AbstractMatrix)
	(size(u, 1) == size(u, 2)) || error("must be a square matrix.")
	(size(u, 1) == 2) || error("input matrix must be 2 by 2")
	Iu = _eye(2)
	return rkron(rkron(UP, UP), Iu) + rkron(rkron(UP, DOWN), Iu) +
	rkron(rkron(DOWN, UP), Iu) + rkron(rkron(DOWN, DOWN), u)
end

const TOFFOLI = CONTROLCONTROL(X)

const CCX = TOFFOLI

const FREDKIN = rkron(UP, _eye(4)) + rkron(DOWN, SWAP)

const CSWAP = FREDKIN

"""
	From custom matrix in produced by kron
"""
two_qubits_mat_from_external(m::AbstractMatrix) = reshape(permute(reshape(m, (2,2,2,2)), (2,1,4,3)), 4, 4)

"""
	From custom matrix in produced by kron
"""
three_qubits_mat_from_external(m::AbstractMatrix) = reshape(permute(reshape(m, (2,2,2,2,2,2)), (3,2,1,6,5,4) ), 8, 8)


function FSIM(theta::Real, phi::Real)
	m = [1 0 0 0; 0 cos(theta) -im*sin(theta) 0; 0 -im*sin(theta) cos(theta) 0; 0 0 0 exp(-im*phi)]
	return reshape(permute(reshape(m, (2,2,2,2)), (2,1,4,3)), 4, 4)
end

function GFSIM(theta::Real, phi::Real, deltap::Real, deltam::Real, deltamoff::Real)
	m = [[1 0 0 0]; [0 exp(im*(deltap+deltam))*cos(theta) -im*exp(im*(deltap-deltamoff))*sin(theta) 0];
    [0 -im*exp(im*(deltap+deltamoff))*sin(theta) exp(im*(deltap-deltam))*cos(theta) 0]; [0 0 0 exp(im*(2*deltap-phi))]]
    return two_qubits_mat_from_external(m)	
end

FSIM(theta::Real, phi::Real, deltap::Real, deltam::Real, deltamoff::Real) = GFSIM(theta, phi, deltap, deltam, deltamoff)
