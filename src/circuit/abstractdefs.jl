

abstract type AbstractQuantumGate{N} end
const AbstractOneBodyGate = AbstractQuantumGate{1}
const AbstractTwoBodyGate = AbstractQuantumGate{2}
const AbstractThreeBodyGate = AbstractQuantumGate{3}

abstract type AbstractQuantumCircuit end
