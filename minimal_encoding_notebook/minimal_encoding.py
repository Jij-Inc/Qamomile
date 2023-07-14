from qiskit import QuantumRegister, AncillaRegister ,ClassicalRegister, QuantumCircuit, Aer, execute
from qiskit.circuit import Parameter, ParameterVector
from qiskit.visualization import plot_state_city, plot_histogram

#libs for get expectation value
from qiskit.primitives import Estimator
from qiskit.circuit.library import RealAmplitudes
from qiskit.quantum_info import SparsePauliOp

import numpy as np
from numpy import pi
import math


#initialize parameters
def init_parameter(nq, l):
    #initialize a parameters
    parameters = ParameterVector('θ', nq*l)
    #create a dictionary of parameters with random values and return it
    return parameters, {parameter: np.random.random() for parameter in parameters}

#generate a circuit
def generate_circuit(nr, na, l,parameters):
    """
    Function to generate qunatum circuit (variational ansatz) for minimal encoding.
    nr: number for qubits 
    na: number of ancilla qubits (which is 1 for this specific encoding)
    l: number of layer, for this specific circuit one layer consists of C-NOT and Ry rotation gate
    parameters: parameters placeholder for the circuit
    """
    #define number of qubits
    nq = nr + na
    qreg_q = QuantumRegister(nr, 'q')
    areg_a = AncillaRegister(na, 'a')
    creg_c = ClassicalRegister(nq, 'c')
    circuit = QuantumCircuit(areg_a, qreg_q, creg_c)

    #add H gate for each qubit
    circuit.h(areg_a[0])
    for i in range(0,nr):
        circuit.h(qreg_q[i])
    circuit.barrier()
  
    #add layers which consist of CNOT and Ry gate
    for j in range(0,l):
        #CNOT
        circuit.cx(qreg_q[0],areg_a[0])
        for i in range(nr):
            if i != 0:
                circuit.cx(qreg_q[i],qreg_q[i-1]) 
        #Ry
        for i in range(nq):
            if i == 0:
                circuit.ry(parameters[nq*j+i], areg_a[i])
            else:
                circuit.ry(parameters[nq*j+i], qreg_q[i-1])  
        circuit.barrier()

    return circuit

def generate_random_qubo():
    return 0

#initialize cost function
def init_cost_funcgtion(A, n):
    """
    A: QUBO matrix A
    n: n_c, the number of classical bits
    """
    #define cost function
    def cost_function(P1, P):
        # first sum of cost function 
        first_sum = 0
        for i in range(n):
            for j in range(n):
                if i != j:
                    first_sum += A[i][j]*(P1[i]*P1[j]/P[i]*P[j])
    
        # second sum of cost function 
        second_sum = 0
        for i in range(n):
            second_sum += A[i][i]*(P1[i]/P[i])
        
        return second_sum + first_sum
    
    return cost_function

# try functions
parameters, theta = init_parameter(5,2)
circuit= generate_circuit(4,1,2, parameters)
#draw a circuit 
# print(circuit)
# print(parameters)
circuit.draw(output='mpl', filename='circuit.png')

#try Qiskit primitives
#get expectation values using qiskit primitives
estimator = Estimator()
#define SparsePauliOp 
#measure all register qubit with computational basis (Z baseis)({0,1})
H1 = SparsePauliOp.from_list([('IIIII',1),('IIIZI',1), ('IIZII',1),('IZIII',1),('ZIIII',1)])

job = estimator.run([circuit],[H1],[list(theta.values())])
job_result = job.result()
print(f"The primitive-job finished with result {job_result}")
