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
import itertools


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

#function to define SparsePauliOp 
def define_pauli_op(nr):
    """
    nr: number of register qubits
    """
    #total number of qubits (nq+na) where na is number of ancilla qubits and is always 1
    n = nr + 1
    #get all binary 2^nq pattern
    lst = [list(i) for i in itertools.product([0, 1], repeat=nr)]
    lst = np.array(lst)
    l = []
    for i in lst:
        l.append(list(np.where(i == 1)[0]))

    #list of SparsePauliOp 
    pauli_op = []
    for i in l:
        if i == []:
            pauli_op.append(SparsePauliOp.from_list([('I'*n,1)]))
        else:
            i = [e + 1 for e in i]
            pauli_op.append(SparsePauliOp.from_sparse_list([('Z'*len(i),i,1)],num_qubits=n))
    return pauli_op

#function to generate random QUBO matrix
def generate_random_qubo(nc):
    """
    nc: number of classical bits
    """
    return np.random.uniform(low=-1.0, high=1.0, size = (nc, nc))

#initialize cost function
def init_cost_funcgtion(A, nc):
    """
    A: QUBO matrix A
    nc: the number of classical bits
    """
    #define cost function
    def cost_function(P1, P):
        # first sum of cost function 
        first_sum = 0
        for i in range(nc):
            for j in range(nc):
                if i != j:
                    first_sum += A[i][j]*(P1[i]*P1[j]/P[i]*P[j])
    
        # second sum of cost function 
        second_sum = 0
        for i in range(nc):
            second_sum += A[i][i]*(P1[i]/P[i])
        
        return second_sum + first_sum
    
    return cost_function

#function that classical optimizer will minimize
def func_to_minimize(nc, nr, na, circuit,):
    """
    This function will be minimized by classical optimizer. 
    theta: parameters of the circuit 
    """
    nq = nr + na
    #get expectation values from a circuit
    #get a list of H (observables), which is a list of SparsePauliOp
    H = define_pauli_op(nr)
    H1 = define_pauli_op(nr)
    Ha = SparsePauliOp.from_list([('IIII', 1/2), ('IIIZ', -1/2)])
    for i in range(len(H1)):
        H1[i] = H1[i].sum([H1[i], Ha])  
    
    A = generate_random_qubo(nc)
    cost_function = init_cost_funcgtion(A, nc)

    estimator = Estimator()
    def func(theta):
        #get a expectation value of each H
        job1 = estimator.run([circuit]*len(H),H,[list(theta.values())]*len(H))
        P = job1.result()
        print(f"The primitive-job finished with result {P}")

        job2 = estimator.run([circuit]*len(H),H1,[list(theta.values())]*len(H))
        P1 = job2.result()
        print(f"The primitive-job finished with result {P1}")

        result = cost_function(P1.values, P.values)
        print(f"The resukt of cost function is {result}")
        return result

    return func

#=====================================================
# try functions
"""
Here, I consider the basic smallest case where 
n_c = 8, thus n_r = 3, n_a = 1, and n_q = 4. Also, l = 4
"""
nc = 8 
nr = 3
na = 1
nq = 4
l = 4


parameters, theta = init_parameter(nq, l) 
"""
here, parameters is just a placeholder for the each parameters on the circuit
theta is a dictionary of parameters with random values
"""
circuit= generate_circuit(nr, na, l, parameters)
#draw a circuit 
# print(circuit)
# print(parameters)
circuit.draw(output='mpl', filename='circuit.png')

#try Qiskit primitives
#get expectation values using qiskit primitives
estimator = Estimator()
#define SparsePauliOp 
H = define_pauli_op(nr)
H1 = define_pauli_op(nr)
Ha = SparsePauliOp.from_list([('IIII', 1/2), ('IIIZ', -1/2)])
for i in range(len(H1)):
    H1[i] = H1[i].sum([H1[i], Ha])

estimator = Estimator()
job1 = estimator.run([circuit]*len(H),H,[list(theta.values())]*len(H))
P = job1.result()
print(f"The primitive-job finished with result {P}")

job2 = estimator.run([circuit]*len(H),H1,[list(theta.values())]*len(H))
P1 = job2.result()
print(f"The primitive-job finished with result {P1}")

#init cost function 
A = generate_random_qubo(nc)
cost_function = init_cost_funcgtion(A, nc)
result = cost_function(P1.values, P.values)
print(f"The resukt of cost function is {result}")
