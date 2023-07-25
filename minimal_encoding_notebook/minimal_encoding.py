from qiskit import QuantumRegister, AncillaRegister ,ClassicalRegister, QuantumCircuit, Aer, execute
from qiskit.circuit import Parameter, ParameterVector
from qiskit.visualization import plot_state_city, plot_histogram
from qiskit.algorithms.optimizers import COBYLA

#libs for get expectation value
from qiskit.primitives import Estimator, Sampler
from qiskit.circuit.library import RealAmplitudes
from qiskit.quantum_info import SparsePauliOp

import numpy as np
from numpy import pi
import math
import itertools
import matplotlib.pyplot as plt


def main():
    # try functions
    """
    Here, I consider the basic smallest case from paper where 
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
    #draw a circuit, comment out appropriate commands below if you like to see the circuit
    # print(circuit)
    # print(parameters)
    # circuit.draw(output='mpl', filename='circuit.png')

    #try classical optimizer
    progress_history = []
    func = init_func(nc, nr, na, circuit, progress_history)
    #number of evaluation of cost function, it can be changed but
    #it should be more than 150 to observe a good result
    n_eval = 300 
    optimizer = COBYLA(maxiter=n_eval, disp=True)
    result = optimizer.minimize(func, list(theta.values()))
    print(f"The total number of function evaluations => {result.nfev}")

    plt.plot(progress_history)
    plt.xlabel('number of iteration')
    plt.ylabel('value of cost function')
    plt.show()


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
    circuit = QuantumCircuit(areg_a, qreg_q)

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
def define_pauli_op(nr, ancilla:bool=False):
    """
    nr: number of register qubits
    ancilla: if True, add ancilla qubit of |1> state, else add pauli I for ancilla qubit
    """
    #total number of qubits (nq+na) where na is number of ancilla qubits and is always 1
    #get all binary 2^nq pattern
    l = [list(i) for i in itertools.product([0, 1], repeat=nr)]
    l = np.array(l)

    #basic pieces of SparsePauliOp 
    #PauliOp for |0> 
    P0 = SparsePauliOp.from_list([('I', 1/2), ('Z', 1/2)])
    #PauliOp for |1> 
    P1 = SparsePauliOp.from_list([('I', 1/2), ('Z', -1/2)])
    #Indentiy op
    Id = SparsePauliOp.from_list([('I', 1)])

    #init list of SparsePauliOp 
    pauli_op = []
    for i in range(len(l)):
        pauli_op.append(SparsePauliOp.from_list([('',1)]))
        for j in l[i]:
            if j == 0:
                pauli_op[i] = pauli_op[i].tensor(P0)
            else: 
                pauli_op[i] = pauli_op[i].tensor(P1)

    #add ancilla qubit
    for i in range(len(pauli_op)):
        if ancilla == True:
            pauli_op[i] = pauli_op[i].tensor(P1)
        else:
            pauli_op[i] = pauli_op[i].tensor(Id)
    
    return pauli_op

#function to generate random QUBO matrix
def generate_random_qubo(nc):
    """
    nc: number of classical bits
    """
    q = np.random.uniform(low = -1.0, high = 1.0, size = (nc, nc))
    return (q + q.T)/2

#initialize cost function
def init_cost_function(A, nc):
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
def init_func(nc, nr, na, circuit,progress_history):
    """
    This function will be minimized by classical optimizer. 
    theta: parameters of the circuit 
    nc: number of classical bits
    nr, na: number of register and ancilla qubits
    circuit: quantum circuit
    progress_history: list to store the progress of the minimization
    """
    nq = nr + na
    #get expectation values from a circuit
    #get a list of H (observables), which is a list of SparsePauliOp
    H = define_pauli_op(nr)
    Ha = define_pauli_op(nr, ancilla=True)
    
    A = generate_random_qubo(nc)
    cost_function = init_cost_function(A, nc)

    estimator = Estimator()
    def func(theta):
        """
        This function will be minimized by classical optimizer. 
        theta: parameters of the circuit 
        """
        #get a expectation value of each H
        job1 = estimator.run([circuit]*len(H),H,[theta]*len(H))
        P = job1.result()
        # print(f"The primitive-job finished with result {P}")

        job2 = estimator.run([circuit]*len(H),Ha,[theta]*len(H))
        P1 = job2.result()
        # print(f"The primitive-job finished with result {P1}")

        result = cost_function(P1.values, P.values)
        progress_history.append(result)
        print(f"The result of cost function is {result}")
        return result

    return func

# not complete
def decode(theta, circuit,):
    circuit.measure_all(inplace=True)
    sampler = Sampler()
    job = sampler.run(circuits=[circuit], parameter_values=[result.x], parameters=[[]])
    job_result = job.result()
    result = [q.binary_probabilities() for q in job_result.quasi_dists]
    beta = list(resultr[0].values())
    print(result)
    circuit.remove_final_measurements(inplace=True)
    #define observable to calculate expectation value
    H = define_pauli_op(nr, ancilla=False)
    Ha = define_pauli_op(nr,ancilla=True)
    #get expectation values from
    job1 = estimator.run([circuit]*len(H),H,[result.x]*len(H))
    P = job1.result()
    # print(f"The primitive-job finished with result {P}")
    job2 = estimator.run([circuit]*len(H),Ha,[result.x]*len(H))
    P1 = job2.result()
    # print(f"The primitive-job finished with result {P1}")

    #compute b_i
    b = []
    a= []
    for i in range(1, len(P.values)):
        b.append(P1.values[i] / P.values[i])
        a[i] = 1 - b[i]

    #need to fix return value
    return 0

if __name__ == '__main__':
    main()
