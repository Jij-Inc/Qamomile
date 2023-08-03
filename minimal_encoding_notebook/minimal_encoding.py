from tkinter import N
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

import dimod

def main():
    # try minimal encoding
    # set basic information
    nc = 4
    nr = math.log2(nc) #in current implementation, this has to retun integer
    na = 1
    nq = int(nr + na)
    l = 4

    if nr.is_integer() == False:
        print("The number of register qubits should be integer")
        return 0
    else:
        nr = int(nr)

    parameters, theta = init_parameter(nq, l) 
    # here, parameters is just a placeholder for the each parameters on the circuit
    # theta is a dictionary of parameters with random values

    circuit= generate_circuit(nr, na, l, parameters)
    #draw a circuit, comment out appropriate commands below if you like to see the circuit
    # print(circuit)
    # print(parameters)
    # circuit.draw(output='mpl', filename='circuit.png')

    #try classical optimizer
    progress_history = []
    A = generate_random_qubo(nc)
    if check_symmetric(A) == False:
        print("The QUBO matrix is not symmetric")
        return 0
    print(A)
    func = init_func(nc, nr, na, circuit, A, progress_history)
    #number of evaluation of cost function, it can be changed but
    n_eval = 300 
    optimizer = COBYLA(maxiter=n_eval, disp=True)
    result = optimizer.minimize(func, list(theta.values()))
    print(f"The total number of function evaluations => {result.nfev}")
    decoded_result = get_ancilla_prob(result.x, circuit, nr)

    plt.plot(progress_history)
    plt.xlabel('number of iteration')
    plt.ylabel('value of cost function')
    plt.show()

#initialize parameters
def init_parameter(nq:int , l:int):
    '''
    Function to initialize parameters for the circuit with random theta values
    Parameters
    ----------
    nq : int
        number of qubits
    l : int
        number of layers
    Returns
    -------
    parameters : numpy array 
        placeholder for parameters
    theta : dictionary
        dictionary of parameters with random values
    '''
    #initialize a parameters
    parameters = ParameterVector('θ', nq*l)
    #create a dictionary of parameters with random values and return it
    theta =  {parameter: np.random.random() for parameter in parameters}
    return parameters, theta

#generate a circuit
def generate_circuit(nr:int, na:int, l:int, parameters:np.array):
    """
    Function to generate qunatum circuit (variational ansatz) for minimal encoding.

    Parameters
    ----------
    nr : int 
        number of register qubit
    na : int
        number of ancilla qubits (which is 1 for this specific encoding)
    l : int
        number of layer, for this specific circuit one layer consists of C-NOT and Ry rotation gate
    parameters : numpy array
        parameters placeholder for the circuit

    Returns
    -------
    circuit : qiskit.circuit.quantumcircuit.QuantumCircuit
        Parameterised quantum circuit
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
        # circuit.cx(qreg_q[0],areg_a[0])
        circuit.cx(areg_a[0],qreg_q[0])
        for i in range(nr):
            if i != 0:
                # circuit.cx(qreg_q[i],qreg_q[i-1]) 
                circuit.cx(qreg_q[i-1],qreg_q[i]) 

        #Ry
        for i in range(nq):
            if i == 0:
                circuit.ry(parameters[nq*j+i], areg_a[i])
            else:
                circuit.ry(parameters[nq*j+i], qreg_q[i-1])  
        circuit.barrier()
    return circuit

#function to define SparsePauliOp 
def define_pauli_op(nr:int, ancilla:bool=False):
    """
    Function to define pauli operator for each possible outcomes in computational basis
    
    Parameters
    ----------
    nr : int
        number of register qubits
    ancilla : bool, optional
        whether to add ancilla qubit |1> or not. The default is False.

    Returns
    -------
    pauli_op : list[SparsePauliOp]
        list of SparsePauliOp for each possible outcomes in computational basis 
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
def generate_random_qubo(nc:int):
    """
    Function to generate random QUBO matrix

    Parameters
    ----------
    nc : int
        number of classical bits
    
    Returns
    -------
    Q : numpy ndarray
        random QUBO matrix
    """
    Q = np.random.uniform(low = -1.0, high = 1.0, size = (nc, nc))
    Q = (Q + Q.T)/2
    return Q

#initialize cost function
def init_cost_function(A:np.ndarray, nc:int):
    """
    Function to initialize cost function for minimal encoding

    Parameters
    ----------
    A : numpy ndarray
        QUBO matrix
    nc : int
        number of classical bits
    
    Returns
    -------
    cost_function : function
        cost function for minimal encoding
    """
    #define cost function
    def cost_function(P1:np.array, P:np.array):
        '''
        Function that compute value of cost function for minimal encoding

        Parameters
        ----------
        P1 : numpy array
            expectation value of each pauli operator for when ancilla qubit is |1>
        P : numpy array
            expectation value of each pauli operator ignoring ancilla qubit

        Returns
        -------
        result : float
            value of cost function
        '''
        # first sum of cost function 
        first_sum = 0
        for i in range(nc):
            for j in range(nc):
                if i != j:
                    # first_sum += A[i][j]*(P1[i]*P1[j]/P[i]*P[j])
                    first_sum += A[i][j]*(P1[i]/P[i])*(P1[j]/P[j])
                # else:
                #     first_sum += A[i][i]*(P1[i]/P[i])
        # second sum of cost function 
        second_sum = 0
        for i in range(nc):
            second_sum += A[i][i]*(P1[i]/P[i])

        result = first_sum + second_sum
        return  result
    
    return cost_function

#function that classical optimizer will minimize
def init_func(nc:int, nr:int, na:int, circuit:qiskit.circuit.quantumcircuit.QuantumCircuit , A:np.ndarray, progress_history:list):
    """
    Function to initialize the function to be minimised by classical optimisor

    Parameters
    ----------
    nc : int
        number of classical bits
    nr : int
        number of register qubits
    na : int 
        number of ancilla qunbits 
    circuit : qiskit.circuit.quantumcircuit.QuantumCircuit
        variational ansatz (parameterized quantum circuit)
    A : numpy ndarray
        QUBO matrix
    progress_history : list
        list to store the progress of the minimization

    Returns
    -------
    func : function
        function to be minimized by classical optimizer
    """
    nq = nr + na
    #get expectation values from a circuit
    #get a list of H (observables), which is a list of SparsePauliOp
    H = define_pauli_op(nr)
    Ha = define_pauli_op(nr, ancilla=True)
    
    cost_function = init_cost_function(A, nc)

    estimator = Estimator()
    # estimator.set_options(shots=100000)
    def func(theta:dict):
        """
        The function to be minimized by classical optimizer. 

        Parameters
        ----------
        theta: dict
            parameters of the circuit 
        
        Returns
        -------
        result : float
            value of the cost function

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
        # print(f"The result of cost function is {result}")
        return result

    return func

def check_symmetric(A:np.ndarray, rtol=1e-05, atol=1e-08):
    '''
    Function to check if a matrix is symmetric or not

    Parameters
    ----------
    A : numpy array
        matrix to be checked
    rtol : float, optional
        relative tolerance. The default is 1e-05.
    atol : float, optional
        absolute tolerance. The default is 1e-08.
    
    Returns
    -------
    bool
        True if matrix is symetric, False otherwise
    '''
    return np.allclose(A, A.T, rtol=rtol, atol=atol)

# not complete
def get_ancilla_prob(theta:dict, circuit:qiskit.circuit.quantumcircuit.QuantumCircuit, nr:int):
    estimator = Estimator()
    #define observable to calculate expectation value
    H = define_pauli_op(nr, ancilla=False)
    Ha = define_pauli_op(nr,ancilla=True)
    #get expectation values from
    job1 = estimator.run([circuit]*len(H),H,[theta]*len(H))
    P = job1.result()
    # print(f"The primitive-job finished with result {P}")
    job2 = estimator.run([circuit]*len(H),Ha,[theta]*len(H))
    P1 = job2.result()
    # print(f"The primitive-job finished with result {P1}")
    #compute b_i and a_i
    b=[]
    a=[]
    for i in range(len(P.values)):
        val = (P1.values[i] / P.values[i])
        b.append(val)
        a.append(1 - b[i])

    final_binary = []
    #compare a and b and pick the larger one
    for i in range(len(a)):
        if a[i] > b[i]:
            final_binary.append(0)
        else:
            final_binary.append(1)

    # print(f"b is Pr(x=1){b}")
    # print(f"a is Pr(x=0){a}")
    #need to fix return value
    # print(f"final binary is {final_binary}")
    final_binary = np.array(final_binary)
    return final_binary

# def get_sample(final_binary:np.array, energy:np.array):
#     sample = dimod.SampleSet.from_samples(
#         dimod.as_samples(final_binary), 
#         'BINARY', 
#         energy=energy,
#     )
#     return sample


if __name__ == '__main__':
    main()
