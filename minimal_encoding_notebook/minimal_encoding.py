import qiskit as qk
from qiskit import QuantumRegister, AncillaRegister , QuantumCircuit
from qiskit.circuit import  ParameterVector
from qiskit_ibm_runtime import QiskitRuntimeService, Session
from qiskit.quantum_info import SparsePauliOp
from qiskit.algorithms.optimizers import COBYLA

import numpy as np
import math
import itertools
import matplotlib.pyplot as plt
import jijmodeling as jm
import dimod
import time
from scipy.optimize import minimize

# from qiskit_aer.primitives import Estimator
# from qiskit_ibm_runtime import Estimator
from qiskit.primitives import Estimator

def main():
    # try minimal encoding
    # set basic information
    nc = 8
    nr = math.log2(nc) #in current implementation, this has to retun integer
    na = 1
    nq = int(nr + na)
    # l = 4
    l = [2, 4, 8, 12, 16, 20]  # uncomment when smapling different number of layers

    if nr.is_integer() == False:
        print("The number of register qubits should be integer")
        return 0
    else: 
        nr = int(nr)


    #generate random QUBO matrix
    A = generate_random_qubo(nc)
    if check_symmetric(A) == False:
        print("The QUBO matrix is not symmetric")
        return 0
    # print(A)

    line_style = ["-", "--", "-.", ":", "-", "--"] 
    # shots = [None,100, 1000, 5000, 10000, 100000] # uncomment when sampling different number of shots
    # shots = None
    shots = 100000
    # For loop to smaple different number of layers
 
    # circuit.draw(output='mpl', filename='circuit.png')
    for l, style in zip(l, line_style):
        # here, parameters is just a placeholder for the each parameters on the circuit
        # theta is a dictionary of parameters with random values
        parameters, theta = init_parameter(nq, l) 
        circuit= generate_circuit(nr, na, l, parameters)
        
        #draw a circuit, comment out appropriate commands below if you like to see the circuit
        # print(circuit)
        # print(parameters)
        
        #try classical optimizer
        progress_history = []
        #number of evaluation of cost function, it can be changed but
        n_eval = 1000
        estimator = Estimator() 
        estimator.set_options(shots=shots)
        cost_function = init_cost_function(A, nc)
        H = define_pauli_op(nr)
        Ha = define_pauli_op(nr, ancilla=True)
        optimizer = COBYLA(maxiter=n_eval,args=(circuit, estimator, cost_function, H, Ha, progress_history),)
        start = time.time()
        result = optimizer.minimize(fun, theta,)
        # result = minimize(fun, theta, args=(circuit, estimator, cost_function, H, Ha, progress_history), 
        #                   method='cobyla', tol = 0.00001, options={'maxiter': n_eval,})
        end = time.time()
        t = end - start
        print(f"The total number of function evaluations => {result.nfev}")
        decoded_result = get_ancilla_prob(result.x, circuit, nr)

        # plt.plot(progress_history, label=f"number of shots = {i}, time : {t}s", linestyle = style)
        plt.plot(progress_history, label=f"number of layer = {l}, time : {t}s")


    plt.xlabel('number of iteration')
    plt.ylabel('value of cost function')
    # plt.title('Minimum encoding with different number of shots (layer = 4) with qiskit.primitives')
    plt.title('Minimum encoding with different number of layer (shots = 100000) with Aer')
    plt.legend()
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
    theta = np.array(list(theta.values()))
    return parameters, theta

#generate a circuit
def generate_circuit(nr:int, na:int, l:int, parameters:np.array)->qk.circuit.quantumcircuit.QuantumCircuit:
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
def define_pauli_op(nr:int, ancilla:bool=False)->list[SparsePauliOp]:
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
                #|0>
                pauli_op[i] = pauli_op[i].tensor(P0)
            else: 
                #|1> 
                pauli_op[i] = pauli_op[i].tensor(P1)

    #add ancilla qubit
    for i in range(len(pauli_op)):
        if ancilla == True:
            pauli_op[i] = pauli_op[i].tensor(P1)
        else:
            pauli_op[i] = pauli_op[i].tensor(Id)
    
    return pauli_op

#function to generate random QUBO matrix
def generate_random_qubo(nc:int)->np.ndarray:
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
    def cost_function(P1:np.array, P:np.array)->float:
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
                    # first_sum += A[i][j]*((P1[i]*P1[j])/(P[i]*P[j]))
                    first_sum += A[i][j]*(P1[i]/P[i])*(P1[j]/P[j])

        # second sum of cost function 
        second_sum = 0
        for i in range(nc):
            second_sum += A[i][i]*(P1[i]/P[i])

        result = first_sum + second_sum
        return  result
    
    return cost_function

#function that classical optimizer will minimize
def fun(params, ansatz, estimator, cost_function, H, Ha, progress_history:list = None):
    '''
    Function to compute value of cost function for minimal encoding

    Parameters
    ----------
    params : dict
        parameters of the circuit
    ansatz : qiskit.circuit.quantumcircuit.QuantumCircuit
        parameterised quantum circuit
    estimator : qiskit.primitives.estimator.Estimator or qiskit_aer.primitives.estimator.Estimator
        estimator to get expectation value
    cost_function : function
        cost function for minimal encoding
    H : list[SparsePauliOp]
        list of SparsePauliOp for each possible outcomes of register qubits in computational basis
    Ha : list[SparsePauliOp]
        list of SparsePauliOp for each possible outcomes in computational basis with ancilla qubit |1>
    progress_history : list, optional
        list to store the progress of the minimization. The default is None.
    
    Returns
    -------
    result : float
        value of the cost function
    '''
    #get expectation values from a circuit
    #get a expectation value of each H
    job1 = estimator.run([ansatz]*len(H),H,[params]*len(H))
    P = job1.result()
    # print(f"The primitive-job finished with result {P}")

    job2 = estimator.run([ansatz]*len(H),Ha,[params]*len(H))
    P1 = job2.result()
    print(f"The primitive-job finished with result {P1}")

    result = cost_function(P1.values, P.values)
    if progress_history is not None:
        progress_history.append(result)
    print(result)
    return result

def check_symmetric(A:np.ndarray, rtol=1e-05, atol=1e-08)->bool:
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
def get_ancilla_prob(theta:dict, 
                     circuit:qk.circuit.quantumcircuit.QuantumCircuit, 
                     nr:int,
                     session:Session = None,
                     num_shots:int = None)->np.array:
    ''' 
    Function to get final binary list from the circuit and optimised parameters.

    Parameters
    ----------
    theta : dict
        optimised parameters
    circuit : qiskit.circuit.quantumcircuit.QuantumCircuit
        parameterised quantum circuit
    nr : int
        number of register qubits
    
    Returns
    -------
    final_binary : numpy array
        final binary list
    
    Note
    ----
    This function compute final binary list based on ancilla qubit probability.
    First, it compute expectation value of each possible outcomes of register qubits, then it compute expectation value of each possible outcomes of register qubits when ancilla qubit is |1>.
    Then, it compute b_i which is possibility of ancilla qubit being |1>. Also, compute a_i which is possibility of ancilla qubit being |0>.
    '''
    if session is None:
        estimator = Estimator()
    else:
        estimator = Estimator(session=session)

    if num_shots is not None:
        estimator.set_options(shots=num_shots)

    #define observable to calculate expectation value
    H = define_pauli_op(nr, ancilla=False)
    Ha = define_pauli_op(nr,ancilla=True)
    #get expectation values from
    job1 = estimator.run([circuit]*len(H),H,[theta]*len(H))
    P = job1.result()
    print(f"The primitive-job finished with result {P}")
    job2 = estimator.run([circuit]*len(H),Ha,[theta]*len(H))
    P1 = job2.result()
    print(f"The primitive-job finished with result {P1}")
    #compute b_i and a_i
    b_sq=[] # list of probability of ancilla qubit being |1>
    a_sq=[] # list of probability of ancilla qubit being |0>
    for i in range(len(P.values)):
        val = (P1.values[i] / P.values[i])**2
        b_sq.append(val)
        a_sq.append(1 - b_sq[i])

    final_binary = []
    #compare a and b and pick the larger one
    for i in range(len(a_sq)):
        if a_sq[i] > b_sq[i]:
            final_binary.append(0)
        else:
            final_binary.append(1)

    print(f"b is Pr(x=1){b_sq}")
    print(f"a is Pr(x=0){a_sq}")
    #need to fix return value
    # print(f"final binary is {final_binary}")
    final_binary = np.array(final_binary)
    return final_binary

def get_sample(final_binary:np.array, energy:np.array)->dimod.SampleSet:
    ''' 
    Function to get sample from final binary and energy.

    Parameters
    ----------
    final_binary : numpy array
        final binary from get_ancilla_prob
    energy : numpy array
        value of the cost function with optimised parameters
    
    Returns
    -------
    sample : dimod.SampleSet
        sample set of final binary and energy
    '''
    sample = dimod.SampleSet.from_samples(
        dimod.as_samples(final_binary), 
        'BINARY', 
        energy=energy,
    )
    return sample


if __name__ == '__main__':
    main()
