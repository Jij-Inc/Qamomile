from minimal_encoding import *
from minimal_encoding_ex import *

import jijmodeling as jm
import jijmodeling.transpiler as jmt
import openjij as oj

import warnings
import geocoder as gc
import math
import numpy as np
import matplotlib.pyplot as plt
import time
from abc import ABC, abstractmethod
from typing import List

def main():
    '''
    This is reproduction of the following paper:
    https://arxiv.org/abs/2306.08507
    '''
    #set the warning off
    warnings.filterwarnings('ignore')
    # Define the problem
    problem = set_problem()
    #list of points, the first location is the depot
    # points = ['千代田区' ,'練馬区', '品川区', '文京区', '荒川区']
    points = ['千代田区' ,'練馬区','品川区','文京区','港区','足立区','杉並区','葛飾区','江戸川区','世田谷区','渋谷区','江東区','荒川区','北区','目黒区','板橋区','中央区']
    n = [index for index in range(len(points))] 

    #get geo data 
    geo_data, distance_data = geo_information(points)
    #generate routes
    routes = greedy_route_generating(points, 4)
    routes['n'] = n

    print(f'routes: {routes}')
    # compile problem
    compiled_model = jmt.core.compile_model(problem, routes, {})
    # # Quadratic Unconstraint Binary Optimization (QUBO) model
    pubo_builder = jmt.core.pubo.transpile_to_pubo(compiled_model=compiled_model)
    # # qubo, const = pubo_builder.get_qubo_dict(multipliers = {'one-city': 0.5, 'one-time': 0.5})
    qubo, const = pubo_builder.get_qubo_dict(multipliers = {'one-time': 500})

    #========= openJij ===================

    sampler = oj.SASampler()

    # solve problem
    # result = sampler.sample_qubo(qubo) 
    result = sampler.sample_qubo(qubo, num_reads=10)

    #get solution
    vrp_solution(result, pubo_builder, compiled_model, geo_data, distance_data ,routes)
    

    # ============== minimal encoding ==============

    # nc = len(routes['route'])
    # nr = math.log2(nc)
    # nr = int(nr)
    # l = 4
    # na = 1
    # nq = int(nr + na)
    # #minimal encoding
    # A = convert_qubo_datatype(qubo, nc)
    # # print(A)
    # A = (A + A.T)
    # for i in range(len(A)):
    #     A[i][i] = A[i][i]/2
    # if check_symmetric(A) == False:
    #     print("The QUBO matrix is not symmetric")
    #     return 0
    
    # final_binary = []
    # energy = []
    # for i in range(10):
    
    #     parameters, theta = init_parameter(nq, l) 
    #     circuit = generate_circuit(nr, na, l, parameters)
    #     progress_history = []
        
    #     # # print(A)
    #     func = init_func(nc, nr, na, circuit, A, progress_history)
    #     n_eval = 1000
    #     optimizer = COBYLA(maxiter=n_eval, disp=False, tol=0.00001)
    #     # optimizer = ADAM(maxiter=n_eval)
    #     result = optimizer.minimize(func, list(theta.values()))
    #     energy = np.array(result.fun)
    #     energy.append(result.fun)
    #     final_binary.append(get_ancilla_prob(result.x, circuit, nr))


    # sample = get_sample(final_binary, energy)
    # vrp_solution(sample, pubo_builder, compiled_model, geo_data, distance_data, routes)

    return 0 


def set_problem():
    '''
    Function to define the VRPTW problem using JijModeling. 
    This formulation is based on the following paper:
    https://arxiv.org/abs/2306.08507 

    Returns
    -------
    problem : JijModeling Problem

    '''
    # Define the problem
    c = jm.Placeholder('c', ndim=1) #cost of each route
    n = jm.Placeholder('n', ndim=1) #node (location)
    d = jm.Placeholder('d', ndim=2).set_latex(r'\delta') #delta

    R = c.shape[0].set_latex("R") #set of routes
    N = n.shape[0].set_latex(r'\mathscr{N}') #set of nodes which is locations
    r = jm.Element('r', belong_to=(R)) 
    i = jm.Element('i', belong_to=(1,N))
    x = jm.BinaryVar('x', shape = R)


    problem = jm.Problem('VRPTW')
    problem += jm.sum(r, c[r]*x[r])
    const = jm.sum(r, d[r,i]*x[r])
    problem += jm.Constraint("one-time", const==1, forall=i)

    return problem

def geo_information(points:list[str]):
    '''
    Function to get geo information from the list of locations and distance between those locations.
    
    Parameters
    ----------
    points : list[str]
        list of points (locations)
    
    Retrns
    -------
    geo_data : dict
        dictionary of geo information, points and latlng_list
    distance_data : dict
        dictionary of distance information, distance matrix
    '''
     # get the latitude and longitude
    latlng_list = []
    try:
        for point in points:
            location = gc.osm(point)
            latlng_list.append(location.latlng)
        # make distance matrix
        num_points = len(points)
        inst_d = np.zeros((num_points, num_points))
        for i in range(num_points):
            for j in range(num_points):
                a = np.array(latlng_list[i])
                b = np.array(latlng_list[j])
                inst_d[i][j] = np.linalg.norm(a-b)
        geo_data = {'points': points, 'latlng_list': latlng_list}
        distance_data = {'d': inst_d}
    except:
        print("Your machine may not connect to the internet. Please check your connection.")
        print("If that's not the case, there may be a problem with the location you entered.")
        return 0

    # print(f'geo_data: {geo_data}')
    # print(f'distance: {distance_data}')

    return geo_data, distance_data

def find_min_distance(distance:list[float])->int:
    ''' 
    Function to find the index of the second minimum value in the numpy array

    Parameters
    ----------
    distance : list[float]
        list of distance from one location to the other locations
    
    Returns
    -------
    min_index : int
        index of the second minimum value
    '''
    #copy the distance array 
    min_distance = min(distance)
    min_index = np.where(distance == min_distance)[0][0]
    # print(f'min_index: {min_index}')
    return min_index

def find_next_location(current_location:int, visited_location:list[int], distance_matrix:list[list[float]])->int:
    ''' 
    Function to find the next location based on the current location and visited location

    Parameters
    ----------
    current_location : int
        index of current location
    visited_location : list[int]
        list of visited location
    distance_matrix : list[list[float]]
        distance matrix
    
    Returns
    -------
    next_location : int
        index of next location
    '''
    #get the distance from current location to other locations
    distance = distance_matrix[current_location].copy()
    #set the distance of visited location to infinity
    # print(f'distance 1: {distance}')
    for i in visited_location:
        distance[i] = math.inf
    # print(f'distance 2: {distance}')
    #find the next location
    next_location = find_min_distance(distance)
    return next_location

def get_delta(route:list[int], n:int)->list:
    '''
    Function to get delata from the route. Delta equal to one if a location is lie on the route, otherwise zero.

    Parameters
    ----------
    route : list[int]
        list of location index
    
    n : int
        number of locations
    
    Returns
    -------
    delta : list[int]
        list of delta
    '''

    delta = np.zeros(n)
    for i in route:
        delta[i] = int(1)
    
    return list(delta)

def plot_route(routes:list[list[int]], geo_data:dict):
    '''
    Function to plot the routes
    '''
    #get the latitude and longitude and distance matrix
    points = geo_data['points']
    latlng_list = geo_data['latlng_list']

    #set the plot so you can plot Japanese
    plt.rcParams['font.family'] = 'Hiragino Maru Gothic Pro'
    #plot the points
    for i in range(len(points)):
        plt.plot(latlng_list[i][0], latlng_list[i][1], marker='o')
        plt.text(latlng_list[i][0], latlng_list[i][1], points[i])
    #plot the route
    for route in routes:
        route_points = []
        for i in route:
            route_points.append(latlng_list[i])
        route_points = np.array(route_points)
        plt.plot(route_points[:,0], route_points[:,1], marker='o')

    plt.title('ルート')
    plt.show()

    return 0

def greedy_route_generating(points:list[str], v:int):
    '''
    Function to randomly generate routes with greedy method
    
    Parameters
    ----------
    points : list[str]
        list of points (locations)
    v : int
        number of vehicles
    
    Retrns
    -------
    routes : dict
        dictionary of routes and cost
    '''
    #get the latitude and longitude and distance matrix
    geo_data, distance_data = geo_information(points)
    distance_matrix = distance_data['d']

    #set the maximum number of location that each vechicle can visit
    p = math.ceil((len(points)-1) / v)

    #initialized visited location 
    visited_location = []
    #generate possible routes
    routes = {
        'route': [],
        'c': [], 
        'd': [], 
        'n': []
    }
    visited_location = []   
    initial_route = []
    for i in range(1, len(points)):
        route = [0, i]
        initial_route.append(route) 
    # print(f'routes: {routes}')
    # print(f'test{routes[0][-1]}')
    # print(f'visited_location: {visited_location}')
    #generate routes
    for i in range(len(points)-1):
        # visited = [0]
        visited_location.append([0])
        count = 0 
        # print(f'visited_location: {visited_location}')
        # print(f'len(visited_location[i]): {len(visited_location[i])}')
        # print(f'len(points)-1: {len(points)-1}')
        while len(visited_location[-1]) != len(points):
            count += 1
            route = [0]
            cost = 0
            routes['route'].append(route)
            routes['c'].append(cost)
            # print(f'range p {p}')
            for j in range(p):
                if count == 1 and j == 0:
                    current_location = routes['route'][-1][-1]
                    next_location = initial_route[i][-1]
                    cost += distance_matrix[current_location][next_location]
                    routes['route'][-1].append(next_location)
                    visited_location[-1].append(next_location)
                else:
                    current_location = routes['route'][-1][-1]
                    next_location = find_next_location(current_location, visited_location[-1], distance_matrix)
                    cost += distance_matrix[current_location][next_location]
                    routes['route'][-1].append(next_location)
                    visited_location[-1].append(next_location)
                
            delta = get_delta(routes['route'][-1], len(points))
            routes['d'].append(delta)
            routes['route'][-1].append(0)
            routes['c'][-1] = cost
    # print(f'visited_location: {visited_location}')

    #plot a route 
    # print(f'routes: {routes}')
    # for route in routes['route']:
    #     print(np.array(geo_data['points'])[route])
    # plot_route(routes['route'], geo_data)

    return routes

# ====== Code provided by 松山さん ========
class LocalSearch(ABC):
    def __init__(self, distance_matrix: List[List[float]]):
        self.distance_matrix = np.array(distance_matrix)
        super().__init__()

    @abstractmethod
    def local_search(self, initial_solution: List[int]) -> List[int]:
        raise NotImplementedError

    def calculate_cost(self, solution: List[int]) -> float:
        return (
            np.sum(self.distance_matrix[solution[:-1], solution[1:]])
            + self.distance_matrix[solution[-1], solution[0]]
        )

class TwoOpt(LocalSearch):
    def local_search(self, initial_solution: List[int]) -> List[int]:
        best_solution = initial_solution
        improved = True

        while improved:
            improved = False
            for i in range(1, len(best_solution) - 2):
                for j in range(i + 1, len(best_solution)):
                    if j - i == 1:
                        continue  # changes nothing, skip then
                    new_solution = best_solution[:]
                    new_solution[i:j] = best_solution[
                        j - 1 : i - 1 : -1
                    ]  # this is the 2-optSwap
                    if self.calculate_cost(new_solution) < self.calculate_cost(
                        best_solution
                    ):
                        best_solution = new_solution
                        improved = True
        return best_solution
#=====================================

def two_opt_for_vrp(distance_data, routes, results):
    '''
    Function to apply TwoOpt to VRP. 

    Parameters
    ----------
    distance_data : dict
        distance matrix
    routes : dict
        dictionary of all routes
    results : list[int]
        list of indices of optimal routes
    
    Returns
    -------
    sol_routes : list[list[int]]
        list of optimised routes
    '''
    to = TwoOpt(distance_data)
    sol_routes = []
    for result in results:
        routes[result] = to.local_search(initial_solution=routes[result]) 
        sol_routes.append(routes[result])
    return sol_routes

def vrp_solution(result:oj.sampler.response.Response, 
                 pubo_builder:jmt.core.pubo.pubo_builder.PuboBuilder,
                 compiled_model:jmt.core.compile.compiled_model.CompiledInstance,
                 geo_data:dict,
                 distance_data:dict,
                 routes:dict):
    '''
    Function to decode OpenJij response to JijModeling sample set and convert it to a solution of VRP.
    It prints out result and plot a solution.

    Parameters
    ----------
    result : oj.sampler.resposne.Response
        OpenJij response, result of optimisation
    pubo_builder : jmt.core.pubo.PUBOBuilder
        PUBO Builder
    compiled_model : jmt.core.model.CompiledModel
        compiled model
    geo_data : dict
        dictionary of geo information, points and latlng_list
    
    Returns
    -------
    nonzero_indices : list[int]
        list of nonzero indices, this indicate which routes are optimal solution for thew problem

    Notes
    -----
    Pubo_builder, compiled_model and data are ones that you define for TSP.
    '''
    results = jmt.core.pubo.decode_from_openjij(result, pubo_builder, compiled_model)
    feasibles = results.feasible()
    print(f'feasibles: {feasibles}')
    objectives = np.array(feasibles.evaluation.objective)
    lowest_index = np.argmin(objectives)
    print(f"Lowest solution index: {lowest_index}, Lowest objective value: {objectives[lowest_index]}")
    nonzero_indices, nonzero_values, shape = feasibles.record.solution["x"][lowest_index]
    print("indices: ", nonzero_indices)
    print("values: ", nonzero_values)

    #get the routes from the indices
    # optimised_routes = get_routs_from_index(nonzero_indices[0], routes)
    #apply TwoOpt
    optimised_routes = two_opt_for_vrp(distance_data['d'], routes['route'], nonzero_indices[0])
    plot_route(optimised_routes, geo_data)

    return nonzero_indices[0]

if __name__ == '__main__':
    main()