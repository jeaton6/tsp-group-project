#!/usr/bin/python3

from which_pyqt import PYQT_VER
from math import inf
if PYQT_VER == 'PYQT5':
	from PyQt5.QtCore import QLineF, QPointF
elif PYQT_VER == 'PYQT4':
	from PyQt4.QtCore import QLineF, QPointF
else:
	raise Exception('Unsupported Version of PyQt: {}'.format(PYQT_VER))




import time
import numpy as np
from TSPClasses import *
import heapq
import itertools


def find_min(arr):
	smallest = inf
	for i in range(len(arr)):
		if arr[i] < smallest:
			smallest = arr[i]
	return smallest


def reduce_matrix(arr):
	low_bound = 0
	num_cities = len(arr[0])
	for i in range(num_cities):
		smallest = find_min(arr[i])
		if smallest == inf:
			continue
		low_bound += smallest
		for j in range(num_cities):
			arr[i][j] -= smallest
	for i in range(num_cities):
		smallest = find_min(arr[:,i])
		if smallest == inf:
			continue
		low_bound += smallest
		for j in range(num_cities):
			arr[j][i] -= smallest
	return low_bound, arr


def infinitize(arr, x, y):
	arr[:,y] = arr[:,y] + np.inf
	arr[x,:] = arr[x,:] + np.inf
	arr[y, x] = np.inf
	return arr


class State:
	def __init__(self, depth, lb, path, cost, city):
		self.depth = depth
		self.path = path
		self.lb = lb
		self.cost = cost
		self.key = self.lb / (self.depth ** 2)
		self.city = city

	def __lt__(self, other):
		return self.city._index < other.city._index

	def __eq__(self, other):
		return self.city._index == other.city._index

	def get_depth(self):
		return self.depth

	def get_path(self):
		return self.path

	def get_cost(self):
		return self.cost

	def get_lb(self):
		return self.lb


class TSPSolver:
	def __init__( self, gui_view ):
		self._scenario = None

	def setupWithScenario( self, scenario ):
		self._scenario = scenario


	''' <summary>
		This is the entry point for the default solver
		which just finds a valid random tour.  Note this could be used to find your
		initial BSSF.
		</summary>
		<returns>results dictionary for GUI that contains three ints: cost of solution, 
		time spent to find solution, number of permutations tried during search, the 
		solution found, and three null values for fields not used for this 
		algorithm</returns> 
	'''
	
	def defaultRandomTour( self, time_allowance=60.0 ):
		results = {}
		cities = self._scenario.getCities()
		ncities = len(cities)
		foundTour = False
		count = 0
		bssf = None
		start_time = time.time()
		while not foundTour and time.time()-start_time < time_allowance:
			# create a random permutation
			perm = np.random.permutation( ncities )
			route = []
			# Now build the route using the random permutation
			for i in range( ncities ):
				route.append( cities[ perm[i] ] )
			bssf = TSPSolution(route)
			count += 1
			if bssf.cost < np.inf:
				# Found a valid route
				foundTour = True
		end_time = time.time()
		results['cost'] = bssf.cost if foundTour else math.inf
		results['time'] = end_time - start_time
		results['count'] = count
		results['soln'] = bssf
		results['max'] = None
		results['total'] = None
		results['pruned'] = None
		return results


	''' <summary>
		This is the entry point for the greedy solver, which you must implement for 
		the group project (but it is probably a good idea to just do it for the branch-and
		bound project as a way to get your feet wet).  Note this could be used to find your
		initial BSSF.
		</summary>
		<returns>results dictionary for GUI that contains three ints: cost of best solution, 
		time spent to find best solution, total number of solutions found, the best
		solution found, and three null values for fields not used for this 
		algorithm</returns> 
	'''

	def greedy( self,time_allowance=60.0 ):
		start = time.time()
		retry = True
		count = 0
		while retry:
			retry = False
			count += 1
			start_city = self._scenario.getCities()[random.randint(0, len(self._scenario.getCities())-1)]
			visited = [start_city]
			curr_city = start_city
			sum = 0
			# pick starting node (random)
			while len(visited) < len(self._scenario.getCities()):
				shortest = inf
				for city in self._scenario.getCities():
					if city not in visited:
						curr_dist = curr_city.costTo(city)
						if curr_dist < shortest:
							shortest = curr_dist
							next_city = city
				if shortest == inf:
					retry = True
					break
				visited.append(next_city)
				sum += shortest
				curr_city = next_city
			final_dist = visited[-1].costTo(start_city)
			if final_dist != inf:  # Did we do this right?
				# visited.append(start_city)
				sum += final_dist
			else:
				retry = True
		end = time.time()
		bssf = TSPSolution(visited)
		results = {}
		results['cost'] = bssf.cost
		results['time'] = end - start
		results['count'] = count
		results['soln'] = bssf
		results['max'] = None
		results['total'] = None
		results['pruned'] = None
		return results
	
	
	
	''' <summary>
		This is the entry point for the branch-and-bound algorithm that you will implement
		</summary>
		<returns>results dictionary for GUI that contains three ints: cost of best solution, 
		time spent to find best solution, total number solutions found during search (does
		not include the initial BSSF), the best solution found, and three more ints: 
		max queue size, total number of states created, and number of pruned states.</returns> 
	'''

	def branchAndBound( self, time_allowance=60.0 ):
		start = time.time()
		cities = self._scenario.getCities()
		count, max_size, pruned, total = 0, 0, 0, 1
		num_cities = len(self._scenario.getCities())

		# Initializing the 2D array is O(n^2) time and space complexity, where n is the number of cities
		costs = np.zeros((num_cities, num_cities))

		# The greedy algorithm is O(n^2) time and space complexity
		bssf = self.greedy()
		for _ in range(4):
			sol = self.greedy(3)
			if sol['cost'] < bssf['cost']:
				bssf = sol
		bssf = (bssf['cost'], bssf['soln'])

		# Create the initial cost matrix
		# Time and space complexity is O(n^2)
		for i in range(num_cities):
			for j in range(num_cities):
				costs[i][j] = cities[i].costTo(cities[j])

		# Matrix cost reduction is O(n^2) time complexity with no added space complexity
		low_bound, costs = reduce_matrix(costs)
		start_state = State(1, low_bound, [cities[0]], costs, cities[0])
		pq = [(start_state.key, start_state)]
		heapq.heapify(pq)

		# Loop will run either until an optimal solution is found or until the time runs out (default 60 seconds)
		while time.time() - start < time_allowance and len(pq) > 0:
			if len(pq) > max_size:
				max_size = len(pq)

			# Popping off a heap queue is O(log n), where n is the number of items on the queue
			# The max number of items on the queue is almost always larger than the number of cities,
			# but still reasonably small (less than 1000 for all configurations tested up to number of cities = 50)
			curr_state = heapq.heappop(pq)[1]
			if curr_state.lb >= bssf[0]:
				pruned += 1
				continue
			elif len(curr_state.path) == num_cities:
				final_cost = TSPSolution(curr_state.path)
				if final_cost.cost < bssf[0]:
					count += 1
					bssf = (final_cost.cost, final_cost)
				else:
					pruned += 1
			else:
				# This will loop n times
				for city in cities:
					# This will loop n minus depth times (average is O(n/2), so it is still O(n))
					if city not in curr_state.path:
						if curr_state.city.costTo(city) < inf:
							# Infinitize is O(n), followed by a O(n^2) matrix reduction
							curr_lb, curr_cost = reduce_matrix(infinitize(curr_state.cost.copy(), curr_state.city._index, city._index))
							curr_path = curr_state.path.copy()
							curr_path.append(city)
							new_lb = curr_state.lb + curr_lb + curr_state.cost[curr_state.city._index, city._index]
							if new_lb < bssf[0]:
								new_state = State(curr_state.depth + 1, new_lb, curr_path, curr_cost, city)
								# Pushing a state onto the queue is O(log n), where n is the number of states on the queue
								heapq.heappush(pq, (new_state.key, new_state))
							else:
								pruned += 1
							total += 1

		results = {}
		results['cost'] = bssf[0]
		results['time'] = time.time() - start
		results['count'] = count
		results['soln'] = bssf[1]
		results['max'] = max_size
		results['total'] = total
		results['pruned'] = pruned + len(pq)
		return results


	''' <summary>
		This is the entry point for the algorithm you'll write for your group project.
		</summary>
		<returns>results dictionary for GUI that contains three ints: cost of best solution, 
		time spent to find best solution, total number of solutions found during search, the 
		best solution found.  You may use the other three field however you like.
		algorithm</returns> 
	'''
	def getFarthestCity(self, visited):
		cities = self._scenario.getCities()
		unvisited = []
		farthest = -1
		next_city = None
		for city in cities:
			if city not in visited:
				unvisited.append(city)
		for city in unvisited:
			dist = inf
			for start_city in visited:
				cost = start_city.costTo(city)
				cost_from = city.costTo(start_city)
				if min(cost, cost_from) < dist:
					dist = min(cost, cost_from)
			if dist >= farthest and dist != inf:
				farthest = dist
				next_city = city
		if farthest == inf:
			return None
		return next_city

	def getClosestCity(self, visited):
		cities = self._scenario.getCities()
		unvisited = []
		closest = inf
		next_city = None
		for city in cities:
			if city not in visited:
				unvisited.append(city)
		for city in unvisited:
			dist = -1
			for start_city in visited:
				cost = start_city.costTo(city)
				cost_from = city.costTo(start_city)
				if min(cost, cost_from) > dist:
					dist = min(cost, cost_from)
			if dist < closest:
				closest = dist
				next_city = city
		if closest == inf:
			return None
		return next_city

	def fancy( self,time_allowance=60.0 ):
		# random.seed(42)
		start = time.time()
		cities = self._scenario.getCities()
		start_city = cities[0]
		path = [start_city]
		farthest = self.getFarthestCity(path) # O(n^2) time complexity
		if farthest is None:
			print("no path exists")
		path.append(farthest)
		while len(path) < len(cities):  # loops n times
			# keep finding farthest
			farthest = self.getFarthestCity(path)  # time complexity O(n^2)
			if farthest is None:
				del path[random.randint(0, len(path)-1)]
				print ("removed an item when finding farthest, current path len:", len(path))
				if len(path) == 0:
					path.append(cities[random.randint(0, len(cities)-1)])
				continue
			# insert it into path
			cost = inf
			for i in range(len(path) + 1):  # loop n times at worst, n/2 average
				temp_path = path.copy()
				if i > len(path):
					temp_path.append(farthest)
				else:
					temp_path.insert(i, farthest)  # O(n) insert
				temp_cost = TSPSolution(temp_path).cost  # O(n) to find cost
				if temp_cost < cost:
					cheapest_path = temp_path
					cost = temp_cost
			# print_path = []
			# for city in path:
			# 	print_path.append(city._name)
			# print (print_path)
			if cost == inf:
				del path[random.randint(0, len(path)-1)]
				print ("removed an item when inserting, current path len:", len(path))
				if len(path) == 0:
					path.append(cities[random.randint(0, len(cities)-1)])
			else:
				path = cheapest_path.copy()

		end = time.time()
		bssf = TSPSolution(path)
		results = {}
		results['cost'] = bssf.cost
		results['time'] = end - start
		results['count'] = 0
		results['soln'] = bssf
		results['max'] = None
		results['total'] = None
		results['pruned'] = None
		return results



