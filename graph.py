
import numpy as np
import random as rd
import copy


# In[1]:
# graph adt

class graph:
    """Graph ADT"""

    def __init__(self):
        self.graph = {}
        self.visited = {}

    def append(self, vertexid, edge, weight):
        """add/update new vertex,edge,weight"""
        if vertexid not in self.graph.keys():
            self.graph[vertexid] = {}
            self.visited[vertexid] = 0
        if edge not in self.graph.keys():
            self.graph[edge] = {}
            self.visited[edge] = 0
        self.graph[vertexid][edge] = weight

    def reveal(self):
        """return adjacent list"""
        return self.graph

    def vertex(self):
        """return all vertices in the graph"""
        return list(self.graph.keys())

    def edge(self, vertexid):
        """return edge of a particular vertex"""
        return list(self.graph[vertexid].keys())

    def edge_reverse(self, vertexid):
        """return vertices directing to a particular vertex"""
        return [i for i in self.graph if vertexid in self.graph[i]]

    def weight(self, vertexid, edge):
        """return weight of a particular vertex"""
        return (self.graph[vertexid][edge])

    def order(self):
        """return number of vertices"""
        return len(self.graph)

    def visit(self, vertexid):
        """visit a particular vertex"""
        self.visited[vertexid] = 1

    def go(self, vertexid):
        """return the status of a particular vertex"""
        return self.visited[vertexid]

    def route(self):
        """return which vertices have been visited"""
        return self.visited

    def degree(self, vertexid):
        """return degree of a particular vertex"""
        return len(self.graph[vertexid])

    def mat(self):
        """return adjacent matrix"""
        self.matrix = [[0 for _ in range(max(self.graph.keys()) + 1)] for i in range(max(self.graph.keys()) + 1)]
        for i in self.graph:
            for j in self.graph[i].keys():
                self.matrix[i][j] = 1
        return self.matrix

    def remove(self, vertexid):
        """remove a particular vertex and its underlying edges"""
        for i in self.graph[vertexid].keys():
            self.graph[i].pop(vertexid)
        self.graph.pop(vertexid)

    def disconnect(self, vertexid, edge):
        """remove a particular edge"""
        del self.graph[vertexid][edge]

    def clear(self, vertexid=None, whole=False):
        """unvisit a particular vertex"""
        if whole:
            self.visited = dict(zip(self.graph.keys(), [0 for i in range(len(self.graph))]))
        elif vertexid:
            self.visited[vertexid] = 0
        else:
            assert False, "arguments must satisfy whole=True or vertexid=int number"


# In[2]:
# algorithms


#
def bfs(ADT, current):
    """Breadth First Search"""

    # create a queue with rule of first-in-first-out
    queue = []
    queue.append(current)

    while queue:

        # keep track of the visited vertices
        current = queue.pop(0)
        ADT.visit(current)

        for newpos in ADT.edge(current):

            # visit each vertex once
            if ADT.go(newpos) == 0 and newpos not in queue:
                queue.append(newpos)


def dfs_itr(ADT, current):
    """Depth First Search without Recursion"""

    queue = []
    queue.append(current)

    # the loop is the backtracking part when it reaches cul-de-sac
    while queue:

        # keep track of the visited vertices
        current = queue.pop(0)
        ADT.visit(current)

        # priority queue
        smallq = []

        # find children and add to the priority
        for newpos in ADT.edge(current):
            if ADT.go(newpos) == 0:

                # if the child vertex has already been in queue
                # move it to the frontline of queue
                if newpos in queue:
                    queue.remove(newpos)
                smallq.append(newpos)

        queue = smallq + queue



def dfs(ADT, current):
    """Depth First Search"""

    # keep track of the visited vertices
    ADT.visit(current)

    # the loop is the backtracking part when it reaches cul-de-sac
    for newpos in ADT.edge(current):

        # if the vertex hasnt been visited
        # we call dfs recursively
        if ADT.go(newpos) == 0:
            dfs(ADT, newpos)



def dfs_topo_sort(ADT, current):
    """Topological sort powered by recursive DFS to get linear ordering"""

    # keep track of the visited vertices
    ADT.visit(current)
    yield current

    # the loop is the backtracking part when it reaches cul-de-sac
    for newpos in ADT.edge(current):

        # if the vertex hasnt been visited
        # we call dfs recursively
        if ADT.go(newpos) == 0:
            yield from dfs_topo_sort(ADT, newpos)


def bfs_path(ADT, start, end):
    """Breadth First Search to find the path from start to end"""

    # create a queue with rule of first-in-first-out
    queue = []
    queue.append(start)

    # pred keeps track of how we get to the current vertex
    pred = {}

    while queue:

        # keep track of the visited vertices
        current = queue.pop(0)
        ADT.visit(current)

        for newpos in ADT.edge(current):

            # visit each vertex once
            if ADT.go(newpos) == 0 and newpos not in queue:
                queue.append(newpos)
                pred[newpos] = current

        # traversal ends when the target is met
        if current == end:
            break

    # create the path by backtracking
    # trace the predecessor vertex from end to start
    previous = end
    path = []
    while pred:
        path.insert(0, previous)
        if previous == start:
            break
        previous = pred[previous]

    # note that if we cant go from start to end
    # we may get inf for distance
    # additionally, the path may not include start position
    return len(path) - 1, path



def dfs_path(ADT, start, end):
    """Depth First Search to find the path from start to end"""

    queue = []
    queue.append(start)

    # pred keeps track of how we get to the current vertex
    pred = {}

    # the loop is the backtracking part when it reaches cul-de-sac
    while queue:

        # keep track of the visited vertices
        current = queue.pop(0)
        ADT.visit(current)

        # priority queue
        smallq = []

        # find children and add to the priority
        for newpos in ADT.edge(current):
            if ADT.go(newpos) == 0:

                # if the child vertex has already been in queue
                # move it to the frontline of queue
                if newpos in queue:
                    queue.remove(newpos)
                smallq.append(newpos)
                pred[newpos] = current

        queue = smallq + queue

        # traversal ends when the target is met
        if current == end:
            break

    # create the path by backtracking
    # trace the predecessor vertex from end to start
    previous = end
    path = []
    while pred:
        path.insert(0, previous)
        if previous == start:
            break
        previous = pred[previous]

    # note that if we cant go from start to end
    # we may get inf for distance
    # additionally, the path may not include start position
    return len(path) - 1, path



def dijkstra(ADT, start, end):
    """Dijkstra's Algorithm to find the shortest path"""

    # all weights in dcg must be positive
    # otherwise we have to use bellman ford instead
    neg_check = [j for i in ADT.reveal() for j in ADT.reveal()[i].values()]
    assert min(neg_check) >= 0, "negative weights are not allowed, please use Bellman-Ford"

    # queue is used to check the vertex with the minimum weight
    queue = {}
    queue[start] = 0

    # distance keeps track of distance from starting vertex to any vertex
    distance = {}
    for i in ADT.vertex():
        distance[i] = float('inf')
    distance[start] = 0

    # pred keeps track of how we get to the current vertex
    pred = {}

    # dynamic programming
    while queue:

        # vertex with the minimum weight in queue
        current = min(queue, key=queue.get)
        queue.pop(current)

        for j in ADT.edge(current):

            # check if the current vertex can construct the optimal path
            if distance[current] + ADT.weight(current, j) < distance[j]:
                distance[j] = distance[current] + ADT.weight(current, j)
                pred[j] = current

            # add child vertex to the queue
            if ADT.go(j) == 0 and j not in queue:
                queue[j] = distance[j]

        # each vertex is visited only once
        ADT.visit(current)

        # traversal ends when the target is met
        if current == end:
            break

    # create the shortest path by backtracking
    # trace the predecessor vertex from end to start
    previous = end
    path = []
    while pred:
        path.insert(0, previous)
        if previous == start:
            break
        previous = pred[previous]

    # note that if we cant go from start to end
    # we may get inf for distance[end]
    # additionally, the path may not include start position
    return distance[end], path



def bellman_ford(ADT, start, end):
    """Bellman-Ford Algorithm,
    a modified Dijkstra's algorithm to detect negative cycle"""

    # distance keeps track of distance from starting vertex to any vertex
    distance = {}
    for i in ADT.vertex():
        distance[i] = float('inf')
    distance[start] = 0

    # pred keeps track of how we get to the current vertex
    pred = {}

    # dynamic programming
    for _ in range(1, ADT.order() - 1):
        for i in ADT.vertex():
            for j in ADT.edge(i):
                try:
                    if distance[i] + ADT.weight(i, j) < distance[j]:
                        distance[j] = distance[i] + ADT.weight(i, j)
                        pred[j] = i

                except KeyError:
                    pass

    # detect negative cycle
    for k in ADT.vertex():
        for l in ADT.edge(k):
            try:
                assert distance[k] + ADT.weight(k, l) >= distance[l], 'negative cycle exists!'
            except KeyError:
                pass

    # create the shortest path by backtracking
    # trace the predecessor vertex from end to start
    previous = end
    path = []
    while pred:
        path.insert(0, previous)
        if previous == start:
            break
        previous = pred[previous]

    return distance[end], path



def a_star(ADT, start, end):
    """A* Algorithm,
    a generalized Dijkstra's algorithm with heuristic function to reduce execution time"""

    # all weights in dcg must be positive
    # otherwise we have to use bellman ford instead
    neg_check = [j for i in ADT.reveal() for j in ADT.reveal()[i].values()]
    assert min(neg_check) >= 0, "negative weights are not allowed, please use Bellman-Ford"

    # queue is used to check the vertex with the minimum summation
    queue = {}
    queue[start] = 0

    # distance keeps track of distance from starting vertex to any vertex
    distance = {}

    # heuristic keeps track of distance from ending vertex to any vertex
    heuristic = {}

    # route is a dict of the summation of distance and heuristic
    route = {}

    # criteria
    for i in ADT.vertex():
        # initialize
        distance[i] = float('inf')

        # manhattan distance
        heuristic[i] = abs(i[0] - end[0]) + abs(i[1] - end[1])

    # initialize
    distance[start] = 0

    # pred keeps track of how we get to the current vertex
    pred = {}

    # dynamic programming
    while queue:

        # vertex with the minimum summation
        current = min(queue, key=queue.get)
        queue.pop(current)

        # find the minimum summation of both distances
        minimum = float('inf')

        for j in ADT.edge(current):

            # check if the current vertex can construct the optimal path
            # from the beginning and to the end
            distance[j] = distance[current] + ADT.weight(current, j)
            route[j] = distance[j] + heuristic[j]
            if route[j] < minimum:
                minimum = route[j]

        for j in ADT.edge(current):

            # only append unvisited and unqueued vertices
            if (route[j] == minimum) and (ADT.go(j) == 0) and (j not in queue):
                queue[j] = route[j]
                pred[j] = current

        # each vertex is visited only once
        ADT.visit(current)

        # traversal ends when the target is met
        if current == end:
            break

            # create the shortest path by backtracking
    # trace the predecessor vertex from end to start
    previous = end
    path = []
    while pred:
        path.insert(0, previous)
        if previous == start:
            break
        previous = pred[previous]

    # note that if we cant go from start to end
    # we may get inf for distance[end]
    # additionally, the path may not include start position
    return distance[end], path



def prim(ADT, start):
    """Prim's Algorithm to find a minimum spanning tree"""

    # initialize
    queue = {}
    queue[start] = 0

    # route keeps track of how we travel from one vertex to another
    route = {}
    route[start] = start

    # result is a list that keeps the order of vertices we have visited
    result = []

    # pop the edge with the smallest weight
    while queue:

        # note that when we have two vertices with the same minimum weights
        # the dictionary would pop the one with the smallest key
        current = min(queue, key=queue.get)
        queue.pop(current)
        result.append(current)
        ADT.visit(current)

        # BFS
        for i in ADT.edge(current):
            if i not in queue and ADT.go(i) == 0:
                queue[i] = ADT.weight(current, i)
                route[i] = current

            # every time we find a smaller weight
            # we need to update the smaller weight in queue
            if i in queue and queue[i] > ADT.weight(current, i):
                queue[i] = ADT.weight(current, i)
                route[i] = current

                # create minimum spanning tree
    subset = graph()
    for i in result:
        if i != start:
            subset.append(route[i], i, ADT.weight(route[i], i))
            subset.append(i, route[i], ADT.weight(route[i], i))

    return subset



def kruskal(ADT):
    """Kruskal's Algorithm to find the minimum spanning tree"""

    # use dictionary to sort edges by weight
    D = {}
    for i in ADT.vertex():
        for j in ADT.edge(i):

            # get all edges
            if f'{j}-{i}' not in D.keys():
                D[f'{i}-{j}'] = ADT.weight(i, j)

    sort_edge_by_weight = sorted(D.items(), key=lambda x: x[1])

    result = []

    # use disjointset to detect cycle
    disjointset = {}
    for i in ADT.vertex():
        disjointset[i] = i

    for i in sort_edge_by_weight:

        parent = int(i[0].split('-')[0])
        child = int(i[0].split('-')[1])

        # first check disjoint set
        # if it already has indicated cycle
        # trace_root function will go to infinite loops
        if disjointset[parent] != disjointset[child]:

            # if we trace back to the root of the tree
            # and it indicates no cycle
            # we update the disjoint set and add edge into result
            if trace_root(disjointset, parent) != trace_root(disjointset, child):
                disjointset[child] = parent
                result.append([parent, child])

                # create minimum spanning tree
    subset = graph()
    for i in result:
        subset.append(i[0], i[1], ADT.weight(i[0], i[1]))
        subset.append(i[1], i[0], ADT.weight(i[0], i[1]))

    return subset



def get_degree_list(ADT):
    """create degree distribution"""

    D = {}

    # if the current degree hasnt been checked
    # we create a new key under the current degree
    # otherwise we append the new node into the list
    for i in ADT.vertex():
        try:
            D[ADT.degree(i)].append(i)

        except KeyError:
            D[ADT.degree(i)] = [i]

    # dictionary is sorted by key instead of value in ascending order
    D = dict(sorted(D.items()))

    return D



def sort_by_degree(ADT):
    """sort vertices by degree"""

    dic = {}
    for i in ADT.vertex():
        dic[i] = ADT.degree(i)

    # the dictionary is sorted by value and exported as a list in descending order
    output = [i[0] for i in sorted(dic.items(), key=lambda x: x[1])]

    return output[::-1]



def dsatur(ADT):
    """graph coloring with dsatur algorithm"""

    # step 1
    # sort vertices by their degrees
    # check matula beck section in the below link for more details
    # https://github.com/je-suis-tm/graph-theory/blob/master/k%20core.ipynb
    # pick the vertex with the largest degree
    selected_vertex = sort_by_degree(ADT)[0]

    # initialize saturation degree
    saturation_degrees = dict(zip(ADT.vertex(),
                                  [0] * ADT.order()))

    # according to brooks theorem
    # upper bound of chromatic number equals to maximum vertex degree plus one
    chromatic_number_upper_bound = range(ADT.degree(selected_vertex) + 1)

    # step 2
    # assign the first color to the vertex with the maximum degree
    color_assignments = {}
    color_assignments[selected_vertex] = 0

    # fill each vertex with color
    while len(color_assignments) < ADT.order():

        # saturation degrees also serve as a queue
        # remove colored vertex from the queue
        saturation_degrees.pop(selected_vertex)

        # update saturation degrees after color assignment
        for node in ADT.edge(selected_vertex):
            if node in saturation_degrees:
                saturation_degrees[node] += 1

        # step 3
        # among uncolored vertices
        # pick a vertex with the largest saturation degree
        check_vertices_degree = [node for node in saturation_degrees if
                                 saturation_degrees[node] == max(saturation_degrees.values())]

        # if there is an equality, choose one with the largest degree
        if len(check_vertices_degree) > 1:
            degree_distribution = [ADT.degree(node) for node in check_vertices_degree]
            selected_vertex = check_vertices_degree[degree_distribution.index(max(degree_distribution))]
        else:
            selected_vertex = check_vertices_degree[0]

        # step 4
        # exclude colors used by neighbors
        # assign the least possible color to the selected vertex
        excluded_colors = [color_assignments[node] for node in ADT.edge(selected_vertex) if node in color_assignments]
        selected_color = [color for color in chromatic_number_upper_bound if color not in excluded_colors][0]
        color_assignments[selected_vertex] = selected_color

    return color_assignments



def get_maximal_independent_set(ADT):
    """fast randomized algorithm to fetch one of the maximal independent sets"""

    # assign random value from uniform distribution to every vertex
    random_val = dict(zip(ADT.vertex(),
                          [rd.random() for _ in range(ADT.order())]))

    # initialize
    maximal_independent_set = []
    queue = [i for i in random_val]

    while len(queue) > 0:
        for node in queue:

            # select the vertex which has larger value than all of its neighbors
            neighbor_vals = [random_val[neighbor] for neighbor in ADT.edge(node) if neighbor in random_val]
            if len(neighbor_vals) == 0 or random_val[node] < min(neighbor_vals):

                # add to mis
                maximal_independent_set.append(node)

                # remove the vertex and its neighbors
                queue.remove(node)
                for neighbor in ADT.edge(node):
                    if neighbor in queue:
                        queue.remove(neighbor)

        # reassign random values to existing vertices
        random_val = dict(zip(queue,
                              [rd.random() for _ in range(len(random_val))]))

    return maximal_independent_set







