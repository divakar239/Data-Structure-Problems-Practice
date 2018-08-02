#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 20 19:15:41 2018

@author: DK
"""
import collections
from collections import defaultdict

### Graphs ###
class graph:
    def __init__(self,vertices):
        #defaultdict holds an empty list at each index; using adjecency list to create graph
        self.graph = defaultdict(list)
        self.V = vertices
        self.E = edges
    
    class egde:
        def __init__(self,x,y):
            self.v1 = x
            self.v2 = y
    
    def add_edge(self,u,v):
        #for directed graphs
        self.graph[u].append(v)
        
        #for undirected graphs
        #self.graph[u].append(v)
        #self.graph[v].append(u)
        #e = edge(u,v)
        #edges.append(e)
        
    def add_egde_weight(self,u,v,w):
        self.graph[u].append((v,w)) #adds a tuple to u where v is the destibation and w is the weight of the edge.
        
        
    #Carry out a bfs from a source 's'
    def bfs(self,s):
        #initialising list to keep track of all visited nodes, as false
        visited = [False]*(len(self.graph))
        
        #Creating a queue and pushing the source in it.
        queue = []
        queue.append(s)
        visited[s] = True
        
        while queue:
            e = queue.pop(0)
            print(e)
            for i in range(0,len(self.graph[e])):
                if visited[i] == False:
                    queue.append(i)
                    visited[i] == True


#shortest single source path in unweighted graph: bfs
    def bfs_shortest_path(self,s):
        visited = [False]*(len(self.graph))
        queue = []
        distances = [-1]*(len(self.graph))      #distances of all vertices
        queue.append(s)
        visited[s] = True
        distances[s] = 0                        #set distance of the source as 0
        
        while(queue):
            e = queue.pop()
            for i in graph[e]:
                if visited[i] == False:
                    if distances[i] == -1:
                        distances[i] = distances[e] + 1 #accumulates the sum of distances from the source 
                        queue.append(i)
                        visited[i] = True

    #Carry out DFS from a source
    def DFS(self,v):
        visited = [False]*(len(self.graph))
        self.DFS_util(v,visited)
        
    def DFS_util(self,v, visited):
        visited[v] == True
        print(v)
        
        for i in self.graph[v]:
            if visited[i] == False:
                return self.DFS_util(i,visited)

# 1) Detect a cycle in the graph
#to detect a back edge, we can keep track of vertices currently in recursion stack of 
#function for DFS traversal. If we reach a vertex that is already in the recursion stack, 
#then there is a cycle in the tree. The edge that connects current vertex to 
#the vertex in the recursion stack is back edge. 
#We have used recStack[] array to keep track of vertices in the recursion stack.
    def detect_cycle(self):
        visited = [False]*len(self.graph)
        rec_stack = [False]*len(self.graph)   # keeps track of all vertices in the stack and helps detect cycle
        for v in self.graph:
            if visited[v] == False:
                if self.detect_cycle_util(v,visited,rec_satck) == True:
                    return True
        return False
            
    def detect_cycle_util(self,v,visited,rec_stack):
        visited[v] = True
        rec_stack[v] = True
        
        for node in self.graph[v]:
            if visited[node] == False:
                if detect_cycle_util(node,visited,rec_stack):
                    return True 
            elif rec_stack[node] == True:
                    return True
        rec_stack[v] == False
        return False
        
# Disjoint sets to find cycle in undirected graph
# we need find parent and construct Union functions
    #recursive check on the parent_array for vertices to find the root of the disjoint set
    def find_parent(self,parent_array,vertex):
        if parent_array[vertex] == -1:
            return vertex
        else:
            return find_parent(parent_array,parent_array[vertex])
    #checks parents of both vertices and then points x to y to encompass them in one set
    def union(self,parent_array,x,y):
        x_set = self.find_parent(parent_array,x)
        y_set = self.find_parent(parent_array,y)
        parent_array[x_set] = y_set

#For each edge, make subsets using both the vertices of the edge.
#If both the vertices are in the same subset, a cycle is found.
# Otherwise add the two vertices of the edge in the same subset by union and report that the edge doesn't form the cylce
    def detect_cycle_unweighted(self):
        parent = [-1]*(len(self.graph))
        
        for e in E:
            x = self.find_parent(parent, e.v1)
            y = self.find_parent(parent, e.v2)
            if x == y:
                return True          #back edge detected as the parent and the child have the same parent
            self.union(e.v1,e.v2)    # else put them both in the same set since they constitute an edge
        
#Count number of forests in a graph:
#Approach :
#1. Apply DFS on every node.
#2. Increment count by one if every connected node is visited from one source.
#3. Again perform DFS traversal if some nodes yet not visited.
#4. Count will give the number of trees in forest.

    def count_trees(self):
        visited = [False]*len(self.graph)
        res = 0                   # keeps count of number of trees in the graph
        for i in range(len(self.graph)):
            if visited[i] == False:
                count_trees_util(i,visited)
                res = res + 1     # everytime we complete dfs from 1 node we increment the count by 1
        return res
        
    def count_trees_util(v,visited):
        visited[v] == True
        for i in self.graph[v]:
            if visited[i] == False:
                count_trees_util(i,visited)
                
            
# Problem : Given Teams A and B, is there a sequence of teams starting with A 
#and ending with B such taht each team in the sequence has beaten the next team in the sequence.
# Solution, model this problem as a graph: 
    # teams = vertices; source = winning and sink is losing
    #perform graph reachability eg dfs or bfs from A to B
    
#use collections.namedtuple('MatchResult',('winning_team','losing_team')) to create match results
# then create an array of these match results called matches
# use this array to construct a graph

MatchResult = collections.namedtuple('MatchResult',('winning_team','losing_team'))

def can_a_beat_b(matches,a,b):
    def build_graph():
        graph = default.dict(set)
        for match in matches:
            graph[match.winning_team].append(match.losing_team)
    return graph
    
    def is_reachable_dfs(graph,curr,dest):
        visited = set()
        return is_reachable_dfs_util(curr,dest,visited)
        
    def is_reachable_dfs_util(curr,dest,visited):
        if curr == dest:
            return True
        
        visited[curr] == True
        for i in graph[curr]:
            if visited[i] == False:
                return is_reachable_dfs_util(i,dest,visited)

#Single source shortest path of a DAG:
#        1. Initialse the dist[] : SHORTEST PATH -> INF, LONGEST PATH -> -INF and source -> 0 ALWAYS
#        2. Perform a topological sort on the graph
#        3. Process the vertices in topological sort and and for each vertex , update its adjacent vertex with the weight of the edge
#        
    dist = [float("inf")]*len(graph)
    dist[s] = 0
    while(stack):
        node = stack.pop()
        for next,weight in graph[node]:
            if dist[next] > dist[node] + weight:
                dist[next] = dist[node] + weight

#For Single Source Longest Path:
    dist = [float("-inf")]*len(graph)
    dist[s] = 0
    while(stack):
        node = stack.pop()
        for next,weight in graph[node]:
            if dist[next] < dist[node] + weight:
                dist[next] = dist[node] + weight

        
# NOTE: since we start with the source and the distance of the source is never INF, due to topological sort dist[i] will never be INF
# Also, since the node holds its distance from the source, for every consecutive vertex we need to add the dist[i](cumulative sum of all edges from the source) + weight(of the edge)


                
            
   
# Problem: Search a Maze
# Given a 2D array of black and white entries representing a maze with designated entrance 
# and exit, find a path from the entrance to the exit. white = open spaces and black = walls

# Solution : model all the white elements as the vertices sorted according to their coordinates
# Perform a DFS from entrance to exit
# NOTE: we can use BFS but we would need to explictly maintain a queue, so avoid it unless we want the shortest path
# the coordiantes are the (i,j) of the element treating as left top corner as the (0,0)


#Union by Rank and FindParent by path compression done to get a worst case O(logn) implementation
#Naive implementations take O(n)
#parents list should maintain tuples (parent,rank) for each index(which is the vertex)
#parents = [(-1,0)] for _ in range(len(self.graph))]   will initialse each vertex's parent as itself and rank as 0

def find_parent_pc(parents,v):
    if parents[v][0] == -1:
        return v
    else:
        parents[v][0] = find_parent_pc(parents,parents[v][0])
    #Here if the node is not its own parent then we recursiely find the parent of the set and then
    #SET THE NODE"S PARENT TO WHAT WE FIND AFTER RECURSION
    # eg parent of 3 is 2, parent of 2 is 1 and parent of 1 is 1 => set is represented by 1 but the set is 3->2->1
    # we find the set representation of 3 as 1 and then set parent of 3 as 1 so that we have 3->1<-2

#Always attach the shorter sets to longer sets
def union_rank(parents,x,y):
    p1 = find_parent_pc(parents,x)
    p2 = find_parent_pc(parents,y)
    if parents[p1][1] < parents[p2][1]:
        parents[p1][0] = p2   #set parent of p1 to p2
        
    elif parents[p1][1] > parents[p2][1]:
        parents[p2][0] = p1
    
    else: #if the ranks are same, set any one as parent and increase its rank
        parents[p1][0] = p2   #set parent of p1 to p2
        parents[p2][1] = parents[p2][1] + 1
        

#NOTE:
    #1. For path compression and union by ranking each vertex needs to store (rank, parent) information with it:
        # these have to be mutable as rank and parents will change , so do NOT use namedtuples as they are immutable objects
    
    #2. to create a list of lists:
        # l = [[]]*3 will create 3 references to the same list, hence on l[0].append(1) the result will be [[1],[1],[1]]
        # Use l = [[] for _ in range(3)]
        
    #3. list.sort() returns None as it sorts the original list in place:
        # Use sorted(list) to assign it to a new or temporary array

#Application of DFS
#1.Topological Sort    -> Normal DFS with a stack that pushes the leaf node.
#2. Cycle in a directed graph
#3. Count number of forests in a graph
#4. Shortest Path in Directed Acyclic Graph using topological sort




# Applications of BFS
# 1) Shortest Path and Minimum Spanning Tree for unweighted graph/in a matrix

        
        
####################

# Q A group of two or more people wants to meet and minimize the total travel distance. You are given a 2D grid of values 0 or 1, where each 1 marks the home of someone in the group.
# The distance is calculated using Manhattan Distance, where distance(p1, p2) = |p2.x - p1.x| + |p2.y - p1.y|.
#For example, given three people living at (0,0), (0,4), and (2,2):
# 1 - 0 - 0 - 0 - 1
# |   |   |   |   |
# 0 - 0 - 0 - 0 - 0
# |   |   |   |   |
# 0 - 0 - 1 - 0 - 0
# The point (0,2) is an ideal meeting point, as the total travel distance of 2+2+2=6 is minimal. So return 6.

# Approach 1: Carry out BFS from every house(1) and add up the distances; then choose the (0) with least sum: O(n^2)

# Approach 2: Sorting (accepted)
# consider 1D problem, The point which is equidistant from either end of the row (that is the median) is the optimal point for meeting.
# We will treat the 2D problem as two independent 1D problems that is find median of the sorted row co-ordinates and sorted column co-ordinates

def minDistance1D(points, origin):
    # points is array of ints since it is in 1D , co-ordinates
    distance = 0
    for point in points:
        distance += abs(point - origin)
    return distance

def minDistance2D(grid2D):
    # we will use these to collect all the positions of 1s
    rows = []
    cols = []

    for i in range(len(grid2D)):
        for j in range(len(grid2D[0])):
            if grid2D[i][j] == 1:
                rows.append(i)
                cols.append(j)

    # After collection of the x coordinates in rows (which by the nature of collection are in sorted order) and y in cols(need to be sorted,
    # we take the median of the two as the origin

    row = rows[len(rows//2)]
    cols.sort()
    col = cols[len(cols//2)]

    # the point they should meet on is the median of the rows and columns
    meetPoint = (row, col)

    dist = (minDistance1D(rows, row) + minDistance1D(cols, col))
    return dist

#Q You are given a m x n 2D grid initialized with these three possible values.

# -1 - A wall or an obstacle.
# 0 - A gate.
# INF - Infinity means an empty room. We use the value 231 - 1 = 2147483647 to represent INF as you may assume that the distance to a gate is less than 2147483647.
# Fill each empty room with the distance to its nearest gate. If it is impossible to reach a gate, it should be filled with INF.
#
# For example, given the 2D grid:
# INF  -1  0  INF
# INF INF INF  -1
# INF  -1 INF  -1
#   0  -1 INF INF
# After running your function, the 2D grid should be:
#   3  -1   0   1
#   2   2   1  -1
#   1  -1   2  -1
#   0  -1   3   4

# Approach 1: Carry out a bfs from every empty room to the gate and report the minimum value of the gate O(n^2)

#Approach 2: Collect all the gates and simultaneously carry out a bfs from them. O(n)

# Assuming the 2D graph is called matrix

def wallsAndGates(matrix):
    directions = [(1,0), (-1,0), (0,1), (-1,0)]   #the permitted directions we can move from a point (row(x-axis), col(y-axis))
    EMPTY = float("INF")
    GATE = 0
    WALL = -1
    bfs(matrix, directions)

def bfs(matrix, directions):
    rows = len(matrix)
    cols = len(matrix[0])
    q = []
    #Collect all the gates and put them into a queue
    for row in rows:
        for col in cols:
            if matrix[row][col] == GATE:
                q.append((row, col))
    #loop to carry out bfs from gates
    while len(q) != 0:
        e = q.pop()
        x = e[0]
        y = e[1]

        for d in directions:
            r = x + d[0]
            c = y + d[1]
            #if the traversing takes us to non-empty or out of bounds then try again with another direction
            if r<0 || c<0 || r>=rows || c>=cols || matrix[r][c] != EMPTY:
                continue
            #if we are within bounds and land on an empty space record the distance from the point from where we moved
            matrix[r][c] = matrix[x][y] + 1
            #add the empty space to the queue to continue the bfs
            q.append((r,c))

