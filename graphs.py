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
        graph = collections.defaultdict(set)
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
            dist[next] = min(dist[next], dist[next] + weight)
#For Single Source Longest Path:
    dist = [float("-inf")]*len(graph)
    dist[s] = 0
    while(stack):
        node = stack.pop()
        for next,weight in graph[node]:
            dist[next] = max(dist[next], dist[node] + weight)

        
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
    return (dist, meetPoint)

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
    visited = collections.defaultdict(bool)
    EMPTY = float("INF")
    GATE = 0
    WALL = -1
    bfs(matrix, directions, visited)

def bfs(matrix, directions, visited):
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
        visited[e] = True
        for d in directions:
            r = x + d[0]
            c = y + d[1]
            #if the traversing takes us to non-empty or out of bounds then try again with another direction
            if r<0 || c<0 || r>=rows || c>=cols || matrix[r][c] != EMPTY || visited[(r,c)] is True:
                continue
            #if we are within bounds and land on an empty space record the distance from the point from where we moved
            matrix[r][c] = matrix[x][y] + 1
            #add the empty space to the queue to continue the bfs
            q.append((r,c))


# Q. Given a non-empty 2D array grid of 0's and 1's, an island is a group of 1's (representing land) connected 4-directionally (horizontal or vertical.)
# You may assume all four edges of the grid are surrounded by water.
# Count the number of distinct islands.
# An island is considered to be the same as another if they have the same shape,
# or have the same shape after rotation (90, 180, or 270 degrees only) or reflection (left/right direction or up/down direction).
#
# Example 1:
# 11000
# 10000
# 00001
# 00011
# Given the above grid map, return 1.
#
# Notice that:
# 11
# 1
# and
#  1
# 11
# are considered same island shapes. Because if we make a 180 degrees clockwise rotation on the first island, then two islands will have the same shapes.
# Example 2:
# 11100
# 10001
# 01001
# 01110
# Given the above grid map, return 2.
#
# Here are the two distinct islands:
# 111
# 1
# and
# 1
# 1
#
# Notice that:
# 111
# 1
# and
# 1
# 111
# are considered same island shapes. Because if we flip the first array in the up/down direction, then they have the same shapes.

# Approach : Canonical hash
# 1. carry out dfs in the matrix from 1s
# use complex numbers in python for easy transformations
# 8 possibble transformations for every point
# For each of 8 possible rotations and reflections of the shape, we will perform the transformation and then translate the shape
# so that the bottom-left-most coordinate is (0, 0). Afterwards, we will consider the canonical hash of the shape to be
# the maximum of these 8 intermediate hashes.

# Intuition
# We determine local coordinates for each island.
# Afterwards, we will rotate and reflect the coordinates about the origin and translate the shape so that the bottom-left-most coordinate is (0, 0). At the end, the smallest of these lists coordinates will be the canonical representation of the shape.
# Algorithm
# We feature two different implementations, but the core idea is the same. We start with the code from the previous problem, Number of Distinct Islands.
# For each of 8 possible rotations and reflections of the shape, we will perform the transformation and then translate the shape so that the bottom-left-most coordinate is (0, 0).
# Afterwards, we will consider the canonical hash of the shape to be the maximum of these 8 intermediate hashes.
# In Python, the motivation to use complex numbers is that rotation by 90 degrees is the same as multiplying by the imaginary unit, 1j.
# In Java, we manipulate the coordinates directly. The 8 rotations and reflections of each point are (x, y), (-x, y), (x, -y), (-x, -y), (y, x), (-y, x), (y, -x), (-y, -x)

def numIslands(grid):
    visited = set() # keep track of visited vertices
    shapes = set()  # final set to keep track of distinct island shapes

    for row in len(grid):
        for col in len(grid[0]):
            shape = set()
            dfs(grid, visited, shape, row, col)
            if shape:
                shapes.add(canonical(shape))
    return len(shapes)

def dfs(grid, visited, shape, r, c):
    if 0<=r<len(grid) and 0<=c<len(grid[0]) and grid[r][c] == 1 and (r,c) not in visited:
        visited.add((r,c))
        shape.add(complex(r,c))

        #recursion on neighoring points
        dfs(grid, visited, shape, r+1, c)
        dfs(grid, visited, shape, r, c+1)
        dfs(grid, visited, shape, r-1, c)
        dfs(grid, visited, shape, r, c-1)

def canonical(shape):
    ans = None
    for k in range(4):
        # k represents the number of rotation i.e. by being applied as an exponent to 1j
        ans = max(ans, translate([z * (1j) ** k for z in shape]))  #the argument of translate causes the rotation on the original points
        ans = max(ans, translate([complex(z.imag, z.real) * (1j) ** k for z in shape])) # the argument of translate flips the original points and then causes the rotation
    return tuple(ans) #has all the points of either z or complex(z.imaginary, z.real) as a tuple eg (1,2,3,...). So, whichever is max is returned as ans


def translate(shape):
    # All this function does is subtract the lowest co-ordinate from all co-ordinates to move the entire shape to center by making the lowest co-ordinate (0,0)
    w = complex(min(z.real for z in shape), min(z.imag for z in shape)) # make the lowest bottom point as (0,0)
    return sorted(str(z-w) for z in shape)

# Note the above solution (without the canonical function) can be use dto find distinct number of islands where translation
# means that islands are same


# Q. Given n processes, each process has a unique PID (process id) and its PPID (parent process id).

# Each process only has one parent process, but may have one or more children processes. This is just like a tree structure. Only one process has PPID that is 0,
# which means this process has no parent process. All the PIDs will be distinct positive integers.
# We use two list of integers to represent a list of processes, where the first list contains PID for each process and
# the second list contains the corresponding PPID.
#
# Now given the two lists, and a PID representing a process you want to kill, return a list of PIDs of processes
# that will be killed in the end. You should assume that when a process is killed, all its children processes will be killed. No order is required for the final answer.
#
# Example 1:
# Input:
# pid =  [1, 3, 10, 5]
# ppid = [3, 0, 5, 3]
# kill = 5
# Output: [5,10]
# Explanation:
#            3
#          /   \
#         1     5
#              /
#             10
# Kill 5 will also kill 10.


# Approach: Create a directed graph as an adjacency list and print out the list of the node in the kill list

def killPID(pid, ppid, kill):
    # Create a directed graph
    graph = collections.defaultdict(list)
    for i in range(len(ppid)):
        graph[ppid[i]].append(graph(pid))

    # Get the list of the element to be killed
    return graph[kill].append(kill)

# Q. Suppose you are at a party with n people (labeled from 0 to n - 1) and among them, there may exist one celebrity. The definition of a celebrity is that all the other n - 1 people know him/her but he/she does not know any of them.
# Now you want to find out who the celebrity is or verify that there is not one.
# The only thing you are allowed to do is to ask questions like: "Hi, A. Do you know B?"
# to get information of whether A knows B. You need to find out the celebrity (or verify there is not one)
# by asking as few questions as possible (in the asymptotic sense).
# You are given a helper function bool knows(a, b) which tells you whether A knows B.
# Implement a function int findCelebrity(n), your function should minimize the number of calls to knows.
# Note: There will be exactly one celebrity if he/she is in the party. Return the celebrity's label if there is a celebrity in the party. If there is no celebrity, return -1.

# Approach 1: O(n^2) graph
# 1. There are (n choose 2) pairs so we will have to usw the 'knows' question that many times ~ O(n^2)
# 2. Keep track of count for incoming and outgoing edges for each node
# 3. If A knows B, increment A's outgoing by 1 and B's incoming by 1
# 4. Check all nodes to find the node with 0 outgoing count

# Approach 2: Use Stack O(n)

# Idea
# If A knows B, then A can’t be celebrity. Discard A, and B may be celebrity.
# If A doesn’t know B, then B can’t be celebrity. Discard B, and A may be celebrity.
# Repeat above two steps till we left with only one person.
# Ensure the remained person is celebrity.

# Steps
# Push all the people into a stack.
# Pop off top two persons from the stack, discard one person based on return status of HaveAcquaintance(A, B).
# Push the remained person onto stack.
# Repeat step 2 and 3 until only one person remains in the stack.
# Check the remained person in stack doesn’t have acquaintance with anyone else.

def findCelebrity(n):
    # declare stack and push all people in it
    stack = []
    for i in range(n):
        stack.append(i)

    person1 = stack.pop()
    person2 = stack.pop()

    while len(stack) > 1:
        if knows(person1, person2):
            # person1 is not the celebrity
            stack.append(person2)
        else:
            # person2 is not celebrity
            stack.append(person1)

    # Check if the remaining person in the stack is indeed celebrity by checking if it knows anyone
    e = stack.pop()
    for i in range(n):
        if i!=e and (knows(e,i) or not knows(i,e)):
            return -1
    return e


# Given a m x n rectangle, how many squares are there in it?

# Examples :
# Input:  m = 2, n = 2
# Output: 5
# There are 4 squares of size 1x1 + 
#           1 square of size 2x2.

# Input: m = 4, n = 3
# Output: 20
# There are 12 squares of size 1x1 + 
#           6 squares of size 2x2 + 
#           2 squares of size 3x3.


# Idea: formula to compute number of squares in mxn matrix is:
# Let us first solve this problem for m = n, i.e., for a square:

# For m = n = 1, output: 1

# For m = n = 2, output: 4 + 1 [4 of size 1×1 + 1 of size 2×2]

# For m = n = 3, output: 9 + 4 + 1 [4 of size 1×1 + 4 of size 2×2 + 1 of size 3×3]

# For m = n = 4, output 16 + 9 + 4 + 1 [16 of size 1×1 + 9 of size 2×2 + 4 of size 3×3 + 1 of size 4×4]

# In general, it seems to be n^2 + (n-1)^2 + … 1 = n(n+1)(2n+1)/6

# Let us solve this problem when m may not be equal to n:

# Let us assume that m <= n

# From above explanation, we know that number of squares in a m x m matrix is m(m+1)(2m+1)/6

# What happens when we add a column, i.e., what is the number of squares in m x (m+1) matrix?
# When we add a column, number of squares increased is m + (m-1) + … + 3 + 2 + 1
# [m squares of size 1×1 + (m-1) squares of size 2×2 + … + 1 square of size m x m]

# Which is equal to m(m+1)/2

# So when we add (n-m) columns, total number of squares increased is (n-m)*m(m+1)/2.

# So total number of squares is m(m+1)(2m+1)/6 + (n-m)*m(m+1)/2.

# Using same logic we can prove when n <= m.

# So, in general,

# Total number of squares = m x (m+1) x (2m+1)/6 + 
#                           (n-m) x m x (m+1)/2 

# when n is larger dimension
# .

# Using above logic for rectangle, we can also prove that number of squares in a square is n(n+1)(2n+1)/6

def countSquares(matrix):
    rows = len(matrix)
    cols = len(matrix[0])

    # If rows < cols, swap them
    if rows<cols:
        rows, cols = cols, rows
    
    numerator = rows*(rows+1)*(2*rows+1)
    denominator = 6 + (cols - rows)*rows*(rows+1)/2
    num_squares = numerator/denominator
        


# Q. There are soldiers and civilians arranged in N x M matrix way, find out the 'K' weak rows in the matrix.
# Weak rows are those where numbers of soldiers are less compare to other siblings row.
# Soldiers are always stand in frontier, means always 1's may appear first and then 0's
# 1 represents soldier
# 0 represents civilian

# ex:
# K = 2
# matrix = [
# [1, 1, 1, 0, 0, 0]
# [1, 1, 0, 0, 0, 0]
# [1, 1, 1, 1, 0, 0]
# [1, 1, 0, 0, 0, 0]
# ]
# here row 1 & 3 are weak rows since they have less numbers of 1's compare to row 0 & 2

# discussed about the approach & also time & space complexity analysis

# Solution:
# Idea: We will maintain a min heap of tuples (index pf last 1 , array)
# We use binary search to find the last index of 1 in each row
# We then pop the first K elements out of the heap

# time -- O(n * log(mn) + k*logm)
# space -- O(m)

def helper(arr, k):
    def get1(arr):
        l, r = 0, len(arr)
        while l < r:
            m = (l+r)/2
            if arr[m] == 0:
                r = m
            else:
                l = m+1
        return l
    q = []
    for i in range(len(arr)):
        num = get1(arr[i])  #log(n)
        heapq.heappush(q,(num,i)) #log(m)
        # n*log(mn)
    res = []
    while k>0 and q:
        res.append(heapq.heappop(q)[1]) #logm
        k-=1
    return res #n * log(mn) + k*logm

arr = [
[1, 1, 1, 0, 0, 0],
[1, 1, 0, 0, 0, 0],
[1, 1, 1, 1, 0, 0],
[1, 1, 0, 0, 0, 0]
]
k = 2
print(helper(arr, k))
#time  --  O(n * log(mn) + k*logm)
#space -- O(m)


# Q. Elevator has two buttons Up and Down , By pressing up elevator goes up by p floors and by pressing down it goes down by q floors. A building has n floors. Given a starting floor s, Can you explain if it's possible to go to floor e.
# A. 
def canReach(s, e, p, q, n):
    visited = set()
    while s != e:
        if s in visited:
            return False
        else:
            visited.add(s)
            if s < e:
                if s + p <= n:
                    s += p
                elif s - q >= 1:
                    s -= q
                else:
                    False
            else:
                if s - q >= 1:
                    s -= q
                elif s + p <= n:
                    s += p
                else:
                    False

    return True
