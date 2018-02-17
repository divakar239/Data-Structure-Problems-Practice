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
#o detect a back edge, we can keep track of vertices currently in recursion stack of 
#function for DFS traversal. If we reach a vertex that is already in the recursion stack, 
#then there is a cycle in the tree. The edge that connects current vertex to 
#the vertex in the recursion stack is back edge. 
#We have used recStack[] array to keep track of vertices in the recursion stack.
    def detect_cycle(self):
        visted = [False]*self.V
        rec_stack = [False]*self.V   # keeps track of all vertices in the stack and helps detect cycle
        for v in V:
            if visited[v] == False:
                return self.detect_cycle_util(v,visited,rec_satck)
            
    def detect_cycle_util(v,visited,rec_stack):
        visited[v] = True
        rec_stack[v] = True
        
        for node in self.graph[v]:
            if visited[node] == False:
                if detect_cycle_util(node,visited,rec_stack):
                    return True 
            elif rec_stack[node] == True:
                    return True
        rec_stack[node] == False
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
                return True    #back edge detected as the parent and the child have the same parent
            self.union(i,j)    # else put them both in the same set since they constitute an edge
        
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
#and ending with Bsuch taht each team in the sequence has beaten the next team in the sequence.
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
#        1. Initialse the dist[] for every vertex as INF and the dist[source] = 0
#        2. Perform a topological sort on the graph
#        3. Process the vertices in topological sort and and for each vertex , update its adjacent vertex with the weight of the edge
#        4. 
#            while(stack):  where stack is topologically sorted
#                i = stack.pop()
#                for node, weight in graph[i]:
    #            if dist[node] > dist[i] + weight:
    #                dist[node] = weight
    
# NOTE: since we start with the source and the distance of the source is never INF, due to topological sort dist[i] will never be INF
# Also, since the node holds its distance from the source, for every consecutive vertex we need to add the dist[i](cumulative sum of all edges from the source) + weight(of the edge)


                
            
   
# Problem: Search a Maze
# Given a 2D array of black and white entries representing a maze with designated entrance 
# and exit, find a path from the entrance to the exit. white = open spaces and black = walls

# Solution : model all the white elements as the vertices sorted according to their coordinates
# Perform a DFS from entrance to exit
# NOTE: we can use BFS but we would need to explictly maintain a queue, so avoid it unless we want the shortest path
# the coordiantes are the (i,j) of the element treating as left top corner as the (0,0)


#Union by Rank and FindParent by path compression
#Naive implementations take log(n)
#parents list should maintain tuples (parent,rank) for each index(which is the vertex)
#parents = [(-1,0)]*len(self.graph)   will initialse each vertex's parent as itself and rank as 0

def find_parent_pc(parents,v):
    if parents[v][0] == -1:
        return v
    else:
        parents[v][0] = find_parent_pc(parents,parents[v][0])
    #Here if the node is not its own parent then we recursiely find the parent of the set and then
    #SET THE NODE"S PARENT TO WHAT WE FIND AFTER RECURSION
    # eg parent of 3 is 2, parent of 2 is 1 and parent of 1 is 1 => set is represented by 1 but the set is 3->2->1
    # we find the set representation of 3 as 1 and then set parent of 3 as 1 so that we have 3->1<-2
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
        


        





#Application of DFS
#1.Topological Sort    -> Normal DFS with a stack that pushes the leaf node.
#2. Cycle in a directed graph
#3. Count number of forests in a graph
#4. Shortest Path in Directed Acyclic Graph using topological sort




# Applications of BFS
# 1) Shortest Path and Minimum Spanning Tree for unweighted graph/in a matrix

        
        
        
        