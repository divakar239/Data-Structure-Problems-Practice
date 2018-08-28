#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 18 21:44:11 2018

@author: DK
"""
class bst_node:
    def __init__(self,data):
        self.data=data
        self.right=None
        self.left=None
        
class bst:
    def __init__(self):
        self.root=None
    
    def insert(self,data):
        if self.root is None:
            self.root=bst_node(data)
        else:
            self.insert_node(self.root,data)
    
    def insert_node(self,root,data):
        if data<=root.data:
            if root.left is not None:
                self.insert_node(root.left,data)
            else:
                root.left=bst_node(data)
                return
        else:
            if root.right is not None:
                self.insert_node(root.right,data)
            else:
                root.right=bst_node(data)
                return

# Q2. input BST and num K, and returns the k largest keys (code above)
# Approach 1: Inorder traversal and print the last k elelments. However, this can prove to be very ineffective especially if the length of the left subtree is really large
# Approach 2: Reverse inorder but stop as soon as the kth element is hit.
    def reverse_inorder_traversal(self,num):
        count = 0
        if self.root == None:
            return
        else:
            return self.reverse_inorder(self.root,num,count)
               
    def reverse_inorder(self,root,k,count):
        count += 1
        if root == None:
            return
        if root.right != None:
            self.reverse_inorder(root.right,k)
        if count>=k:
            return
        print(root.data)
        if root.left != None:
            self.reverse_inorder(root.left,k)
   
#Inorder : iteration
    def inorder_iteration(self):
        if self.root == None:
            return None
        
        stack = []
        curr_node = self.root
        end = False
        while(not end):
            
            if curr_node != None:
                stack.append(curr_node)
                curr_node = curr_node.left
            else:
                if len(s) > 0:
                    e = stack.pop()
                    print(e.data)
                    if e.right != None:
                        curr_node = e.right
                else:
                    end = True  

#Preorder : iteration
    def preorder_iteration(self):
        if self.root == None:
            return
        stack = []
        curr_node = self.root
        stack.append(curr_node)
        
        while(len(stack) != 0):
            curr_node = stack.pop()
            print(curr_node.data)
            if curr_node.right != None:
                stack.append(curr_node.right)
            if curr_node.left != None:
                stack.append(curr_node.left)
#Postorder : iteration
    
        
                                                                                                                                                                                                                                                                          
            
# Q3. Print the Lowest common ancestor of two given nodes:
# Approach 1: Carry out a post order traversal, the LCA is the first node traversed after the two given nodes only if the two nodes are present in the same subtree O(n)

# Approach 2: This will work if the values stored in the BST are different
# Idea: if the values of the 2 nodes are smaller than the root then their LCA lies in the left subtree. Similarly, LCA is in right subtree 
# step 1: if root's value is same as either of the node then root is LCA
# step 2: if value of 1 node is less and other is greater than the root, LCA is the root
# step 3: if both are smaller, LCA is in the left subtree
# step 4: if both are smaller, LCA is in the right subtree
    def find_LCA(self,a,b):
        if self.root == None:
            return
        else:
            return self.find_LCA_util(self.root,a,b)
            
    def find_LCA_util(self,root,a,b):
        #assumption that a.data<=b.data
        if root.data == a.data or root.data == b.data:
            return root
        if root.data>a.data and root.data<b.data:
            return root
        if root.data>a.data and root.data>b.data:
            return self.find_LCA_util(root.left,a,b)
        if root.data<a.data and root.data<b.data:
            return self.find_LCA_util(root.right,a,b)
            
# Find height of a binary tree / also find balance factor that is the difference in height of the left and right subtree
    def height(self,root):
        if root == None:
            return 0
        lheight = self.height(root.left)
        rheight = self.height(root.right)
        
        return max(lheight,rheight) + 1      # accumulates the height of the tree

    def balance(self,root):
        if root == None:
            return 0
        l_h = height(root.left)
        r_h = height(root.right)
        
        return abs(l_h - r_h)
    

    
            
#####Recontruction of trees #####
## To reconstruct a binary tree we need:
    # 1. Inorder Traversal
    # 2. either post order or pre order traversal
###Side question: Construct a binary tree given a) inorder b) preorder traversal
#Note: all traversals carried out by recursion have T(n) = O(n) and S(n) = O(h) where h is the height of the tree
#Approach: The first element of the preorder is the root of the tree. In the inorder sequence every element to the left of this root belongs to the left subtree and all to the right belong to the right subtree.
# so, we can solve this by recursion where we keep taking an element from the preorder sequence to find the respective subtrees for each element.

# example: inorder D B E A F C
#          Preorder sequence: A B D E C F   : here A is the root of the tree, DBE are the left subtree. B is the root of left subtree and D is B's left subtree ...

# for post order we will do the same but start from the last element since that will be the root.
# T(n) = O(n^2) since for every element in preorder/post order we will have to search it up in inorder. Search takes O(n) for 1 element.
# we can use a hashmap to reduce complexity to O(n)

# For BST we only need the pre-order or post-order. Inorder is not required.and alone it can't give a unique BST.


#Mirror a BST:
    
    #Approach 1: Mirror left sub tree and right sub tree; then swap the subtrees
    
    def mirror(node):
        if node == None:
            return node
        else:
            ltree = mirror(node.left)
            rtree = mirror(node.right)
       node.left = rtree
       node.right = ltree
       return node
    
    #Approach 2: reverse the inorder of the tree and get the preorder of the tree; use these to reconstruct a tree

    # To check if two trees are identical, compare the inorder and preorder traversals of both trees

# Q. Check if two trees are symmetric O(n)
def checkSymmetry(root1, root2):
    if root1 is None and root2 is None:
        return True
    if root1 is None or root2 is None:
        return False
    return root1.data == root2.data and checkSymmetry(root1.left, root2.right) and checkSymmetry(root1.right, root2.left)


##### BSTs #####

# Q1. Find inorder successor and predecessor of target in BST
#1. If root is NULL
#      then return
#2. if key is found then
#    a. If its left subtree is not null
#        Then predecessor will be the right most 
#        child of left subtree or left child itself.
#    b. If its right subtree is not null
#        The successor will be the left most child 
#        of right subtree or right child itself.
#    return
#3. If key is smaller then root node
#        set the successor as root
#        search recursively into left subtree
#    else
#        set the predecessor as root
#        search recursively into right subtree
#Can we not just use the inorder traversal to give successor as the next element and prredecessor as the previous element
# Yes, we can but that is O(n): for successor we need to search the right subtree if there is one
def successor_predecessor(node,target):
    successor = None
    predecessor = None
    if node == None:
        return
    if target.data == node.data:
        if target.left != None:
            #for predecessor we want the right most leaf node of the left subtree
            tmp = target.left
            while(tmp.right != None):
                tmp = tmp.right
            predecessor = tmp
        if target.right != None:
            #for succesor we want the left most leaf of the right subtree
            tmp = tmp.right
            while(tmp.left != None):
                tmp = tmp.left
            successor = tmp
    #if the target data is less than the node then successor is the node and we search the left subtree for the predecessor
    elif target.data < node.data:
        successor = node
        successor_predecessor(node.left,target)
    #if target data is more than the node data then the predecessor is the node and we search the right subtree for the successor
    elif target.data > node.data:
        predecessor = node
        successor_predecessor(node.right,target)

# the above question in separate parts
def successor(root, target):
    node = root
    if target.right != None:
        temp = target.right
        while temp != None:
            temp = temp.left
        return temp
    else:
        if node.data > target.data:
            # When the target has no right subtree and is in the kleft subtree of the root , then the root is the successor
            return node
        if node.data < target.data:
            # If the target is the right subtree, then the parent is the successor
            return successor(node.right, target)

def predecessor(root, target):
    # everything is the opposite of successor()
    node = root
    if target.left != None:
        temp = target.left
        while temp != None:
            temp = temp.right
        return temp
    else:
        if node.data > target.data:
            return predecessor(node.left, target)
        if node.data < target.data:
            return node


#Delete a node in binary search tree
def delete_node(root,target):
    if root == None:
        return None
    if target.data < root.data:
        left = delete_node(root.left,target.right)
    elif target.data > root.data:
        right = delete_node(root.right,target)
    else:
        if root.left == None:
            temp = root.right
            root = None
            return temp
        elif root.right == None:
            temp = root.left
            root = None
            return temp
        else:
            temp = min_tree(root.right)   #find min value of the right subtree
            root.data = temp.data
            root.right = delete_node(root.right,target)
            
#Q. Given a binary search tree and a target value K. The task is to find the node with minimum absolute difference with given target value K.
def getClosestNode(root, target_val):
    if root is None:
        return None

    min_val = float("INF")
    min_node = None
    getClosestNodeUtil(root, target_val, min_val, min_node)
    return (min_val, min_node)

def getClosestNodeUtil(root, target_val, min_val, min_node):
    if root.data == target_val:
        min_val = 0
        min_node = root

    # Updating the values
    if min_val > abs(root.data - target_val):
        min_val = abs(root.data - target_val)
        min_node = root

    if target_val < root.data:
        getClosestNodeUtil(root.left, target_val, min_val, min_node)
    if target_val > root.data:
        getClosestNodeUtil(root.right, target_val, min_val, min_node)

#Q. Largest BSt subtree where largest means number of nodes

# Data structure to hold multiple values for every node
class NodeValue:
    def __init__(self):
        self.size = 0
        self.is_bst = True
        self.min = float("inf")
        self.max = float("-inf")


class LargestBST:

    def largestBST(self, root):
        max_size = largestBSTUtil(root)
        return max_size.size

    def largestBSTUtil(self, root):
        # End condition where a single node is a BST of size 0
        if root is None:
            return NodeValue()

        l = self.largestBSTUtil(root.left)
        r = self.largestBSTUtil(root.right)

        #Note: this is local not global, since every node will have its own NodeVal
        node_val = NodeValue()

        # if the subtrees are not BSTs
        if l.is_bst == False or r.is_bst == False or l.max > root.data or r.min < root.data:
            node_val.is_bst = False

        # If subtrees are BSTs , setting all the parameters of the NodeVal
        node_val.is_bst = True
        node_val.size = l+r+1

        if root.left is not None:
            node_val.min = l.min
        else:
            node_val.min = root.data

        if root.right is not None:
            node_val.max = r.max
        else:
            node_val = root.data

        return node_val

###########################################################################
def make_tree():
    t=bst()
    t.insert(108)
    t.insert(285)
    t.insert(107)
    t.insert(243)
    t.insert(286)
    t.insert(401)
    t.insert(107)
    t.insert(-10)
    return t
    
    
#Given a BINARY TREE, print all root-to-leaf paths: time complexity O(n^2), space complexity O(n)
# the time complexity is O(n + lh) where 'n' is the time to go through every node and 'h' is the length of the array that is used to store the path. 'l' is the number of times the array is printed
# if the tree is a list then l=1 and h=n => we get O(n+n) = O(n)
# if the tree is perfectly balanced then l=n/2 and h=log(n) => O(n+nlogn) = O(nlogn)

def tree_paths(node,n):
    path=[]
    return tree_paths_util(node,path,0)
    
def tree_path_util(node,arr,pathlen):
    if node == None:
        return
    arr[pathlen] = node.data
    pathlen = pathlen + 1
    if node.left==None and node.right==None:
        for i in range(pathlen):
            print(arr[i])
    else:
        tree_path_util(node.left,path,pathlen)
        tree_path_util(node.right,path,pathlen)
        

# add problem 1: given 'sum' find which root to leaf path yields it
# before printing the elements in "if node.left == None and node.right == None: add elements and compare with sum.

###############################################################################
# Sum all the root to leaf paths

def sum_paths(node):
    curr_sum = 0
    sum_paths_util(node,curr_sum)

def sum_paths_util(node,sum):
    if node == None:
        return 
    sum = sum + node.data
    if node.left == None and node.right == None:
        return sum
    else:
        l_sum = sum_paths_util(node.left,sum)
        r_sum = sum_paths_util(node.right,sum)
        return l_sum + r_sum     #this adds all the paths


# maximum sum of root to leaf path
# in the end condition of a leaf node, add the elements and compare with an int max, update max if max<sum
# Another way of solving this problem, in O(1) space: 
# Idea is to sum at every node till we hit a child, compare it with max and if mx<sum then store the leaf node
def tree_max_sum_constant_space(node,n):
    if node == None:
        return 0
    curr_sum = 0
    max_val = float("-inf")                        #used to save the max sum of the paths
    target_leaf = None                 #will record the leaf node of the path with maximum sum
    tree_max_sum_constant_space_util(node,curr_sum,max_val)
    
def tree_max_sum_constant_space_util(node,curr_sum,max_val):
    if node == None:
        return
    curr_sum = curr_sum + node.data
    if node.left == None and node.right == None:
        if max_val < curr_sum:
            max_val = curr_sum
            target_leaf = node
    else:
        tree_max_sum_constant_space_util(node.left,curr_sum,max_val)
        tree_max_sum_constant_space_util(node.right,curr_sum,max_val)

#This will print the path from the root to the leaf node. NOTE: this works on any binary tree, not just BSTs.
#If a tree is not a BST we can't use the comparison property to print the ancestors i.e. path till the leaf node
def print_path(node,target_leaf):
    if node == None:
        return False
    # if 1. target_leaf is the node then print OR 2. target_leaf is present in left descendent tree OR target_leaf is present in right descendent tree
    elif target_leaf == node || print_path(node.left, target_leaf) || print_path(node.right, target_leaf):
        print(node.data)
    

# add problem 2: given 'sum' find all paths that add upto 'sum', paths don't need to start at roots and end at leaves. but path should go downwards
# before printing run method: that is find the continuous subarray in the path that adds upto sum; since the path is listed from parent tp child, it satisfies the go downwards property
def cont_subarray_neg_sum_v2(array,sum):
storage = collections.defaultdict(int)
#index = 0
curr_sum = 0
for i in range (len(array)):
    curr_sum = curr_sum + array[i]
#The if statement below is true when the subarray starting from 0 to i adds up to sum
    if(curr_sum == sum):
        for k in range(0, i+1):
            print(array[k])
        return
#The if statement below is true when subaaray from (curr_sum - sum) to i adds up to sum
    elif(storage[curr_sum - sum] == 1):
        for k in range (curr_sum-sum, i+1):
            print(array[k])
        return
    storage[array[i]] = 1
    #index = i
print("No subarray found")
        

# Problem: Given a binary tree, find the maximum path sum. The path may start and end at any node in the tree.
# Approach:Starting from a node there are 4 ways a path can go :
    #1. Node
    #2. max path through left subtree + node
    #3. max path through right subtree + node
    #4. max of left subtree + node + max right sub tree
    
def find_max_sum(node):
    res = float("-inf")
    find_max_sum_util(node, res)
    return res
    
def find_max_sum_util(node,res):
    if node == None:
        return 0
    l_val = find_max_util(node.left,res)    #sum of the left tree
    r_val = find_max_util(node.right,res)   #sum of the right tree
    
    max_single_path = max(max(l_val,r_val) + node.data, node.data)
    max_bw_single_and_leftToright = max(max_single_path, l_val + r_val + node.data)
    
    res = max(max_single_path, max_bw_single_and_leftToright) #final max value
    
    return max_single_path

# Print all ancestors/parents of a target node in a binary tree:
def ancestors(node,target):
    if node == None:
        return False
    if target.data == node.data:
        return True
    if ancestors(node.left,target) or ancestors(node.right,target):
        print(node.data)
        return True
    return False
        
# Check if a Binary Tree is balanced:
    #1. check if the balance factor is <= 1
    #2. check if every left subtree is balanced
    #3. check if every right subtree is balanced
    
def is_balanced(node):
    #return true as an empty tree is balanced
    if node == None:
        return True
    # get the heights of the subtrees
    ltree = height(node.left)
    rtree = height(node.right)
    
    if abs(rtree - ltree)<=1 and is_balanced(node.left) and is_balanced(node.right):
        return True
    return False
    
def height(node):
    if node == None:
        return -1
    l_h = height(node.left)
    r_h = height(node.right)
    
    return max(l_h,r_h) + 1


#another way of checking if a tree is balanced; this method doesn't check the right subtree if the left subtree is unbalanced, hence it makes less comparisons
#alternate questions on this theme:
    #1.write a program to return the largest size of subtree that is balanced
    #2.define K as the balance factor and find a node in the tree such that it is not K balanced but all of its descendents are
    
#the named tuple allows us to hold two fields for a node, the balance property and the height of the node
balanced_status_with_height = collections.namedtuple('balanced_status_with_height',('balance_check','height'))
def check_balanced(node):
    if node == None:
        return balanced_status_with_height(True,-1)   #for empty tree, balance = True and height is -1
    
    left_bool = check_balanced(node.left)             #returns a tuple
    if left_bool.balance_check == False:
        return balanced_status_with_height(False,0)
    
    right_bool = check_balanced(node.right)
    if right_bool.balance_check == False:
        return balanced_status_with_height(False,0)
        
    is_balanced = (abs(left_bool.height - right_bool.height) <= 1)
    height = max(left_bool.height,right_bool.height) + 1
    return balanced_status_with_height(is_balanced,height)    # this tuple is returned for every node with the values 1. balance 2.height
    
# Find LCA of two nodes in a binary tree

def LCA(root,x,y):
    if root == None:
        return None
    #if any is equal to the root, then LCA is root
    if x.data == root.data or y.data == root.data:
        return root
        
    l = LCA(root.left,x,y)
    r = LCA(root.right,x,y)
    
    # the idea is that if both l and r are non-null then the root is the LCA as both are in separate subtrees
    if l and r:
        return root
     
    #otherwise return whichever subtree they are both in
    elif l:
        return l
    elif r:
        return r
    

#Q Invert a binary tree:

def invert(node):
    if node == None:
        return None
    #Recursively call on left subtree and set the right pointer to the parent
    if node.left != None:
        l_node = invert(node.left)
        l_node.right = node
    if node.right != None:
        r_node = invert(node.right)
        r_node.left = node
    node.left = None
    node.right = None
    return node


#Given a binary tree, write a program to count the number of Single Valued Subtrees.
# A Single Valued Subtree is one in which all the nodes have same value. 
#Expected time complexity is O(n).

def count_singly(node):
    count = 0
    count_singly_util(node,count)
    return count
    
def count_singly_util(node,count):
    if node == None:
        return True          #every leaf node is a singly so return true
    
    #recursively check for all left and right singly subtrees
    lt = count_singly_util(node.left,count)
    rt = count_singly_util(node.right,count)
    
    #if any of the subtrees is not singly then return false
    if lt == False or rt == False:
        return False
    
    #if both subtrees are singly, then check if the value of the root and the children are different -> false
    if node.data != node.left.data:
        return False
        
    if node.data != node.right.data:
        return False
    
    #otherwise increment count and return true
    count = count + 1
    return True
    

#Q. Given a binary tree, find the length of the longest consecutive sequence path.
# The path refers to any sequence of nodes from some starting node to any node in the tree along the parent-child connections. The longest consecutive path need to be from parent to child (cannot be the reverse).

# For example,
#    1
#     \
#      3
#     / \
#    2   4
#         \
#          5
# Longest consecutive sequence path is 3-4-5, so return 3.
#    2
#     \
#      3
#     /
#    2
#   /
#  1
# Longest consecutive sequence path is 2-3,not3-2-1, so return 2.


# Approach 1: bottom-up like post order

def longestConsecutive(root):
    max = 0

    dfs(root, max)
    return max

def dfs(root, max):
    #end condition
    if root is None:
        return 0
    #traverse the children
    l = dfs(root.left, max) + 1
    r = dfs(root.right, max) + 1

    #rewrite  children heights to 1 if their value is not the continuous successor of the root
    if l.data != root.data + 1:
        l = 1
    if r.data != root.data + 1:
        r = 1

    #consider max of the children
    length = max(l, r) + 1
    if max<length:
        max = length

# Q. Boundary of Binary Tree
# Given a binary tree, return the values of its boundary in anti-clockwise direction starting from root.
# Boundary includes left boundary, leaves, and right boundary in order without duplicate nodes.
# Left boundary is defined as the path from root to the left-most node.
# Right boundary is defined as the path from root to the right-most node.
# If the root doesn't have left subtree or right subtree, then the root itself is left boundary or right boundary.
# Note this definition only applies to the input binary tree, and not applies to any subtrees.
# The left-most node is defined as a leaf node you could reach when you always firstly travel to the left subtree if exists. If not, travel to the right subtree. Repeat until you reach a leaf node.
# The right-most node is also defined by the same way with left and right exchanged.


# Approach1: divide the problem into three sub problems O(n)
# 1. collect the left path (iteration)
# 2. collect the leaves (recursion)
# 3. collect the right path (iteration)

# Approach2: Using pre-order traversal
# In order to distinguish between the various kinds of nodes, we make use of a as follows:
# Flag=0: Root Node.
# Flag=1: Left Boundary Node.
# Flag=2: Right Boundary Node.
# Flag=3: Others(Middle Node).
#
# We make use of three lists , , to store the appropriate nodes and append the three lists at the end.
# We go for the normal preorder traversal, but while calling the recursive function for preorder traversal using the left child
# or the right child of the current node, we also pass the information indicating the type of node that the current child behaves like.

# For obtaining the flag information about the left child of the current node, we make use of the function leftChildFlag(node, flag). In the case of a left child, the following cases are possible, as can be verified by looking at the figure above:
#
# The current node is a left boundary node: In this case, the left child will always be a left boundary node. e.g. relationship between E & J in the above figure.
#
# The current node is a root node: In this case, the left child will always be a left boundary node. e.g. relationship between A & B in the above figure.
#
# The current node is a right boundary node: In this case, if the right child of the current node doesn't exist, the left child always acts as the right boundary node. e.g. G & N. But, if the right child exists, the left child always acts as the middle node. e.g. C & F.
#
# Similarly, for obtaining the flag information about the right child of the current node, we make use of the function rightChildFlag(node, flag). In the case of a right child, the following cases are possible, as can be verified by looking at the figure above:
#
# The current node is a right boundary node: In this case, the right child will always be a right boundary node. e.g. relationship between C & G in the above figure.
#
# The current node is a root node: In this case, the right child will always be a left boundary node. e.g. relationship between A & C in the above figure.
#
# The current node is a left boundary node: In this case, if the left child of the current node doesn't exist, the right child always acts as the left boundary node. e.g. B & E. But, if the left child exists, the left child always acts as the middle node.


def boundary(root):
    left_boundary = []
    right_boundary = []
    leaves = []

    preorder(root, left_boundary, right_boundary, leaves, 0)

    for leaf in leaves:
        left_boundary.append(leaf)
    for node in reversed(right_boundary):
        left_boundary.append(node)

# to check flag information of a node
def isLB(flag):
    return(flag == 1)

def isRB(flag):
    return(flag == 2)

def isL(node):
    return(node.left is None and node.right is None)

def isRoot(flag):
    return(flag == 0)

# obtain the flag information of the left and right nodes of the current node
def isLeftChildFlag(node, flag):
    if isLB(flag) || isRoot(flag):
        return 1
    elif isRB(flag) and node.right is None:
        return 2
    else:
        return 3

def isRightChildFlag(node, flag):
    if isRB(flag) || isRoot(flag):
        return 2
    elif isLB(flag) and node.right is None:
        return 1
    else:
        return 3


def preorder(root, lb, rb, lfs, flag):
    if root is None:
        return
    if isRB(flag):
        rb.append(root)
    if isLB(flag):
        lb.append(root)
    if isL(root):
        lfs.append(root)

    preorder(root.left, lb, rb, lfs, isLeftChildFlag(root, flag))
    preorder(root.right, lb,rb, lfs, isRightChildFlag(root, flag))

# Q. Given a binary tree with n nodes, your task is to check if it's possible to partition the tree to two trees
# which have the equal sum of values after removing exactly one edge on the original tree.
# Input:
#     5
#    / \
#   10 10
#     /  \
#    2   3
#
# Output: True
# Explanation:
#     5
#    /
#   10
#
# Sum: 15
#
#    10
#   /  \
#  2    3
#
# Sum: 15

# Approach: idea is to know the sum of the entire tree, say 'total'
# Then all we need to do is check if any subtree equals the half of the 'total'
# record all the sums of subtrees in a hash map

def checkSplit(root):
    seen = []
    total = sumTree(root, seen)

    # To deal with the case 0
    #                        \
    #                         0
    seen.pop()
    return (total/2 in seen)


def sumTree(root, seen):
    if root is None:
        return 0

    # Traversing both subtrees
    left_sum = sumTree(root.left, seen)
    right_sum = sumTree(root.right, seen)

    # Adding the sum to the list
    seen.append(left_sum+right_sum+root.data)

    # The last sum in the list is the sum of the entire tree
    return seen[-1]

# Q. Construct a binary tree from a string consisting of parenthesis and integers.
# The whole input represents a binary tree. It contains an integer followed by zero, one or two pairs of parenthesis.
# The integer represents the root’s value and a pair of parenthesis contains a child binary tree with the same structure.
# Always start to construct the left child node of the parent first if it exists.

# Examples:
#
# Input : "1(2)(3)"
# Output : 1 2 3
# Explanation :
#            1
#           / \
#          2   3
# Explanation: first pair of parenthesis contains
# left subtree and second one contains the right
# subtree. Preorder of above tree is "1 2 3".
#
# Input : "4(2(3)(1))(6(5))"
# Output : 4 2 3 1 6 5
# Explanation :
#            4
#          /   \
#         2     6
#        / \   /
#       3   1 5


class newNode:
    def __init__(self, value):
        self.value = value
        self.left = None
        self.right = None

def findIndex(string, start, end):
    if start > end:
        return -1

    stack = []
    for i in range(start,end):
        if string[i] == '(':
            stack.append(string[i])
        if string[i] == ')':
            if stack[-1] == '(':
                stack.pop()
            # if the stack is empty then return the index as this the element contained inside the parenthesis
            if len(stack) == 0:
                return i
    # if not found then return -1
    return -1

def getTree(string, start, end):
    if start > end:
        return None
    # Create a node
    root = newNode(string[start] - '0')
    index = -1

    if start+1 < end and string[start+1] == '(':
        index = findIndex(string, start+1, end)

    # if the index for the complimentary ')' is found then
    if index != -1:
        # the left node will be from the chracter next to '(' till the character just before the complimentary ')'
        root.left = getTree(string, start+2, index-1)
        # the right node will be from the chracter next to '(' after the ')' till the character just before the complimentary ')'
        root.right = getTree(string, index+2, end-1)
    return root

# Q. Given a binary tree where every node has a unique value, and a target key k, find the value of the nearest leaf node to target k in the tree.
# Here, nearest to a leaf means the least number of edges travelled on the binary tree to reach any leaf of the tree.
# Also, a node is called a leaf if it has no children.
# In the following examples, the input tree is represented in flattened form row by row.
# The actual root tree given will be a TreeNode object.
# Example 1:
#
# Input:
# root = [1, 3, 2], k = 1
# Diagram of binary tree:
#           1
#          / \
#         3   2
#
# Output: 2 (or 3)
#
# Explanation: Either 2 or 3 is the nearest leaf node to the target of 1.
# Example 2:
#
# Input:
# root = [1], k = 1
# Output: 1
#
# Explanation: The nearest leaf node is the root node itself.
# Example 3:
#
# Input:
# root = [1,2,3,4,null,null,null,5,null,6], k = 2
# Diagram of binary tree:
#              1
#             / \
#            2   3
#           /
#          4
#         /
#        5
#       /
#      6
#
# Output: 3
# Explanation: The leaf node with value 3 (and not the leaf node with value 6) is nearest to the node with value 2.


# Approach: 1. Use dfs from root of the tree to create a graph (adjacency list)
# 2. Carryout bfs from the target node to find the closest leaf node

def create_graph(root):
    import collections
    graph = collections.defaultdict(list)
    dfs(root, None, graph)
    return graph

def dfs(curr_node, par_node, graph):
    if curr_node is not None:
        # Adding the parent node to current node and vice versa to create an undirected graph (directed graph is useless here)
        graph[curr_node].append(par_node)
        graph[par_node].append(curr_node)

        # Recurse on the children of teh current node
        dfs(curr_node.left, curr_node, graph)
        dfs(curr_node.right, curr_node, graph)

def closest_node(root, target_node):
    min_val = float("inf")
    res = None
    graph = create_graph(root)
    return bfs(target_node, graph, min_val, res)

def bfs(target_node, graph, min_val, res):
    visited = [False] * (len(graph))
    distance = [-1]*len(graph)
    q = []
    q.append(target_node)
    distance[target_node] = 0

    while q:
        e = q.pop()
        visited[e] = True
        for v in graph[e]:
            if visited[v] is False and distance[v] == -1:
                q.append(v)
                visited[v] = True
                distance[v] = distance[e] + 1

            if len(graph[v]) == 0 and min > distance[v]:
                #This is a leaf node
                min_val = distance[v]
                res = v
    return v


# Q. Binary Tree Upside Down
# Given a binary tree where all the right nodes are either leaf nodes with a sibling (a left node that shares the same parent node) or empty, flip it upside down and turn it into a tree where the original right nodes turned into left leaf nodes. Return the new root.
#
# For example:
# Given a binary tree {1,2,3,4,5},
#     1
#    / \
#   2   3
#  / \
# 4   5
# return the root of the binary tree [4,5,2,#,#,3,1].
#    4
#   / \
#  5   2
#     / \
#    3   1

# Approach: O(n) worst condition is when the tree is a linked list where n is the number of nodes in it
def fliptree(root):
    # end conditions
    if root is None:
        return None
    if root.left is None and root.right is None:
        return root

    # recursive call on the left nodes
    flippednode = fliptree(root.left)

    # perform alterations once the left node is obtained
    root.left.left = root.right
    root.left.right = root
    root.left = root.right = None

    # return the new tree
    return flippednode


# Q. Given a binary tree, collect a tree's nodes as if you were doing this: Collect and remove all leaves, repeat until the tree is empty.

# Example:
# Given binary tree
#
#           1
#          / \
#         2   3
#        / \
#       4   5
# Returns [4, 5, 3], [2], [1].

# Approach: Carry out a dfs and add all the nodes at the same height in an array O(n)

def getLeaves(root):
    result = [] #this will be the final list of lists which will be the output
    getLeavesUtil(root, result)
    return result

def getLeavesUtil(root, result):
    if root is None:
        return -1  # end condition; return -1

    left_height = getLeavesUtil(root.left, result)
    right_height = getLeavesUtil(root.right, result)
    curr_height = max(left_height, right_height) + 1

    # This ensures that all the nodes at the same height end uo in the same array; A new array is added to result when the curr_height is changed
    if len(result) < curr_height:
        result.append([])

    # This line adds every node with the same height (curr_height) into the same array
    result[curr_height].append(root)

    # Since height controls the creation of the result array, the function returns the curr_height
    return curr_height