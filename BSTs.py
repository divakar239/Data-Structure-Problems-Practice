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
        if(self.root==None):
            self.root=bst_node(data)
        else:
            self.insert_node(self.root,data)
    
    def insert_node(self,root,data):
        if(data<=root.data):
            if(root.left!=None):
                self.insert_node(root.left,data)
            else:
                root.left=bst_node(data)
                return
        else:
            if(root.right!=None):
                self.insert_node(root.right,data)
            else:
                root.right=bst_node(data)
                return

# Q2. input BST and num K, and returns the k largest keys (code above)
# Approach 1: Inorder traversal and print the last k elelments. However, to=is can prove to be very ineffective especially if the length of the left subtree is really large
# Approach 2: Reverse inorder but stop as soon as the kth element is hit.
    def reverse_inorder_traversal(self,num):
        if self.root == None:
            return
        else:
            return self.reverse_inorder(self.root,num)
               
    def reverse_inorder(self,root,k):
     
        if root == None:
            return
        if root.right != None:
            self.reverse_inorder(root.right,k)
       
        if(i<=k):
            print(root.data)
        if root.left != None:
            self.reverse_inorder(root.left,k)

            
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
            return self.ind_LCA_util(root.left,a,b)
        if root.data<a.data and root.data<b.data:
            return self.find_LCA_util(root.right,a,b)
            
# Find height of the tree / also find balance factor that is the difference in height of the left and right subtree
    def height(self,root):
        if root == None:
            return 0
        lheight = self.height(root.left) + 1
        rheight = self.height(root.right) + 1
        
        tree_height = max(lheight,rheight)
        balance_factor = abs(lheight - rheight)
        
            
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

# For BST we only need the prorde or post order. Inorder is not required.and alone it can't give a unique BST.


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

##### BSTs #####

# Q1. Find first key that would appear immediately after the given value in the BST

def succesor(tree,value):
    if value == tree.root.data:
        return tree.root.right.data
    elif value < tree.root.data:
        return succesor(tree.root.left, value)
    else:
        return succesor(tree.root.right, value)
        
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
    
    
#Given a binary tree, print all root-to-leaf paths: O(n^2)

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
# before printing the elements in "if node.left == None and node.right == None: add eleemnts and compare with sum.

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
        






