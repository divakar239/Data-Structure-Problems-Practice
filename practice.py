#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 28 19:28:50 2017

@author: DK
"""


####Singly Linked List:        
#class linked_list:
#    class node:
#        def __init__(self,data):
#            self.data=data
#            self.next=None

#    def __init__(self):
#        self.head=None
#    
#    def insert(self,data):
#        temp = node(data)
#        if(self.head == None):
#            self.head=temp
#        else:
#            temp.next=self.head
#            self.head=temp
#            
#    def search(self,data):
#        node=self.head
#        if(node ==None):
#            return -1
#        while(node.next!=None):
#            if(node.data == data):
#                return True
#            node=node.next
#        else:
#            return False
#                
#    def traversal(self):
#        node=self.head
#        if(node==None):
#            return -1
#        while(node != None):
#            print(node.data)
#            node=node.next
#            
#    def remove(self,data):
#        prev=self.head
#        if(prev==None):
#            return -1
#        curr=prev.next
#        while(curr!=None):
#            if(curr.data==data):
#                prev.next=curr.next
#                curr.next=None
#                #print("done")
#                return
#            curr=curr.next
#            prev=prev.next
#            
#    def reverse(self,node):
#        temp=node
#        if(temp.next == None):
#            self.head=temp
#            return
#        self.reverse(temp.next)
#        temp2=temp.next
#        temp2.next=temp
#        temp.next=None

###test code
#class number:
#    def __init__(self, number):
#        self.number=number
#        
#class person:
#
#    def __init__(self,num):
#        self.number_1=number(num)
#        
        
###BST
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
                
    ##Traversals            
    def preorder_traversal(self):
        if(self.root!=None):
            self.preorder(self.root)
            
    def preorder(self,root):
        if(root==None):
            return
        print(root.data)
        
        if(root.left):
            self.preorder(root.left)
        if(root.right):
            self.preorder(root.right)
            
    def postorder_traversal(self):
        if(self.root!=None):
            self.postorder(self.root)
            
    def postorder(self,root):
        if(root==None):
            return
        if(root.left):
            self.postorder(root.left)
        if(root.right):
            self.postorder(root.right)
        print(root.data)
          
    def inorder_traversal(self):
        if(self.root!=None):
            self.inorder(self.root)
            
    def inorder(self,root):
        if(root==None):
            return
        if(root.left):
            self.inorder(root.left)
            
        print(root.data)
        
        if(root.right):
            self.inorder(root.right)
            
    def reverse_inorder_traversal(self):
        if self.root == None:
            return
        else:
            return self.reverse_inorder(self.root)
               
    def reverse_inorder(self,root):
        if root == None:
            return
        if root.right != None:
            self.reverse_inorder(root.right)
        print(root.data)
        if root.left != None:
            self.reverse_inorder(root.left)
            

### Finding a pair of numbers in an array which add up to a sum s

#Method 1: using dictionaries : T(n) = O(n)        space complexity = O(n)
    
#def pair_1(s,a=[]):
#    bool_dict={}
#    for i in range(0,10):
#        bool_dict[i] = False;
#    for j in range(0,len(a)):
#        if(s-a[j] >=0 and bool_dict[s-a[j]] == True):
#            print(a[j],s-a[j])
#        else:
#            bool_dict[a[j]]=True
#         
##Method 2:  using sliding window  : T(n) = O(nlog(n)) due to sort() = O(nlogn) space complexity = O(1)
#
#def pair_2(s,a=[]):
#    b=a[:]
#    b.sort()
#    size=len(a)
#    l=0
#    r=size-1
#
#    while(l<r):
#        if(b[l]+b[r] == s):
#            print(b[l],b[r])
#            l=l+1
#            r=r-1
#            
#        elif(b[l]+b[r] > s):
#            r=r-1
#            
#        else:
#            l=l+1
#
#### Finding a triplet of numbers in an array which add up to a sum 's' : T(n) = O(n^2)      space complexity = O(1)
#def pair_3(s,a=[]):
#    b=a[:]
#    b.sort()
#    size=len(a)
#    r=size-1
#
#    for i in range(0,len(b)):
#        print("test")
#        l=i+1
#        x=a[i]
#        while(l<r):
#            if(b[l]+b[r] == s-x):
#                print(x,b[l],b[r])
#                l=l+1
#                r=r-1
#                
#            elif(b[l]+b[r] > s-x):
#                r=r-1
#                
#            else:
#                l=l+1

        
        
            
        
        
    
        