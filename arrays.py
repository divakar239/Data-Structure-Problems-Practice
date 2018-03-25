#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 14 15:49:43 2018

@author: DK
"""

##### POINTERS ###############

# 1. consider overwriting rather than deleting
# 2. writing from the back is faster than writing from the front
# 3. To keep space : O(1) use the array itself for operations by pointers 

##############################

# Q  Implement a function that given an integer array and its length, returns the length of the longest 
#sequence of alternating odd and even numbers. For instance, in the array 112365546, 
#it will return 5 (for the sequence 12365).

#Approach : Naive t:O(n^2)
def count_odd_even(a):
    count = 0
    max = 0
    for i in range(0,len(a)):
        val = a[i]%2
        count = 1                       #reset the counter to 1 as it is the beginning of a new subarray
        for j in range(i+1,len(a)):
            if a[j]%2 != val:
                count = count + 1
                if max < count:
                    max = count
                val = a[j]%2
            elif a[j]%2 == val:
                break
    return max

#Q Given an array of size n, the array contains numbers in range from 0 to k-1 where k is a positive integer and k <= n.
# Find the maximum repeating number in this array in T:O(n) and S:O(1)

# Approach 1: run two for loops : T:O(n^2) S:O(1) 
# Approach 2: run a for loop, maintain a hash map : T:O(n) S:O(n)
# Approach 3: t: O(n) and S: O(1)

# same idea can be used to segrgate odd and even numbers
def max_frequency(a,k):
    #modify the array by adding adding k to the a[i]%k element: adding K will retain the original value of the value on value%k
    for i in range(0,len(a)):
        a[a[i]%k] = a[a[i]%k] + k
    #fetch the maximum value
    max = 0
    index = 0
    for i in range(0,len(a)):
        if max < a[i]:
            max = a[i]
            index = 0
            #index of the max value is the most frequently appearing number in the array
    return index
            
        
# Q Dutch Flag Partition problem: given a pivot arrange all elements of thearray< pivot at the start followed by all 
#elements equal to the pivot followed by all the elements > pivot in time: O(n) and space: O(1)

# Remember: the concept of partitioning from quicksort is very important as we can treat the array to have multiple
#categories within itself and rearrange it without extra space

# This technique can be used to solve variants of this problem:
    # 1. Given a value , group all its instances together
    # 2. group element of type A at the start and element of type B to the end
def dutch_flag_partition(pivot_index,a):
    pivot = a[pivot_index]
    
    #smaller elements put first
    index_smaller = 0
    for j in range(len(a)):
        if a[j] < pivot:
            a[j], a[index_smaller] = a[index_smaller],a[j]
            index_smaller = index_smaller + 1
    
    #grouping larger elements
    index_larger = len(a) - 1
    for j in reversed(range(len(a))):
        if a[j] > pivot:
            a[j],a[index_larger] = a[index_larger],a[j]
            index_larger = index_larger - 1
        elif a[j] < pivot:
            break
   
        
# Q Write a program which takes in an array of digits encoding a non negative decimal "D" and returns
# D+1 as the result eg [1,2,9] gives [1,3,0]

def add_one(a):
    a[len(a)-1] = a[len(a) - 1] + 1
    for i in reversed(range(1,len(a))):
    if a[i] == 10:
        a[i] = 0
        a[i-1] = a[i-1] + 1
    else:
        break
    # check if the first element is 10 then make it 1 and add a zero to the end of the array
    if a[0] == 10:
        a[0] = 1
        a.append(0)
# Variant: same can be used to solve adding two bit strings i.e the technique of changing the first element of the sum array is 2 then change it to 1 and append a 0 to the end.

# Q. remove duplicates from a sorted array : time: O(n) and sapce: O(1)
def remove_duplicates(a):
    j = 0
    for i in range(1,len(a)):
        if a[i] != a[i-1]:
            a[j] = a[i-1]
            j = j + 1
    a[j] = a[len(a) - 1]
    return a

#Q input an array denoting daily stock prices and return the maximum profit that 
#can be made by buying and selling one share of that stock

#Approach : time:O(n) and space : O(1)

#CONCEPT: The idea is that the selling price should be the minimum of the first 
#subarray and the selling price should be the maximum of the second subarray.
#NOTE: finding the absolute maximum and minimum prices here won't help as they 
#could be in differenet subarrays

# To find the maximum profit we will compute the difference of the current price 
#and the minimum that we have found so uptill that point.

#eg [310,315,275,295,260,270,290,230,255,250] , the minimum price till:
    # 0th place is 310
    # 1st place is 310
    # 2nd place is 275 and so on , so the array with the minimum array till each current position is:
        # [310,310,275,275,260,260,260,230,230,230]
        # To get the maximum profit , we have to subtract these two arrays and choose the maximum number
        
def buy_sell_stock(prices):
    min_price_so_far = float('inf')
    max_profit = 0
    for curr_price in prices:
        curr_profit = price - min_price_so_far                      # difference of the two arrays
        max_profit = max(max_profit,curr_profit)
        min_price_so_far = min(curr_price,min_price_so_far)         # creates the min array
    return max_profit

# Find largest sum pair in an unsorted array in one pass
# approach: define two variables, max, sec_max
# sec_max tracks max as it is altered 

def find_max_sum_pair(a):
    first_max = max(a[0],a[1])
    sec_max = min(a[0],a[1])
    for i in range(2,len(a)):
        if first_max < a[i]:
            sec_max = first_max
            first_max = a[i]
        elif sec_max < a[i] and a[i] != first_max:
            sec_max = a[i]
    return (first_max + sec_max)

# Q Print all prime factors of a number
def prime_factors(n):
    factors = []
    #first remove all the even factors by dividing by two
    while n%2 == 0:
        factors.append(2)
        n = n//2
    #remaining num will be odd
    for i in range(3,int(math.sqrt(n)),2):
        if n%i == 0:
            factors.append(i)
            n = n//i
    #if leftover is > 2 then it has to be prime
    if n > 2:
        factors.append(n)
    return factors

# Check if a number is prime
def check_prime(n):
    #check if it is even
    if n%2 == 0:
        return False
    for i in range (3,n):
        if n%i == 0:
            return False
    return True
    
# Print all prime numbers till k


# remove element at key in space : O(1)
def remove(a,key):
    for i in range(key+1,len(a)):
        #print(a[i])
        a[i-1] = a[i]
    a = a[:-1]
    return a
    
# reorder elements to place even numbers first and odd at the end in space O(1)
# idea is to maintain 3 partitions in the array, even , odd and unclassified. 
# We start with unclassified and use one if the pointer either even or odd as an anchor which proceeds only when the condition at that point is met
# the other pointer is used to scan its half of the array and shrink the unclassified part of the array by swapping elements with the other pointer

def collect_even_odd(a):
    even = 0
    odd = len(a) - 1
    while even < odd:
        #use odd as an anchor
        if a[odd]%2 == 1:
            odd -= 1
        #use even to scan the first half of the array
        else:
            # 1. swap the elements
            temp = a[odd]
            a[odd] = a[even]
            a[even] = temp
            # 2. make even proceed to continue scan
            even += 1

# Apply a permutation to a given array:
    # eg array is A = [a b c d] and permutation P = [2 0 1 3]
    # output should be [b c a d] that is index i in P gives the element P[i] that is mapped to it
    
#approach 1: space : O(n)
def app_perm(A,P):
    b = A
    for i in range (len(P)):
        b[P[i]] = A[i]
    return b
    
# approach 2: space : O(1)
# idea is a permutation can be split into set of cyclic permutations which are ones that move all elements by a fixed offset
# each independent cyclic permutation can be applied independently
# subtract len(A) from the value in P to indicate that the permutation was performed

def app_perm_v2(A,P):
    for i in range(len(A)):
        next = i
        #check if the P[i] permutation is greater than 0 , only then apply it
        while p[next] >= 0:
            A[i], A[P[next]] = A[P[next]], A[i]
            temp = P[next]
            P[next] -= len(P)
            next = temp


#Q.Count minimum number of subsets with continuous elements: O(nlogn)
def count_subsets(a):
    a=a.sort()
    count = 0
    for i in range(len(a)):
        if a[i] != a[i+1]:
            count += 1
    return count
    
# Q.Smallest subarray with sum greater than a given value
# Approach 1: O(n^2)
def smallest_sub(a,s):
    min_len = len(a) + 1
    for i in range(len(a)):
        curr_sum = a[i]
        for j in range(i+1, len(a)):
            curr_sum += a[j]

            # to handle negative numbers , ignore curr_sum if it becomes negative
            if curr_sum < 0:
                break
            
            if curr_sum > s and min_len > j-i:
                min_len = j-i
                break
        return min_len

#Approach O(n)
def smallest_sub(a,s):
    min_len = len(a) + 1
    start, end = 0
    curr_sum = 0
    
    while(end<len(a)):
        while end<len(a) and curr_sum<s:
            curr_sum += a[end]
            end += 1
            
        while curr_sum>s and start<len(a):
            if min > end-start:
                min = end-start
            curr_sum - a[start]
            start += 1
    return min_len
       

# Q. sort a stack in place that is using recursion
def sort_stack(s):
    if len(s) != 0:
        e = s.pop()
        # sort the remaining stack
        sort_stack(s)
        # insert the top item back
        sorted_insert(s,e)
        
def sorted_insert(s,x):
    #if the stack is empty or x > top of the stack
    if len(s) == 0 or x>s[len(s) - 1]:
        s.append(x)
        return
    #if x<top of the stack then remove the top and recur to find the correct position of x
    else:
        e = s.pop()
        sorted_insert(s,x)
    #once x is inserted push the removed top back to the stack
    s.append(e)
    
# Q. Reverse a stack in place using recursion (same idea as above)
def reverse_stack(s):
    if len(s) != 0:
        e = s.pop()
        reverse_stack(s)
        reversed_insert(s,e)

def reversed_insert(s,x):
    if len(s) == 0:
        s.append(x)
        return
    else:
        e = pop()
        reversed_insert(s,x)
        s.append(e)
    