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


# Q Wiggle Sort
#Given an unsorted array nums, reorder it in-place such that nums[0] <= nums[1] >= nums[2] <= nums[3]....
#For example, given nums = [3, 5, 2, 1, 6, 4], one possible answer is [1, 6, 2, 5, 3, 4].

#Approach 1: Sort the array and then starting from the second element, swap consecutive pairs
# example, 1,2,3,4,5,6   ->  1,3,2,5,4,6   : O(nlogn + n)

#Approach 2: Make use of the even or odd positions of the elements

def wiggleSort(arr):
    for i in range(len(arr)):
        if((i%2 == 0 && arr[i] > arr[i+1]) || (i%2 == 1 && arr[i] < arr[i+1])):
            temp = arr[i]
            arr[i] = arr[i+1]
            arr[i+1] = temp
            return arr


# Given a time represented in the format "HH:MM", form the next closest time by reusing the current digits. There is no limit on how many times a digit can be reused.
# You may assume the given input string is always valid. For example, "01:34", "12:09" are all valid. "1:34", "12:9" are all invalid.

# Input: "19:34"
# Output: "19:39"
# Explanation: The next closest time choosing from digits 1, 9, 3, 4, is 19:39, which occurs 5 minutes later.
# It is not 19:33, because this occurs 23 hours and 59 minutes later.

# Approach 1: Simulate the clock going forward by one minute. Each time it moves forward, if all the digits are allowed, then return the current time.
# Since we are representing time in minutes, we will use an integer 't' s.t. (0<= t <= 24*60)
# So, to get the hours we need t/60 and to get the minutes are t%60
# each digit of hours can be found by (hours/10, hours%10)

def nextClosestTime(time):
    #first get the integer 't'
    t = 60*int(time[:2]) + int(time[2:])

    #build list of all the allowed numbers in the string
    allowed = []
    for x in time:
        if x != ":":
            allowed.append(int(x))

    while True:
        t += 1
        hours = t/60
        mins = t%60

        obtained = []
        # get the respective digits
        hour_digit1 = hours/10
        hour_digit2 = hours%10

        min_digit1 = mins/10
        min_digit2 = mins%10

        obtained.append(hour_digit1)
        obtained.append(hour_digit2)
        obtained.append(min_digit1)
        obtained.append(min_digit2)

        for digit in obtained:
            if digit in allowed:
                ans = (hours,mins)
                return ans



# Q. There is a garden with N slots. In each slot, there is a flower. The N flowers will bloom one by one in N days. In each day, there will be exactly one flower blooming and it will be in the status of blooming since then.
#
# Given an array flowers consists of number from 1 to N. Each number in the array represents the place where the flower will open in that day.
#
# For example, flowers[i] = x means that the unique flower that blooms at day i will be at position x, where i and x will be in the range from 1 to N.
#
# Also given an integer k, you need to output in which day there exists two flowers in the status of blooming, and also the number of flowers between them is k and these flowers are not blooming.
#
# If there isn't such day, output -1.
# Input:
# flowers: [1,3,2]
# k: 1
# Output: 2
# Explanation: In the second day, the first and the third flower have become blooming.
#
# Input:
# flowers: [1,2,3]
# k: 1
# Output: -1

# Approach 1: find min and max of the flower array. Those are the first and the last flowers.
# Iterate through the array and check if  1) curr_flower - min - 1 == k  or 2) max - curr_flower - 1 == k for every entry in the active data structure

def findMinMax(arr):
    # O(n)
    min = int("INF")
    max = int("-INF")

    for num in arr:
        if num<min:
            min = num
        elif num>max:
            max = num
    ans = (min,max)
    return ans

def emptySlots(flowers): #O(n^2) as for each flower added to acive we need to query min/max which is O(n)
    # Active keeps track of the current blooming flower and allows us to check if the neighbors of that flower satisfy the condition
    # we want active to be a sorted data structure but can also query min/max everytime on it for every new entry
    active = []
    day = 0
    for curr_flower in flowers:
        day += 1

        active.append(curr_flower)
        res = findMinMax(active)

        if res[1] - curr_flower - 1 == k || curr_flower - res[0] -1 ==k:
            return day

    return -1


# Q. Given an array of meeting time intervals consisting of start and end times [[s1,e1],[s2,e2],...] (si < ei), determine if a person could attend all meetings.
#
# For example,
# Given [[0, 30],[5, 10],[15, 20]],
# return false.

# Ans : 1.Sort the times by the start times and check if the end time of the current period is less than the start time of the next

# Q. Given a list of words and two words word1 and word2, return the shortest distance between these two words in the list.
#
# For example,
# Assume that words = ["practice", "makes", "perfect", "coding", "makes"].
#
# Given word1 = “coding”, word2 = “practice”, return 3.
# Given word1 = "makes", word2 = "coding", return 1.

#Approach: O(n) use two indices to scan the array and find the minimum distance between the two words
def wordsDsitance(words, w1, w2):
    i1 = -1
    i2 = -1
    dist = float("INF")

    for i in range(len(words)):
        if words[i] == w1:
            i1 = i
        if words[i] == w2:
            i2 = i
        if i1 != -1 and i2 != -1:
            if dist > abs(i1-i2):
                dist = abs(i1-i2)
    return dist



# Q.  Write a function to generate the generalized abbreviations of a word.
# Example:
# Given word = "word", return the following list (order does not matter):
# ["word", "1ord", "w1rd", "wo1d", "wor1", "2rd", "w2d", "wo2", "1o1d", "1or1", "w1r1", "1o2", "2r1", "3d", "w3", "4"]

# Answer: Using Bit Manipulation

# 1. consider the word and 'x' which will be a bit representation of every single possible combination
#for example, x=1010 will represent w1r1(because word[0] = w and x is being shifted left) i.e. if the bit position has 1 then the character will be swapped with a number and if not then it remains

def getAbbr(word):
    # x is the bit representation and is incremented by 1 to get the next one
    for x in range(len(word)):
        return getAbbrUtil(word,x)

def getAbbrUtil(word, x):
    abbr = []  #the abbreviation will be constructed in it
    k = 0     # keep track of the number depending on the bit at the character's position

    #Looping through every bit position in x
    for i in range(len(word)):
        # 0 means retain the word
        if x&1 == 0:
            if k!= 0:
                abbr.append(k)
                k = 0
            #always append the character at the current position i
            abbr.append(word[i])
        else:
            k += 1
        x = x>>1
    #Adding the last k in case it is non-zero
    if k!=0:
        abbr.append(k)
    return abbr

#Q. Numbers can be regarded as product of its factors. For example,
# 8 = 2 x 2 x 2;
#   = 2 x 4.
# Write a function that takes an integer n and return all possible combinations of its factors.
# Note:
# You may assume that n is always positive.
# Factors should be greater than 1 and less than n.
# Examples:
# input: 1
# output:
# []
# input: 37
# output:
# []
# input: 12
# output:
# [
#   [2, 6],
#   [2, 2, 3],
#   [3, 4]
# ]
# input: 32
# output:
# [
#   [2, 16],
#   [2, 2, 8],
#   [2, 2, 2, 4],
#   [2, 2, 2, 2, 2],
#   [2, 4, 4],
#   [4, 8]
#]

# Approach: use recursion on every quotient obtained on dividing the number by factor
# maintain a single_list to contain the current factors and a final_list to contain all the single_lists

def getFactCombinations(num):
    final_list = []
    single_list = []
    getFactCombinationsUtil(2, 1, num, final_list, single_list)
    return final_list

def getFactCombinationsUtil(factor, curr_product, num, final_list, single_list):
    # end condition for recursion
    if factor>num || curr_product>num:
        return

    # if the current product is equal to the number then, we get the list of factors in the single_list
    # so, add it to the final list
    if curr_product == num:
        final_list.append(single_list)
        return

    # looping through all the factors of the number and recursively checking every quotient we get
    for i in range(factor, num):
        if i*curr_product > num:
            break

        if num%i == 0:
            # add the factor to the single_list
            single_list.append(i)

            # Recursive call on the current factor i.e. i an dthe current product which is i*curr_product
            getFactCombinationsUtil(i, i*curr_product, num, final_list, single_list)

            # remove the last factor added
            single_list.pop(len(single_list)-1)



# Q. Design a max stack that supports push, pop, top, peekMax and popMax.
#
# push(x) -- Push element x onto stack.
# pop() -- Remove the element on top of the stack and return it.
# top() -- Get the element on the top.
# peekMax() -- Retrieve the maximum element in the stack.
# popMax() -- Retrieve the maximum element in the stack, and remove it. If you find more than one maximum elements, only remove the top-most one.
# Example 1:
# MaxStack stack = new MaxStack();
# stack.push(5);
# stack.push(1);
# stack.push(5);
# stack.top(); -> 5
# stack.popMax(); -> 5
# stack.top(); -> 1
# stack.peekMax(); -> 5
# stack.pop(); -> 1
# stack.top(); -> 5


# Approach : use two stacks s1 and s2. Treat s1 as the main normal stack and s2 which always keeps the max element at the top
# O(n) due to popMax as we have to go through all elements of stack1 to delete the element. All other operations are O(1)
class MaxStack:
    stack1 = []
    stack2 = []
    max = float('-inf')

    def push(self, x):
        self.stack1.append(x)
        if max < x:
            self.stack2.append(x)

    def pop(self):
        self.stack2.pop()
        return self.stack1.pop()

    def top(self):
        return self.stack1[-1]

    def peekMax(self):
        return self.stack2[-1]

    def popMax(self):
        max_elem = self.stack2.pop()
        buffer = []
        while self.stack1.top() != max_elem:
            buffer.append(self.stack1.top())
        self.stack1.pop()
        while len(buffer) != 0:
            self.stack1.append(buffer.pop())


