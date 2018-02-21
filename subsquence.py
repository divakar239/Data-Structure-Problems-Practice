###SubArray with a given sum ###

# Q: Given an array of non-negative numbers, find continuous subarray with sum to S.

### Approach number 1
#   - O(n^2) as we need nested for loops to find the continuous pair

def cont_subarray_sum_v1(array, sum):
    index = 0
    for i in range(len(array)):
        temp_sum = 0
        temp_sum = sum - array[i]
        index = i 
        for j in range(i+1,len(array)):
            if temp_sum == 0:
                for k in range(i,index + 1):
                    print(array[k])
                return
            temp_sum = temp_sum - array[j]
            index = j
    print("No substring found")
    
### Approach number 2
#   - O(n) as we use the sliding window approach
#   - Start with two pointers that represent the subarray.
#   - If the sum of current subarray is smaller than S, move the end pointer one step forward.
#   - If the sum is larger than S, move the start pointer one step forward.
#   - Once the current sum equals to S, we found the target.

def cont_subarray_sum_v2(array,sum):
    left_ptr = 0
    right_ptr = 1
    temp_sum = array[right_ptr] + array[left_ptr]

    while(left_ptr<right_ptr):
        if(temp_sum < sum):
            right_ptr = right_ptr + 1
            temp_sum = temp_sum + array[right_ptr]

        elif(temp_sum  > sum):
            temp_sum = temp_sum - array[left_ptr]
            left_ptr = left_ptr + 1
            
        else:
            for i in range(left_ptr,right_ptr+1):
                print(array[i])
            return
    print("No subarray found")


    
### Handling negative numbers

### Approach 1 : O(n^2) same as above
def cont_subarray_neg_sum_v1(array, sum):
    index = 0
    for i in range(len(array)):
        temp_sum = 0
        temp_sum = sum - array[i]
        index = i 
        for j in range(i+1,len(array)):
            if temp_sum == 0:
                for k in range(i,index + 1):
                    print(array[k])
                return
            temp_sum = temp_sum - array[j]
            index = j
    print("No substring found")
    

###Approach 2: Since, the array has negative numbers the sliding window can't be used as it
#it changed the pointers based on the assumption
#An efficient way is to use a map. 
#The idea is to maintain sum of elements encountered so far in a variable (say curr_sum). 
#Let the given number is sum. 
#Now for each element, we check if curr_sum – sum exists in the map or not. 
#If we found it in the map that means, we have a subarray present with given sum, else we insert curr_sum into the map and proceed to next element. 
#If all elements of the array are processed and we didn’t find any subarray with given sum, then subarray doesn’t exists.
import collections

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
    


### Q Find all possible subsequences of an array ###

### Approach 1 (iteration)
# O(n*2^n) iterate through all 2^n subsequences

# There will be 2^n subsequences. this is the lower time bound.
# We will use binary numbers each of length 'n' to encode a subsequence for example:
    # array = [a,b] ; n = 2
    # 00 = []
    # 01 = [a]
    # 10 = [b]
    # 11 = [a,b]

# for each subset, we will shift 1 right from position 0 to n-1 in the encoding, hence the time complexity
def count_subsequences(array):
    for i in range(0,pow(2,len(array))):
        #print('test')
#Check if jth bit in the counter is set, If set then print jth element from arr[]
        for j in range(0, len(array)):
            if i&(1<<j):                    # this shifts 1 right by current value of 'j' which aims to iterate through all 'n' positions for 1 subset          
                print(str(array[j]) + " ")
        print('\n')
        
### Approach 2 (recursion)
## idea is if we know 's' subsets of A and want to find the subsets of 'B' where B > A
#the subsets of B will be of two types 1. all contained in 's' 2. 's' + x where x is the element not present in A

def subsets(s):
    #end condition; return null set if no element is present
    if len(s) == 0:
        return [[]]
    #divide the array from the first element into two groups a[0] and a[1:]
    h = s[0]
    t = s[1:]
# first compute the subsets of A which is s[1:] excluding s[0]
    ss_excl_h = subsets(t)
# add s[0] to every element of ss_excl_h
    ss_incl_h = []
    for ss in ss_excl_h:
        ss_incl_h.append(ss + [h])
    #ss_incl_h = (([h] + ss) for ss in ss_excl_h)
    #print(type(ss_incl_h))
    return ss_incl_h + ss_excl_h
    
    
# Q1 Search an element in a rotated array
#Input arr[] = {3, 4, 5, 1, 2}
#Element to Search = 1
#  1) Find out pivot point and divide the array in two
#      sub-arrays. (pivot = 2) /*Index of 5*/
#  2) Now call binary search for one of the two sub-arrays.
#      (a) If element is greater than 0th element then
#             search in left array
#      (b) Else Search in right array
#          (1 will go in else as 1 < 0th element(3))
#  3) If element is found in selected sub-array then return index
#     Else return -1.

# Q2 Rotate a linked list by k nodes counter clockwise which also means clockwise by n-k
# idea : 1. Grab the last node and connect last_node.next = head
#        2. run for loop to grab the kth node
#        3. head = kth_node.next 
#        4. kth_node = None 

#Given an unsorted array of integers, find the length of the longest consecutive elements sequence. 
#For example, 
#Given [100, 4, 200, 1, 3, 2], 
#The longest consecutive elements sequence is [1, 2, 3, 4] in time:O(n)

def consec_elem(a):
    s = set()
    length = 0
    
    #add all the elements to the set
    for elem in a:
        s.add(elem)
    
    for i in range(len(a)):
        if (a[i]-1) not in s:
            curr_elem = a[i]
            while curr_elem in s:
                curr_elem = curr_elem + 1
            length = max(length,curr_elem - a[i])
    return length    

#Partition a set into two subsets such that the difference of subset sums is minimum
#Input:  arr[] = {1, 6, 11, 5} 
#Output: 1
#Explanation:
#Subset1 = {1, 5, 6}, sum of Subset1 = 12 
#Subset2 = {11}, sum of Subset2 = 11        
    
# Recursion: The recursive approach is to generate all possible sums from all the values of array and to check which solution is the most optimal one.
#To generate sums we either include the i’th item in set 1 or don’t include, i.e., include in set 2.

def find_min_rec(a,length,sumcalculated,sumtotal):
    if(length == 0):
        return abs((sumtotal-sumcalculated)-sumcalculated)
    else:
        res1 = find_min_rec(a,length-1,sumcalculated + a[length - 1],sumtotal) # included in 1st set
        res2 = find_min_rec(a,length-1,sumcalculated,sumtotal)
    return min(res1,res2)
    
def find_min(a):
    sumtotal = 0
    for e in a:
        sumtotal = sumtotal + e;
    return find_min_rec(a,len(a),0,sumtotal)
    

# Partition problem is to determine whether a given set can be partitioned into two subsets such that the sum of elements in both subsets is same.
#arr[] = {1, 5, 11, 5}
#Output: true 
#The array can be partitioned as {1, 5, 5} and {11}
#
#arr[] = {1, 5, 3}
#Output: false 
#The array cannot be partitioned into equal sum sets.
#
#1) Calculate sum of the array. If sum is odd, there can not be two subsets with equal sum, so return false.
#2) If sum of array elements is even, calculate sum/2 and find a subset of array with sum equal to sum/2.
#
#The first step is simple. The second step is crucial, it can be solved either using recursion or Dynamic Programming.
#
#Let isSubsetSum(arr, n, sum/2) be the function that returns true if 
#there is a subset of arr[0..n-1] with sum equal to sum/2
#
#The isSubsetSum problem can be divided into two subproblems
# a) isSubsetSum() without considering last element 
#    (reducing n to n-1)
# b) isSubsetSum considering the last element 
#    (reducing sum/2 by arr[n-1] and n to n-1)
#If any of the above the above subproblems return true, then return true. 
#isSubsetSum (arr, n, sum/2) = isSubsetSum (arr, n-1, sum/2) ||
#                              isSubsetSum (arr, n-1, sum/2 - arr[n-1])


# This is step 2 of the solution
def find_partition_util(arr,n,sum):
    if sum == 0:
        return True
    if n == 0 and sum != 0:
        return False
    if arr[n-1] > sum:
        return find_partition_util(arr,n-1,sum)
    else:
        return find_partition_util(arr,n-1,sum) or find_partition_util(arr,n-1,sum - arr[n-1])

#This is the first step of the solution
def find_partition(arr,n):
    sum = 0
    for elem in arr:
        sum = sum + elem
    if sum%2 == 1:
        return False
    else:
        return find_partition_util(arr,n,sum)
        
        
