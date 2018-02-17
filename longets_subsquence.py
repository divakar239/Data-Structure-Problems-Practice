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
### Generate all the subsequences of an array
#An efficient way is to use a map. 
#The idea is to maintain sum of elements encountered so far in a variable (say curr_sum). 
#Let the given number is sum. 
#Now for each element, we check if curr_sum – sum exists in the map or not. 
#If we found it in the map that means, we have a subarray present with given sum, else we insert curr_sum into the map and proceed to next element. 
#If all elements of the array are processed and we didn’t find any subarray with given sum, then subarray doesn’t exists.
import collections

def cont_subarray_neg_sum_v2(array,sum):
    storage = collections.defaultdict(int)
    index = 0
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
        index = i
    print("No subarray found")
    


### Q Find all possible subsequences of an array ###

### Approach 1
# O(n*2^n) iterate through all 2^n subsequences

def count_subsequences_v1(array):
    for i in range(0,2^len(array)):
        print('test')
        for j in range(0, len(array)):
            if i&(1<<j):
                print(array[j])
        print('\n')
# Longest Subsequence Problem

