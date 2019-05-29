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
            for k in range (storage[curr_sum-sum]+1, i+1):
                print(array[k])
            return
        storage[curr_sum] = i
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


# Python program to print all
# subset combination of n
# element in given set of k element .

# The main function that
# prints all combinations
# of size r in arr[]

def subsets_size(arr, r):
    # A temporary array to
    # store all combination
    # one by one
    data = list(range(r))

    # Print all combination
    # using temporary
    # array 'data[]'
    subsets_size_util(arr, k, data, 0, 0)

# arr[] ---> Input Array
# data[] ---> Temporary array to
#             store current combination
# start & end ---> Staring and Ending
#                  indexes in arr[]
# index ---> Current index in data[]
# r ---> Size of a combination
#        to be printed
def subsets_size_util(a, k, temp, index_data, index_arr):
    # Current combination is
    # ready to be printed,
    # print it
    if index_data == k:
        for j in range(r):
            print(data[j], end=" ")
        print(" ")
        return

    # When no more elements
    # are there to put in data[]
    if index_arr >= len(a):
        return

    # current is included,
    # put next at next
    # location
    data[index_data] = a[index_arr]
    subsets_size_util(a, k, temp, index_data+1, index_arr+1)

    # current is excluded,
    # replace it with
    # next (Note that i+1
    # is passed, but index
    # is not changed)
    subsets_size_util(a, k, temp, index_data, index_arr+1)

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
# Approach : 1. Create a hash set to store all elements for constant lookup
# check if the element is the beginning of the sequence by checking if a[i] - 1 is present in the set
# If the element is the first , then count number of elements in the consecutive starting with this element
# if count is more than the current count then update the count

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


# This is step 2 of the solution where n is the length of the array
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
        return find_partition_util(arr,n,sum//2)


# Longest Substring Without Repeating Characters

# Naive : O(n^3)
# Approach : create a is_unique function to check if the substring being iterated from a start to end is unique
# iterate through all substrings and update the length of the substring to the maximum

def longest_substring(s):
    length = 0
    for i in range(len(s)):
        for j in range(i+1, len(s)):
            if is_unique(s,i,j):
                length = max(length, j-i)

    return length

def is_unique(s, start, end):
    storage = set()             # since it is local to this function, for every new string it always starts empty and breaks as soon as it hits a duplicate character in that substring

    for i in range(start, end):
        if s[i] in storage:
            return False
        else:
            storage.add(s[i])
    return True

# To make this O(N^2) in the is_unique function instead of checking the entire substring everytime, just check s[j]
def longest_substring_v2(s):
    length = 1

    if len(s) == 1:
        return 1

    for i in range(len(s)):
        storage = set()             # initialising the set everytime a new substring starts
        storage.add(s[i])           # always add the first element of the substring in the set when it starts

        for j in range(i+1, len(s)):
            if s[j] not in storage:
                storage.add(s[j])
                length = max(length, j-i+1)
            else:
                break
    return length

# Approach 2:
    # use sliding window: i set to an element while j moves forward checking if the element its on is in the set
    # if it is it removes the ith element from the set and shifts i to the right by 1
def longest_substring_v3(s):
    length = 0
    i = 0
    j = 0
    storage = set()
    while i<len(s) and j<len(s):
        if s[j] not in storage:
            storage.add(s[j])
            length = max(length, j-i)
            j += 1
        else:
            storage.remove(s[i])
            i+=1
    return length

# Q. find longest palindrome substring in a string
# Approach 1: O(n^3) similar to the above question: loop through the string and check if each substring is a palindrome
# for this will have to create a palindrome method


# Q. find continuous triplet which increases; that is a subarray T:O(n) and space: O(1) uses sliding window
def increasingTripletContinuous(nums):
        count = 1
        i = 0
        j = 1
        while i<len(nums) and j<len(nums):

            if count == 3:
                return True
            if nums[i] < nums[j]:
                count += 1

            else:
                count = 1


            i += 1
            j += 1

        return False

# Q. same as above but a subsequence that is not continuous
# idea is to find any 3 elements a,b,c such that a<b<c
def increasingTriplet(nums):
        a, b = float("inf"), float("inf")

        for c in nums:
            #1. use c to find absolute minimum
            if a >= c:
                a = c
            elif b >= c:
            #2. after absolute minimum find next minimum and set it to b
                b = c
            else:  # a < b < c
            # if none of the above happens that means a and b have been fixed and c is greater than both
                return True
        return False

# Q. remove duplicates and reverse string
# use the hash map only to keep track of which is duplicate and NOTHING MORE
# keep adding every element to a list whenever it is first entered into the hash map
# this will maintain order of the string and remove duplicates

def remove_reverse(s):
    new_str = []
    storage = collections.defaultdict(int)
    for i in range(len(s)):
        if storage[s[i]] == 0:
            storage[s[i]] = 1
            #terminating += 1
            new_str.append(s[i])
    for i in reversed(range(len(new_str))):
        print(new_str[i])
    return new_str


# Q. Check for balanced parenthesis
# create dictionary: key = left parts and value = right parts
# push all the left parts in a stack
# for every right part
# 1.if the stack is empty means that it can't form a pair
# 2.the popped element popped doesn't match the corresponding value of the key in the storage dictionary

def check_parenthesis(s):
    stack = []
    storage = {'(':')', '{':'}','[':']'}
    for char in s:
        if storage[char]:
            stack.append(char)
        elif storage[stack.pop()] != char or len(stack) == 0:
            return False
    return True



#Q. group all anagrams together in an array of strings : O(n^2)
group = collections.namedtuple("group",("sum","index"))

def anagrams(s):
    res_l=[]
    b=[]
    for i in range(len(s)):
        res = add(s[i])
        res_l.append(group(res,i))
    #NOTE: LIST.SORT(0 DOES SORTING IN PLACE AND RETURNS NONE)
    b = sorted(res_l)
    return b

def add(s):
    r = 0
    for i in range(len(s)):
        r += ord(s[i])
    return r


    # NOTE: given a string 's', find the number of palindrome permutations of the string
    # 1. check if any permutation of the string can be a palindrome i.e
        # if length is even , then count frequency of every character using a hash map; every character should occur even number of times
        # if length is odd, then only 1 char should be odd all the other should be even
    # 2. if the string checks out the palindrome test then the number of palindrome permutations are ((LEN(S)//2)!)

# Q1 Check if string is a palindrome
def check_palindrome(s):
    for i in range(0,len(s)-1):
        if(s[i] != s[len(s)-1-i]):
            print("Not Palindrome")
            return
    print("Palindrome")


# Q. reverse the words in a string:
    # eg the cat is bad -> bad is cat the

#def rev_words(sentence):
#    i = 0
#    j = 0
#    s = list(sentence)
#
#
#
#def reverse(s):
#    b = list(s)
#    for k in range(len(s)//2):
#        tmp = b[len(b)-k-1]
#        b[len(b)-k-1] = b[k]
#        b[k] = tmp
#        print(b)
#    t = ''.join(b)
#    return t


def reverse_list(letters, first=0, last=None):
    "reverses the elements of a list in-place"
    if last is None:
        last = len(letters)
    last -= 1
    while first < last:
        letters[first], letters[last] = letters[last], letters[first]
        first += 1
        last -= 1

def reverse_words(string):
    """reverses the words in a string using a list, with each character
    as a list element"""
    characters = list(string)
    reverse_list(characters)
    first = last = 0
    while first < len(characters) and last < len(characters):
        if characters[last] != ' ':
            last += 1
            continue
        reverse_list(characters, first, last)
        last += 1
        first = last
    if first < last:
        reverse_list(characters, first, last=len(characters))
    return ''.join(characters)


#Q. Given a string s, return all the palindromic permutations (without duplicates) of it. Return an empty list if no palindromic permutation could be form.
# For example:
# Given s = "aabb", return ["abba", "baab"].
# Given s = "abc", return [].

# Approach 1: Generate all the permutations which have len(word) and check if each of them is a palindrome

#Approach 2: Smarter way of doing approach1 (backtracking)
# 1. create the odd/even palindrome check function which checks if every permutation of the same length is indeed
#  a permutation
# 2. Now to create the permutations of say aabb, create a list of length 2 (half of 4) and add the distinct
# characters in it,
# so it will contain 'a' and 'b'.
# now create permutations of this new set and then add the reverse of it to its end; This prevents us from listing all unnecessary permutations which we know are not palindromes
# eg perm = ab, now add reverse ba to its end. We get abba as the permutation


#Q. One Edit Distance
#Given two strings S and T, determine if they are both one edit distance apart.

def oneEditDistance(s1, s2):
    # if the difference in length is more than 1, it can't be one edit distance
    if abs(len(s1) - len(s2)) > 1:
        return False
    #Initialising the no.of edits and the two pointers to traverse the strings
    num_edits = 0
    p1 = 0
    p2 = 0

    while p1<len(s1) and p2<len(s2):
        # either the two characters are same or different

        #Characters are different
        if s1[p1] != s2[p2]:
            #increment num_edits
            num_edits = num_edits + 1

            #if num_edits is greater than 1, return False
            if num_edits > 1:
                return False

            # if length difference is 1, then remove character from the longer string
            if len(s1) > len(s2):
                p1 = p1 + 1
            elif len(s1) < len(s2):
                p2 = p2 + 1
            #If both have the same length then just remove the character that doesn't match, since the rest will match
            else:
                p1 = p1 + 1
                p2 = p2 + 1
        #if characters match, then increment both
        else:
            p1 = p1 + 1
            p2 = p2 + 1

        #In case all characters are same, check if the last character in a word is extra (after while loop which breaks when either p1 or p2 equals the length of the string)
        if p1<len(s1) or p2<len(s2):
            num_edits = num_edits + 1

        return num_edits==1

# Q. Reverse Words in a String II
# For example,
# Given s = "the sky is blue",
# return "blue is sky the".

# Approach: 1. reverse the entire string and then reverse every individual word O(n^2)
def reverseWord(w):
    s = list(w)
    p1 = 0
    p2 = len(s) - 1

    while p1<=p2:
        temp = s[p1]
        s[p1] = s[p2]
        s[p2] = temp
        p1 = p1 + 1
        p2 = p2 - 1
    return s

def reverseSentence(w):
    s = reverseWord(w)
    print(s)
    p1 = 0
    a = []
    for i in range(len(s)):
        if s[i] == ' ':
            print(i)
            a.append(reverseWord(s[p1:i]))
            p1 = i + 1  #set p1 to start of new word
    a.append(reverseWord(s[p1:len(s)]))
    return a


#Q. Given a string, we can "shift" each of its letter to its successive letter, for example: "abc" -> "bcd".
#  We can keep "shifting" which forms the sequence:
# "abc" -> "bcd" -> ... -> "xyz"
# Given a list of strings which contains only lowercase alphabets, group all strings that belong to the same shifting sequence.
#
# For example, given: ["abc", "bcd", "acef", "xyz", "az", "ba", "a", "z"],
# A solution is:
#
# [
#   ["abc","bcd","xyz"],
#   ["az","ba"],
#   ["acef"],
#   ["a","z"]
# ]

# Approach: 1. abc, bcd, xyz belong to same group because the difference between the subsequent characters is the same
# eg in abc, b-a = 1 and c-b = 1. Similarly, in xyz, y-x = 1and z-y = 1. So, abc and xyz belong to the same group
# So, maintain a dictionary where key is the string of the difference and the
# value is the index of the word corresponding to that difference
# Hence, dict["aa"] because 1=a, 2=b, ...will contain [abc,bcd,xyz] which are the words with 11 as the differeneces between the characters

def differenceWord(word):
    res=""
    for i in range(1,len(word)):
        difference = ord(word[i-1]) - ord(word[i])
        #In case of overflow eg zab
        if difference<0:
            difference = difference + 26

        res = res + chr(difference + ord('a'))
    return res

def groupWords(arr):
    #Create a dictionary with keys as strings and values as list
    d = collections.defaultdict(list)

    for word in arr:
        diff = differenceWord(word)
        d["diff"].append(word)

    for key in d.keys():
        for word in d[key]:
            print(word)


# Q. Print maximum number of A’s using given four keys
# Imagine you have a special keyboard with the following keys:
# Key 1:  Prints 'A' on screen
# Key 2: (Ctrl-A): Select screen
# Key 3: (Ctrl-C): Copy selection to buffer
# Key 4: (Ctrl-V): Print buffer on screen appending it
#                  after what has already been printed.
#
# If you can only press the keyboard for N times (with the above four
# keys), write a program to produce maximum numbers of A's. That is to
# say, the input parameter is N (No. of keys that you can press), the
# output is M (No. of As that you can produce).
#
# Input:  N = 3
# Output: 3
# We can at most get 3 A's on screen by pressing
# following key sequence.
# A, A, A
#
# Input:  N = 7
# Output: 9
# We can at most get 9 A's on screen by pressing
# following key sequence.
# A, A, A, Ctrl A, Ctrl C, Ctrl V, Ctrl V
#
# Input:  N = 11
# Output: 27
# We can at most get 27 A's on screen by pressing
# following key sequence.
# A, A, A, Ctrl A, Ctrl C, Ctrl V, Ctrl V, Ctrl A,
# Ctrl C, Ctrl V, Ctrl V


# Approach 1: Recursion
# 1. For all N<7, the result is N
# 2. An optimal sequence which produces maximum As will contain (ctrlA, ctrlC) followed only by ctrlVs
# 3. We have to find the position in the sequence after which we can acheive step 2
# 4. The minimum number of ctrlVs we can have after (ctrlA, ctrlC) is 1
# 5. So, we iterate through every position starting from N-3 to 1 and compute the number of As which can be acheived
# 6. All we are doing is starting with the assumption that (ctrlA, ctrlC, ctrlV) are placed at the end 3 places of the sequence
# and we are just shifting them 1 place to the left every time till we hit the position that gives us the maximum As

def findMaxAs(n):
    #end condition
    if n<7:
        return n

    maxAs = 0
    #iterating through all positions starting from n-3
    for i in reversed(range(1,n-3)):
        #recursion
        current_num =(n-i-1)*findMaxAs(i)
        if current_num > maxAs:
            maxAs = current_num
    return maxAs

# Approach 2: We are recomputing the break point for every position in the sequence
# We can instead store results for every position we have computed in a table and find the max As in a bottom up approach

def findMaxAsv2(n):
    # end condition
    if n < 7:
        return n
    # array to store results
    results = []

    # filling up initial values in the array
    for i in range(1,7):
        results[i] = i

    for m in range(7,n):
        results[m-1] = 0
        for j in reversed(range(1,n-3)):
            curr_num = (m-j-1)*results[j-1]
            if curr_num > results[j-1]
                results[j-1] = curr_num
    return results[n-1]

#Q. The API: int read4(char *buf) reads 4 characters at a time from a file.
# The return value is the actual number of characters read. For example, it returns 3 if there is only 3 characters left
# in the file.
# By using the read4 API, implement the function int read(char *buf, int n) that reads n characters from the file.
#
# Note:
# The read function may be called multiple times.

#Assumption that read4(string) is pre defined
def read(destination_buffer, n):
    #initialising all parameters
    buffer = [] #intermediate buffer
    total_chars = 0
    offset = 0
    eof = False
    chars_in_buffer = 0

    while not eof and total_chars<n:
        if chars_in_buffer == 0:
            chars_in_buffer = read4(buffer)
            eof = chars_in_buffer<4
        num_chars_used = min(chars_in_buffer, n - total_chars)
        for i in range(num_chars_used):
            destination_buffer[total_chars+i] = buffer[offset+i]
        total_chars += num_chars_used
        chars_in_buffer -= num_chars_used
        offset = (offset+num_chars_used)%4
    return total_chars


# Q. Shortest Distance I
# Given a list of words and two words word1 and word2, return the shortest distance between these two words in the list.
# For example, Assume that words = ["practice", "makes", "perfect", "coding", "makes"].
# Given word1 = "coding", word2 = "practice", return 3. Given word1 = "makes", word2 = "coding", return 1.

# Approach: One Pass O(n)
def shortestdistance1(words, word1, word2):
    index1 = -1
    index2 = -1
    min_dist = float('inf')

    for i in range(len(words)):
        if words[i] == word1:
            index1 = i
        if words[i] == word2:
            index2 = i
        if index1 != -1 and index2 != -1:
            if min_dist > abs(index1, index2)
                min_dist = abs(index1, index2)

    return min_dist

# Q. Shortest Distance II
# Extension of the first part that is the array of words is fixed but can be called multiple times for different words
class wordDistance:
    def __init__(self, words):
        # Create an array that holds the index of every word in words to have a O(1) lookup
        self.word_index = collections.defaultdict(list)
        for i in range(len(words)):
            self.word_index[words[i]].append(i)

    def shortest_distance(self, word1, word2):
        i = j = 0
        min_distance = float('inf')

        while i<(len(self.word_index[word1])) and j<(len(self.word_index[word2])):
            min_distance = min(min_distance, abs(self.word_index[word1][i], self.word_index[word2][j]))
            if self.word_index[word1][i]<self.word_index[word2][j]:
                i = i+1
            else:
                j = j+1
        return min_distance

# Q. Given a string s and a dictionary of words dict, determine if s can be segmented into a space-separated sequence of
# one or more dictionary words.

# For example, given
# s = "leetcode",
# dict = ["leet", "code"].
#
# Return true because "leetcode" can be segmented as "leet code".

# Approach1 : Naive O(n^2)
def wordBreakI(s, dict):
    return wordBreakIUtil(s, dict, 0)

def wordBreakIUtil(s, dict, start_pos):
    # If only 1 charater
    if start_pos == len(s):
        return True
    # Check every word in the dictionary
    for word in dict:
        length = len(word)
        end_pos = start_pos + length
        # If the end_pos is greater than the length of the string then the current word wasn't found and exit the loop to check for other words
        if end_pos > len(s):
            continue
        # If the current word is found then recursively call function on the next position
        if s[start_pos : end_pos+1] == word:
            if wordBreakIUtil(s, dict, start_pos+length):
                return True
    return False

# Approach 2: Assume the entire dictionary is entered into a trie
# now at any given moment 'i' the string can be treated as two substrings s[0:i] and s[i:]
# So, we search for s[0:i] in the trie and recursively call the function on s[i:]


# Check if 's' is a substring of 't'
# Approach 1: Naive O(n^2) two for loops checking every element of both strings
# Approach 2: O(n) using hashing (Rabin Karp)

def check_substring(t, s):
    # No substring found as s is longer than t
    if len(s) > len(t):
        return -1

    import functools
    BASE = 26

    # Hashes of first len(s) substring
    # eg p has an initial value of 0 (as mentioned by third argument)
    #p= (0xBASE) + first_char
    #p= (pxBASE) + second _char where the p used is (0xBASE) + first_char; p keeps accumulating

    hash_t = functools.reduce(lambda p, char: p*BASE + ord(char), t[:len(s)], 0)
    hash_s = functools.reduce(lambda p, char: p*BASE + ord(char), s, 0)

    # Needed for rolling hash; BASE raised to the highest index in s
    pow_s = BASE**max(len(s) - 1, 0)

    # Check the hashses, if tehy are equal then check the actual substring to be safe from hash collisions
    for i in range(len(s),len(t)):
        if hash_t == hash_s and t[i-len(s): i] == s:
            return i-len(s)

        # Rolling hash
        hash_t -= ord(t[i-len(s)])*pow_s
        hash_t = hash_t*BASE + ord(t[i])

    if hash_t == hash_s and t[-len(s):] == s:
        return len(t) - len(s)

    # Not a substring
    return -1

# Another implementation

def check_substring_v2(t,s):
    if len(s) > len(t):
        return False
    BASE = 3 #Any prime number

    # First set of hashes
    s_hash = 0
    for i in range(len(s)):
        s_hash += ord(s[i])*(BASE**i)

    t_hash = 0
    for i in range(len(t)):
        t_hash += ord(t[i])*(BASE**i)

    # Iterate through t and compare substrings with s
    for i in range(len(t)):
        # Check for hash match and substring equality in case of hash match to avoid collisions
        if t_hash == s_hash and t[i:i+len(s)] == s:
            return True

        # Rolling hash
        t_hash -= ord(t[i])
        t_hash = t_hash//BASE
        t_hash += ord(t[i+len(s)])*(BASE**(len(s)-1))
    # if no match
    return False