#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan  4 20:19:25 2018

@author: DK
"""
import math

### Recursion

# Prob 1 : Calculate sum of list of numbers
def sum_list(array):
    
    #end condition
    if len(array) == 1:
        return array[0]
    #recursion
    else:
        return array[0] + sum_list(array[1:])

#Prob 2: Convert an integer in base 10 to a string in any base

def convert_to_base(num, base):
    to_string = "0123456789abcdef"
#end condition
    if num<base:
        return to_string[num]

#recursion
    else:
        return convert_to_base(num//base, base) + to_string[num%base]      

# the // operator in python is Floor division -> the quotient is reported after the decimal is removed
# this is done so that the indices of the to_string are integers

  
#Prob 3: Write a Python program of recursion list sum. 
#Test Data: [1, 2, [3,4], [5,6]]
#Expected Result: 21

def sum_list(num_list):
    result = 0
    for i in range(0,len(num_list)):
        if type(num_list[i]) == type([]):
            result = result + sum_list[num_list[i]]
        else:
            result = result + num_list[i]
    return result
    
#Note in the above function we used to check th type of the element of the num_list to see if it is a list.
#This is because using len() on integrs is invalid


#Prob 4: Get factorials of a number

def get_factorial(num):
    if num <= 1:
        return 1
    else:
        return get_factorial(num-1) * num

#Prob 5: Fibonacci numbers

def get_fibonacci(num):
    if num == 1 or num == 2:
        return 1
    else:
        return get_fibonacci(num-1) + get_fibonacci(num-2)

#Prob 6: Write a Python program to get the sum of a non-negative integer
# sumDigits(345) -> 12
# sumDigits(45) -> 9 

def get_num_sum(num):
    result = 0
    for element in list(str(num)):
        result += int(element)
    return result

def get_num_sum_recur(num):
    if num == 0:
        return 0
    else:
        return get_num_sum_recur(int(num/10)) + num%10

#Prob 7: Write a Python program to calculate the sum of the positive integers of n+(n-2)+(n-4)... (until n-x =< 0).
#sum_series(6) -> 12
#sum_series(10) -> 30

def sum_series(num):
    if num <= 0:
        return 0
    else:
        return sum_series(num-2) + num

#Prob 8: Write a Python program to calculate the harmonic sum of n-1.
def harmonic_sum(num):
    if num == 1:
        return 1
    else:
        return harmonic_sum(num-1) + 1/num

#Prob 9: Write a Python program to calculate the geometric sum of n-1
def geometric_sum(num):
    if num<0:
        return 1
    else:
        return geometric_sum(num-1) + 1/(pow(2,num-1))

    
#Prob 10: Write a Python program to calculate the value of 'a' to the power 'b'. Go to the editor
#Test Data : 
#(power(3,4) -> 81 
def get_power(num,power):
    if power == 1:
        return num
    elif power == 0:
        return 1
    else:
        return num*get_power(num, power-1)
        
#Prob 11: Write a Python program to find  the greatest common divisor (gcd) of two integers.
def get_gcd(a,b):
    low = min(a,b)
    high = max(a,b)
    
    if high%low == 0:
        return low
    else:
        return get_gcd(high%low,low)

        
    
     