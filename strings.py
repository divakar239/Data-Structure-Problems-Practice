#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 14 15:49:07 2018

@author: DK
"""

import math
import string

# Q1 Check if string is a palindrome

def check_palindrome(s):
    for i in range(0,len(s)-1):
        if(s[i] != s[len(s)-1-i]):
            print("Not Palindrome")
            return
    print("Palindrome")
    
### s[~i] is s[-(i+1)] that is ith element from the last
#Alternate

def is_palindrome(s):
    return all(s[i] == s[~i] for i in range(0, len(s)//2))
    
### Note: the any() function returns TRUE if all the vaues being compared are correct which means all of them yield a true when compared in an if statement.
### The check only runs till half of the string since that is all we need to get s[i] and s[~i]
### eg odd : "tit" ; i runs till the 't' at 0 index
### eg even : "boob" ; i runs till 'o' at 1 index
    

##### IMPT NOTE ON STRINGS #####
#strings are immutable objects that means:
    
    #s="boob" 
    #s=s+"tit" will create a new copy of the string "boob" and append "tit" to it. then make s point to this new string.

# Hence, the following code has O(n^2) as 'n' copies will be made.

def add_char(s,n,m):
    for i in range(0,m):
        s+=m
        
# Q2 function to implement int to string conversion

# idea is to use chr(int)
#use ascii trick, '0' -> 48 so add sum=ord("0")+character and then take chr(sum)
# int are not iterable so we can't access them as array elements as list(int), so we will filter out the last digit by int%10 and then apply the same procedure to int//10
def int_to_string(i):
    
    is_negative = False
    if i<0:
        i, is_negative = -i, 