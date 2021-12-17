# -*- coding: utf-8 -*-
"""
Created on Fri Dec 17 12:42:43 2021

@author: olive
Test script
"""
print('Hello World')

f = open("demofile2.txt", "a")
f.write("Now the file has more content!")
f.close()

#open and read the file after the appending:
f = open("demofile2.txt", "r")
print(f.read())
