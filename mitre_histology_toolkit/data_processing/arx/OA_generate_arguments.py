#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 20 18:22:58 2021

@author: sguan
"""
import os


output_path = './tiles/'
file_path = '../OA_slides/'

def stringFilter(list1, list2):
    return [n for n in list1 if
             any(m in n for m in list2)]

files = os.listdir(file_path)
files = stringFilter(files, ['.svs', '.czi'])
files = stringFilter(files, ['SYN'])

with open('OA_SYN_SLIDES.txt', 'w') as f:
    for item in files:
        f.write("%s\n" % item)