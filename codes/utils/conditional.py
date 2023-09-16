# -*- coding: utf-8 -*-
"""
Created on Fri Apr  1 20:16:54 2022

@author: alienware

attempt on wrtitign conditionals
"""
import numpy as np
import torch

def calcErrorConditional(serr):
    # conditional probability p(s|zhat)
    ones = torch.sum(serr).item()
    num_el = torch.numel(serr)
    # frequency_table = [0, (num_el-ones)/num_el, 1]
    frequency_table={0:torch.numel(serr)-ones, 1:ones}
    return frequency_table

def calcConditional(shat):
    # conditional probability p(s|zhat)
    frequency_table={-1:0, 1:0}
    for sign in torch.sign(shat).flatten():
        if sign.item()==0: # consider 0 as positive number
            frequency_table[1] += 1
        else:
            frequency_table[sign.item()] += 1
    return frequency_table

def estmConditional(zhat):
    frequency_table={-1:0, 1:0}
    for z in zhat:
        if z.item() >= 0:
            frequency_table[1] += 1
        else:
            frequency_table[-1] += 1
    return frequency_table
