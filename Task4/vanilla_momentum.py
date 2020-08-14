# -*- coding: utf-8 -*-
"""
Created on Fri Aug 14 09:55:53 2020

@author: Shail Thakkar
"""

import numpy as np
import time

X = [0.5,2.5]
Y = [0.2,0.9]

def f(w,b,x):
    return 1.0/(1.0 + np.exp(-(w*x+b)))

def error(w,b):
    err = 0.0
    for x,y in zip(X,Y):
        fx = f(w,b,x)
        err += 0.5*(fx - y)**2
    return err
    
def grad_b(w,b,x,y):
    fx = f(w,b,x)
    return (fx-y)*(fx)*(1-fx)

def grad_w(w,b,x,y):
    fx = f(w,b,x)
    return (fx-y)*(fx)*(1-fx)*x

def do_gradient_descent():
    seconds = time.time()
    w,b,eta,max_epoch = -2,-2,1.0,1000
    for i in range(max_epoch):
        print("________________________________")
        print("Iteration No." + str(i))
        dw = 0
        db = 0
        for x,y in zip(X,Y):
            dw =+ grad_w(w,b,x,y)
            db =+ grad_b(w,b,x,y)
        print("The Weight is:")
        w = w - eta*dw
        print(w)
        print("The Bias is:")
        b = b - eta*db
        print(b)
        print("The Error is:")
        print(error(w,b))
    print("Time Taken = " , time.time() - seconds)

def momentum_gradient_descent():
    seconds = time.time()
    w,b,eta,max_epoch = -2,-2,1.0,1000
    gamma = 0.1
    w_old,b_old = 0,0
    for i in range(max_epoch):
        print("________________________________")
        print("Iteration No." + str(i))
        dw = 0
        db = 0
        for x,y in zip(X,Y):
            dw =+ grad_w(w,b,x,y)
            db =+ grad_b(w,b,x,y)
        print("The Weight is:")
        w = w - ((gamma * w_old)+(eta*dw))
        print(w)
        print("The Bias is:")
        b = b - ((gamma * b_old)+(eta*db))
        print(b)
        print("The Error is:")
        print(error(w,b))
    print("Time Taken = " , time.time() - seconds)

            
momentum_gradient_descent()
