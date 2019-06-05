#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 26 14:57:51 2019

@author: alexander
"""

import numpy as np

from polar.polar_common import polar_hpw
from polar.polar import polar_encode

# Hardware functions
def R0():
    return 0

def C2(u0, u1):
    return (u0 ^ u1, u1)

def R1_2(L0):
    bits = L0 < 0
    return C2(bits[0], bits[1])

def F(L0, L1):
    if L0 == -1 or L1 == -1:
        print("Error in calling F")

    Z0_0 = np.full(3, -1)

    Z0_0[0] = abs(L0)
    Z0_0[1] = abs(L1)
    Z0_0[2] = min(Z0_0[0], Z0_0[1])

    if (L0 > 0) ^ (L1 > 0) == 0:  # The signs are the same
        sign = 1
    else:  # The signs are different
        sign = -1

    return sign * Z0_0[2]

def G(L0, L1, u):
    if not u in [0,1]:
        print("Error in calling G")
    if u == 1:
        sign = -1
    else:
        sign = 1

    return (L1 + sign * L0)

def select(B):
    A = polar_hpw(N)[-K:]  # A is assumed placed in ROM

    u_hat = np.full(K, -1, np.int8)
    for i in range(K):
        u_hat[i] = B[A[i]]

    return u_hat

# Configuration
N = 16
K = 3

# Input data
u = np.array([1, 0, 1])
# u = np.random.binomial(1, 0.5, K)

# Encode
x = polar_encode(N, K, u)

# Channel
L0 = np.zeros(x.shape)  # Probability that received bit is zero
L0[x == 0] = 0.9
L0[x == 1] = 0.1
LR0 = L0 / (1 - L0)    # Likeliky ratio = (prob. bit is zero) / (prob. bit is one)
LLR = np.log(LR0)

#u = np.array([1,0,0])
#x = np.array([1, 1, 0, 0, 1, 1, 0, 0], dtype=np.uint8)
#LLR = np.array([-2.19722458, -2.19722458, -2.19722458, -2.19722458,
#                 2.19722458,  2.19722458,  2.19722458,  2.19722458,
#                -2.19722458, -2.19722458, -2.19722458, -2.19722458,
#                 2.19722458,  2.19722458,  2.19722458,  2.19722458])

    # Memories
B = np.full(N, 0, np.int8)
Z0 = np.full(N//2, -1.0)
Z1 = np.full(2, -1, np.int8)



# Computations

# S1
for i in range(8):
    Z0[i] = G(LLR[2*i], LLR[2*i + 1], B[i])

# S2
for i in range(4):
    Z0[i] = G(Z0[2*i], Z0[2*i+1], B[8+i])

# S3
for i in range(2):
    Z0[4 + i] = F(Z0[2*i], Z0[2*i+1])

# S4
Z0[4] = G(Z0[4], Z0[5], B[12])
B[13] = Z0[4] < 0

# S5
Z1[0:2] = C2(B[12], B[13])

# S6
for i in range(2):
    Z0[i] = G(Z0[2*i], Z0[2*i+1], Z1[i])

# S7
B[14:16] = R1_2(Z0[0:2])

u_hat = select(B)

print("u:    ", u)
print("u_hat:", u_hat)