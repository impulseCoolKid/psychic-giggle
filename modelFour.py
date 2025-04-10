import numpy as np
import simplifiedModel as sm
#seting up the model 

#MARK: model variables
# from table 1 
TAU = 0.020 # in ms
M= 200
N = 2*M
B = np.block([[np.eye(M, M)],
              [np.zeros((M, M))]])
C = np.block([np.eye(M, M),np.zeros((M, M))])
SIGMA = 1
KAPPA = np.pi / 4
ALPHA = sm.ALPHA
ALPHA_PRIME = sm.ALPHA_PRIME

#MARK: THIS MODEL DIRECT SETUP


#MARK: W setup

def W_gen(ALPHA=ALPHA_PRIME):
    psiList = np.array([2 * np.pi * n / M for n in range(M)])
    W3_all = np.zeros((M, M))
    for i in range(M):
        for j in range(M):
            W3_all[i, j] = sm.V(psiList[i] - psiList[j],KAPPA)
    W3 = sm.R_recscale(W3_all, ALPHA)
    return W3


w_corner = W_gen()

W_block = np.block([[w_corner, -w_corner],
              [w_corner, -w_corner]])

W =  W_block 



#MARK: model running
# argument = theta, stepnum, C, r,M, W, B, N
theta = sm.theta
totalTimeSteps = sm.numSteps


#dec, error, r = bnr.bigOutput( C,M, W, B, N)
r = sm.dynamics(N, M, W, B, totalTimeSteps, TAU, theta, KAPPA, TIME_STEP=0.001)

def decodeError(stepNum, theta):
    thetaDecoded, error = sm.decode(theta, stepNum, C, r, M, KAPPA)
    return error

#MARK: Q6
w_corner_6 = W_gen(5)
W_block_6 = np.block([[w_corner_6, -w_corner_6],
              [w_corner_6, -w_corner_6]])

W_6 =  W_block_6
r_Q6 = sm.dynamics(N, M, W_6, B, totalTimeSteps, TAU, theta, KAPPA, TIME_STEP=0.001)

def decodeErrorQ6(stepNum, theta):
    thetaDecoded, error = sm.decode(theta, stepNum, C, r_Q6, M, KAPPA)
    return error

#MARK: Q7
B_Q7 = np.block([[np.eye(M, M)],
              [-np.eye(M, M)]])

r_Q7 = sm.dynamics(N, M, W, B_Q7, totalTimeSteps, TAU, theta, KAPPA, TIME_STEP=0.001)

def decodeErrorQ7(stepNum, theta):
    thetaDecoded, error = sm.decode(theta, stepNum, C, r_Q6, M, KAPPA)
    return error


multiple = 1.68
B_Q7_b = np.block([[multiple*w_corner],[-multiple*w_corner]]) 

r_Q7_b = sm.dynamics(N, M, W, B_Q7_b, totalTimeSteps, TAU, theta, KAPPA, TIME_STEP=0.001)

def decodeErrorQ7b(stepNum, theta):
    thetaDecoded, error = sm.decode(theta, stepNum, C, r_Q7_b, M, KAPPA)
    return error


#MARK: Q8
alpha8 = 100000000099999999
alpha8b = 16
#part a
w_corner_8 = W_gen(alpha8)
W_block_8 = np.block([[w_corner_8, -w_corner_8],
              [w_corner_8, -w_corner_8]])

W_8 =  W_block_8
r_Q8 = sm.dynamics(N, M, W_8, B, totalTimeSteps, TAU, theta, KAPPA, TIME_STEP=0.001)


#MARK: Q8 part b
w_corner_8b = W_gen(alpha8b)
W_block_8b = np.block([[w_corner_8b, -w_corner_8b],
              [w_corner_8b, -w_corner_8b]])

W_8b =  W_block_8b
r_Q8b = sm.dynamics(N, M, W_8b, B, totalTimeSteps, TAU, theta, KAPPA, TIME_STEP=0.001)