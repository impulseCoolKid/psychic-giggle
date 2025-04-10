import numpy as np
import simplifiedModel as sm


#seting up the model 

#MARK: model variables
# from table 1 
TAU = 0.020 # in ms
M= 200
N = M
B = np.eye(N, M)
C = np.eye(M,N) 
ALPHA = sm.ALPHA
ALPHA_PRIME = sm.ALPHA_PRIME
KAPPA = np.pi / 4

#MARK: THIS MODEL DIRECT SETUP

#MARK: W setup
W_random = np.load('wmodel2.npy')
W_symetric = W_random + W_random.T

W = sm.R_recscale(W_random + W_random.T , alpha=ALPHA) #recurrent connection strengths

#MARK: model running
# argument = theta, stepnum, C,M, W, B, N
theta = sm.theta
totalTimeSteps = sm.numSteps

#dec, error, r = bnr.bigOutput( C,M, W, B, N)
r = sm.dynamics(N, M, W, B, totalTimeSteps, TAU, theta, KAPPA, TIME_STEP=0.001)

def decodeError(stepNum, theta):
    thetaDecoded, error = sm.decode(theta, stepNum, C, r, M, KAPPA)
    return error







