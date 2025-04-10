import numpy as np
import baseNeuralReceranceModels as bnr
import simplifiedModel as sm

#seting up the model 

#MARK: model variables
# from table 1 
TAU = 0.020 # in ms
M= 200
N = M
B = np.eye(N, M)
C = np.eye(M,N) 
SIGMA = 1
KAPPA = np.pi / 4
ALPHA = sm.ALPHA
ALPHA_PRIME = sm.ALPHA_PRIME

#MARK: model parameters
theta = sm.theta

totalTimeSteps = sm.numSteps
#MARK: W setup

def W_gen():
    psiList = np.array([2 * np.pi * n / M for n in range(M)])
    W3_all = np.zeros((M, M))
    for i in range(M):
        for j in range(M):
            W3_all[i, j] = sm.V(psiList[i] - psiList[j],KAPPA)
    W3 = sm.R_recscale(W3_all, ALPHA)
    return W3

W = W_gen() #recurrent connection strengths

#MARK: model running

#dec, error, r = bnr.bigOutput( C,M, W, B, N)
r = sm.dynamics(N, M, W, B, totalTimeSteps, TAU, theta, KAPPA, TIME_STEP=0.001)

def decodeError(stepNum, theta):
    thetaDecoded, error = sm.decode(theta, stepNum, C, r, M, KAPPA)
    return error
