import numpy as np
import baseNeuralReceranceModels as bnr
import simplifiedModel as sm
#seting up the model 

#MARK: model variables
# from table 1 
TAU = 0.020 # in ms
M= 200
N = M
B = np.eye(M, M)
C = np.eye(M,N) 
SIGMA = 1
KAPPA = np.pi / 4
ALPHA = sm.ALPHA
ALPHA_PRIME = sm.ALPHA_PRIME

#MARK: model parameters

r = np.zeros((sm.numSteps,N)) #momentary firing rate of V1 neuron i
W = np.zeros((N,N)) #recurrent connection strengths


#MARK: model running
# argument =C,M, W, B, N
theta = sm.theta

totalTimeSteps = sm.numSteps

#dec, error, r = bnr.bigOutput( C,M, W, B, N)
r = sm.dynamics(N, M, W, B, totalTimeSteps, TAU, theta, KAPPA, TIME_STEP=0.001)

def decodeError(stepNum, theta):
    thetaDecoded, error = sm.decode(theta, stepNum, C, r, M, KAPPA)
    return error