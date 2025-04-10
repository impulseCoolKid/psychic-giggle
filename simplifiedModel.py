import numpy as np

#MARK: W functions 
def R_recscale(w, alpha):
    eigenvalues = np.linalg.eigvals(w)
    # Find the largest eigenvalue magnitude (works for both real and complex eigenvalues)
    largest_eigenvalue = np.max(np.abs(eigenvalues))
    
    # Scale the matrix by alpha divided by the largest eigenvalue magnitude
    w_scaled = w * (alpha / abs(largest_eigenvalue))
    
    return w_scaled

#MARK: sub functions
def V(z,kappa):
    return np.exp((np.cos(z) - 1) / (kappa**2))

#MARK: MODEL
# step theta -> h 
def splitThetaIntoVector(theta, M, KAPPA):
    listOfDir =  np.array([2 * np.pi * n / M for n in range(M)])
    directions = V(listOfDir - theta, KAPPA)
    return directions
    
#step theta -> h -> r
def dynamics(N, M, W, B, totalTimeSteps, TAU, theta, KAPPA, TIME_STEP ):

    r = np.zeros((totalTimeSteps,N)) #momentary firing rate of V1 neuron i
    
    r[0] = (B @ splitThetaIntoVector(theta, M, KAPPA)) #* TIME_STEP/TAU #initial condition with scalling


    for step in range(1, totalTimeSteps):
        r_dot = (-r[step-1] + W @ r[step-1] )
        r[step] = r[step - 1] + r_dot * TIME_STEP/TAU 

    return r

#step O
def outO(stepnum, C, r, SIGMA = 1):
    return  C @ r[stepnum] + SIGMA * np.random.normal(0, 1, 200)

def decode(theta, stepnum, C, r, M, KAPPA):
    # Decode the orientation from the output
    o = outO(stepnum, C, r)
    listOfDir = [2 * np.pi * n / M for n in range(M)]

    top  = np.sum (o * np.sin(listOfDir))
    bottom = np.sum (o * np.cos(listOfDir))
    thetaDecoded = np.atan2(top, bottom)

    #error
    error = np.arccos(np.cos(thetaDecoded - theta))

    return (thetaDecoded, error)

#seting up the model 
#MARK: model variables
theta = np.pi
TIME_STEP = 0.001
TOTAL_TIME = 0.15
numSteps = int(TOTAL_TIME / TIME_STEP)


ALPHA = 0.9#5 #0.9
ALPHA_PRIME = 0.9#5
#MARK: model variables
