import modelOne
import modelTwo
import modelThree
import modelFour
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.colors as colors
import simplifiedModel as sm

#data collection 

#MARK: Q1

r1 = modelOne.r
r2 = modelTwo.r
r3 = modelThree.r
r4 = modelFour.r
def plotLineGraphAll():
    # Define the time indices (in ms) to plot
    time_indices = [0, 20, 60]
    
    # Define x-ticks for angles in terms of π
    xticks = [0, np.pi/2, np.pi, 3*np.pi/2, 2*np.pi]
    xtick_labels = ['0', 'π/2', 'π', '3π/2', '2π']
    
    # Define colors for each time index
    colors = ['blue', 'green', 'red']
    
    # Prepare a list of models with their labels
    models = [
        (r1, 'Model 1'),
        (r2, 'Model 2'),
        (r3, 'Model 3'),
        (r4, 'Model 4')
    ]
    
    # Create 2x2 subplots for the 4 models
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes = axes.flatten()
    
    for i, (model, label) in enumerate(models):
        ax = axes[i]
        m_model = model.shape[1]
        
        if label == 'Model 4':
            m_model_half = m_model // 2
            angles = np.linspace(0, 2 * np.pi, m_model_half, endpoint=False)
        else:
            angles = np.linspace(0, 2 * np.pi, m_model, endpoint=False)
        
        # Plot each time index on the same axes with a different color
        for j, t_idx in enumerate(time_indices):
            if label == 'Model 4':
                # For Model 4, plot both halves separately
                data_first_half = model[t_idx, :m_model_half]
                data_second_half = model[t_idx, m_model_half:]
                ax.plot(angles, data_first_half, color=colors[j], linestyle='-', label=f't = {t_idx} ms exhititory neurons')
                ax.plot(angles, data_second_half, color=colors[j],linestyle='--', label=f't = {t_idx} ms inhibitory neurons')
            else:
                data = model[t_idx]
                ax.plot(angles, data, color=colors[j], label=f't = {t_idx} ms')
        
        ax.set_title(label)
        ax.set_xlabel('preferred orientations')
        ax.set_ylabel('r value')
        ax.set_xticks(xticks)
        ax.set_xticklabels(xtick_labels)
        ax.legend()
    
    plt.tight_layout()
    plt.show()
#plotLineGraphAll()

def plotLargestNeuronReponse():
    plt.figure(figsize=(10, 6))
    models = [(r1, 'Model 1'), (r2, 'Model 2'),(r3, 'Model 3'), (r4, 'Model 4 α′ = 5')]#(r1, 'Model 1'), (r2, 'Model 2'),(r3, 'Model 3')
    
    for r, label in models:
         neuron_index = 100 
         time_steps = np.arange(r.shape[0])
         plt.plot(time_steps, r[:, neuron_index], label=label)
    
    plt.xlabel('Time Step')
    plt.ylabel('Response')
    plt.title('Response of Neuron 100 Over Time for All Models')
    plt.legend()
    plt.tight_layout()
    plt.show()

#plotLargestNeuronReponse()
#MARK: Q2
#why does model 2 have a much noiser response than the rest 
w2 = modelTwo.W
def plotEigDecompOfWAngle(w):
    # Calculate eigenvalues and eigenvectors
    eigenvalues, eigenvectors = np.linalg.eig(w)
    
    # Sort eigenvalues and corresponding eigenvectors in descending order (largest eigenvalue first)
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]
    
    # Compute the angle differences between each eigenvector and the largest eigenvector
    v0 = eigenvectors[:, 0]
    angles_diff = []
    for i in range(eigenvectors.shape[1]):
        vi = eigenvectors[:, i]
        # Compute dot product (assuming eigenvectors are normalized) and clip to [-1, 1]
        dot_val = np.dot(vi, v0)
        dot_val = np.clip(dot_val, -1, 1)
        # Use the absolute value of the dot product to get the smallest angle between directions
        angle = np.arccos(np.abs(dot_val))  # angle in radians
        angles_diff.append(angle)
    angles_diff = np.array(angles_diff)
    
    # Normalize eigenvalues
    eigenvalues = eigenvalues / np.max(np.abs(eigenvalues))
    
    print("Eigenvalues:")
    print(eigenvalues)
    print("Angle differences (radians):")
    print(angles_diff)
    
    # Create a single plot for the scatter plot
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Plot scatter points for eigenvalue vs. angle difference
    for i in range(len(eigenvalues)):
        if i == 0:
            ax.scatter(np.real(eigenvalues[i]),angles_diff[i], color='red', s=20, label='Largest eigenvalue')
        else:
            ax.scatter( np.real(eigenvalues[i]),angles_diff[i], color='blue', s=5)
    
    ax.set_title('Eigenvalue vs. Angle Difference from Largest Eigenvector')
    ax.set_ylabel('Angle difference (radians)')
    ax.set_xlabel('Eigenvalue (Real part)')
    ax.legend()
    
    plt.tight_layout()
    plt.show()

def imagvsreal(w):
    # Calculate eigenvalues and eigenvectors
    eigenvalues, eigenvectors = np.linalg.eig(w)
    
    # Sort eigenvalues and corresponding eigenvectors in descending order (largest eigenvalue first)
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]
    
    # Create a single plot for the scatter plot
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Plot scatter points for real vs. imaginary parts of eigenvalues
    ax.scatter(np.real(eigenvalues), np.imag(eigenvalues), color='blue', s=5)
    
    ax.set_title('Real vs. Imaginary Parts of Eigenvalues')
    ax.set_ylabel('Imaginary Part')
    ax.set_xlabel('Real Part')
    
    plt.tight_layout()
    plt.show()

    #plotEigDecompOfWAngle(w2)
    #imagvsreal(w2)

    #MARK: Q3
    #imagvsreal(modelThree.W)
    #imagvsreal(modelFour.W)

def multiEigPlots():
    # Calculate eigenvalues for each matrix
    #eigenvalues1, _ = np.linalg.eig(w1)
    eigenvalues2, _ = np.linalg.eig(w2)
    eigenvalues3, _ = np.linalg.eig(modelThree.W)
    
    # Sort eigenvalues in descending order (largest eigenvalue first)
    #eigenvalues1 = eigenvalues1[np.argsort(eigenvalues1)[::-1]]
    eigenvalues2 = eigenvalues2[np.argsort(eigenvalues2)[::-1]]
    eigenvalues3 = eigenvalues3[np.argsort(eigenvalues3)[::-1]]
    
    # Create a single plot for the scatter plot
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Plot the real vs imaginary parts for each weight 
    alpha = 0.5
    #ax.scatter(np.real(eigenvalues1), np.imag(eigenvalues1), color='blue', s=10, label='w4', alpha=alpha)
    ax.scatter(np.real(eigenvalues2), np.imag(eigenvalues2), color='green', s=10, label='w2',alpha=alpha)
    ax.scatter(np.real(eigenvalues3), np.imag(eigenvalues3), color='red', s=10, label='w3',alpha=alpha)
    
    ax.set_title('Eigenvalues for Recurrent Matrices')
    ax.set_xlabel('Real Part')
    ax.set_ylabel('Imaginary Part')
    ax.legend()
    
    plt.tight_layout()
    plt.show()

#multiEigPlots()

def histEigValues(w):
    """Plot a histogram of the magnitudes of the eigenvalues of matrix w."""
    eigenvalues, _ = np.linalg.eig(w)
    magnitudes = np.abs(eigenvalues)
    
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.hist(magnitudes, bins=20, color='blue', alpha=0.7)
    ax.set_title('Histogram of Eigenvalue Magnitudes')
    ax.set_xlabel('Magnitude of Eigenvalue')
    ax.set_ylabel('Frequency')
    
    plt.tight_layout()
    plt.show()

#histEigValues(modelTwo.W)
#histEigValues(modelThree.W)
#MARK: Q4

#show the time course of decoding error for each model

def collateError(model,theta = np.pi):

    # Collect the decoding error for each model
    errors = []

    for stepnum in range(sm.numSteps):
        error = model.decodeError(stepnum, theta)
        errors.append(error)
    return np.array(errors)

def plotErrorOverTime():
    model1_errors = collateError(modelOne)
    model2_errors = collateError(modelTwo)
    model3_errors = collateError(modelThree)
    model4_errors = collateError(modelFour)
    time_steps = np.arange(sm.numSteps)

    # Create a figure and axis
    fig, ax = plt.subplots(figsize=(12, 6))
    # Plot the errors for each model
    ax.plot(time_steps, model1_errors, label='Model 1', color='blue')
    ax.plot(time_steps, model2_errors, label='Model 2', color='green')
    ax.plot(time_steps, model3_errors, label='Model 3', color='red')
    ax.plot(time_steps, model4_errors, label='Model 4', color='purple')
    # Set labels and title
    ax.set_xlabel('Time Step')
    ax.set_ylabel('Decoding Error')
    ax.set_title('Decoding Error Over Time for Different Models')
    ax.legend()
    # Show the plot
    plt.tight_layout()
    plt.show()

def errorsAdveraged():
    model1_errors = collateError(modelOne)
    model2_errors = collateError(modelTwo)
    model3_errors = collateError(modelThree)
    model4_errors = collateError(modelFour)

    mean1 = np.mean(model1_errors)
    mean2 = np.mean(model2_errors)
    mean3 = np.mean(model3_errors)
    mean4 = np.mean(model4_errors)

    print("Model 1 Mean Error:", mean1)
    print("Model 2 Mean Error:", mean2)
    print("Model 3 Mean Error:", mean3)
    print("Model 4 Mean Error:", mean4)

#plotErrorOverTime()
#errorsAdveraged()

#MARK: Q5
#alpha' is 5

#MARK: Q6
#alpha is 5
def collateErrorQ6(model,theta = np.pi):

    # Collect the decoding error for each model
    errors = []

    for stepnum in range(sm.numSteps):
        error = model.decodeErrorQ6(stepnum, theta)
        errors.append(error)
    return np.array(errors)


def plotErrorOverTimeQ6():
    model4_errorsQ6 = collateErrorQ6(modelFour)
    model4_errors = collateError(modelFour)
    time_steps = np.arange(sm.numSteps)

    # Create a figure and axis
    fig, ax = plt.subplots(figsize=(12, 6))
    # Plot the errors for each model
   
    ax.plot(time_steps, model4_errors, label='Original Model 4', color='purple')
    ax.plot(time_steps, model4_errorsQ6, label='Model 4 with new B', color='magenta')
    # Set labels and title
    ax.set_xlabel('Time Step')
    ax.set_ylabel('Decoding Error')
    ax.set_title('Decoding Error Over Time for Different Models')
    ax.legend()
    # Show the plot
    plt.tight_layout()
    plt.show()

#plotErrorOverTimeQ6()



#MARK: Q7
#MARK: Color Map for Model 4

def plotColorMapForW(w):
    plt.figure(figsize=(8, 6))
    plt.imshow(w, aspect='auto', cmap='viridis')
    plt.colorbar(label='Weight Value')
    plt.title('Color Map for Model 4 Weight Matrix')
    plt.xlabel('Column Index')
    plt.ylabel('Row Index')
    plt.tight_layout()
    plt.show()
#plotColorMapForW(modelFour.W)
#plotColorMapForW(modelFour.B_Q7_b)

def plotLargestNeuronReponseQ7():
    """Plot the response of neuron 100 for all models' r over all time values.
    If a model's response array has fewer than 101 neurons, use the last neuron available."""
    plt.figure(figsize=(10, 6))
    models = [(modelFour.r_Q7, 'Model 4, B = [[I],[-I]'), (r4, 'Model 4'),(modelFour.r_Q7_b, 'Model 4, B = [[W],[-W]]')]#(r1, 'Model 1'), (r2, 'Model 2'),(r3, 'Model 3')
    
    for r, label in models:
         neuron_index = 100 
         time_steps = np.arange(r.shape[0])
         plt.plot(time_steps, r[:, neuron_index], label=label)
    
    plt.xlabel('Time Step')
    plt.ylabel('Response')
    plt.title('Response of Neuron 100 for α′ = 0.9')
    plt.legend()
    plt.tight_layout()
    plt.show()
#plotLargestNeuronReponseQ7()


def collateErrorQ7(model,theta = np.pi):

    # Collect the decoding error for each model
    errors = []

    for stepnum in range(sm.numSteps):
        error = model.decodeErrorQ7(stepnum, theta)
        errors.append(error)
    return np.array(errors)

def collateErrorQ7b(model,theta = np.pi):

    # Collect the decoding error for each model
    errors = []

    for stepnum in range(sm.numSteps):
        error = model.decodeErrorQ7b(stepnum, theta)
        errors.append(error)
    return np.array(errors)

def plotErrorOverTimeQ7():
    model4_errorsQ7 = collateErrorQ7(modelFour)
    model4_errorsQ7b = collateErrorQ7b(modelFour)
    model4_errors = collateError(modelFour)

    time_steps = np.arange(sm.numSteps)

    # Create a figure and axis
    fig, ax = plt.subplots(figsize=(12, 6))
    # Plot the errors for each model
   
    ax.plot(time_steps, model4_errors, label=' Model 4', color='purple')
    ax.plot(time_steps, model4_errorsQ7, label='Model 4, B = [[I],[-I]]', color='green')
    ax.plot(time_steps, model4_errorsQ7b, label='Model 4 B = [[W],[-W]]', color='magenta')
    # Set labels and title
    ax.set_xlabel('Time Step')
    ax.set_ylabel('Decoding Error')
    ax.set_title('Decoding Error Over Time for Different Models')
    ax.legend()
    # Show the plot
    plt.tight_layout()
    plt.show()

#plotErrorOverTimeQ7()
#MARK: Q8
def plotLargestNeuronReponseQ8():
    plt.figure(figsize=(10, 6))
    models = [(r4, 'Model 4 α′ = {}'.format(sm.ALPHA_PRIME) ), (modelFour.r_Q8, 'Model 4 α′ = {}'.format(modelFour.alpha8)),(modelFour.r_Q8b, 'Model 4 α′ = {}'.format(modelFour.alpha8b))]
    
    for r, label in models:
         neuron_index = 88 
         time_steps = np.arange(r.shape[0])
         plt.plot(time_steps, r[:, neuron_index], label=label)
    
    plt.xlabel('Time Step')
    plt.ylabel('Response')
    plt.title('Response of Neuron 100 Over Time for All Models')
    plt.legend()
    plt.tight_layout()
    plt.show()

plotLargestNeuronReponseQ8()
