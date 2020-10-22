"""
Project code+scripts for 8DC00 course.
"""

import numpy as np
import cad_util as util
import matplotlib.pyplot as plt
import registration as reg
import cad
import scipy
from IPython.display import display, clear_output
import scipy.io


def nuclei_measurement():

    fn = '../data/nuclei_data.mat'
    mat = scipy.io.loadmat(fn)
    test_images = mat["test_images"] # shape (24, 24, 3, 20730)
    test_y = mat["test_y"] # shape (20730, 1)
    training_images = mat["training_images"] # shape (24, 24, 3, 21910)
    training_y = mat["training_y"] # shape (21910, 1)

    montage_n = 300
    sort_ix = np.argsort(training_y, axis=0)
    sort_ix_low = sort_ix[:montage_n] # get the 300 smallest
    sort_ix_high = sort_ix[-montage_n:] #Get the 300 largest

    # visualize the 300 smallest and the 300 largest nuclei
    X_small = training_images[:,:,:,sort_ix_low.ravel()]
    X_large = training_images[:,:,:,sort_ix_high.ravel()]
    fig = plt.figure(figsize=(16,8))
    ax1  = fig.add_subplot(121)
    ax2  = fig.add_subplot(122)
    util.montageRGB(X_small, ax1)
    ax1.set_title('300 smallest nuclei')
    util.montageRGB(X_large, ax2)
    ax2.set_title('300 largest nuclei')

    # dataset preparation
    imageSize = training_images.shape
    
    # every pixel is a feature so the number of features is:
    # height x width x color channels
    numFeatures = imageSize[0]*imageSize[1]*imageSize[2]
    training_x = training_images.reshape(numFeatures, imageSize[3]).T.astype(float)
    test_x = test_images.reshape(numFeatures, test_images.shape[3]).T.astype(float)

    ## training linear regression model
    #---------------------------------------------------------------------#
    # TODO: Implement training of a linear regression model for measuring
    # the area of nuclei in microscopy images. Then, use the trained model
    # to predict the areas of the nuclei in the test dataset.
    trainXones=util.addones(training_x)
    theta,_ = reg.ls_solve(trainXones,training_y)
    
    predicted_y=util.addones(test_x).dot(theta)
    
    # Computation of the error
    E_test = np.sum((predicted_y-test_y)**2)/np.shape(test_x)[0]
    print( 'Error of the model:', E_test)
    
    #---------------------------------------------------------------------#

    # visualize the results
    fig2 = plt.figure(figsize=(16,8))
    ax1  = fig2.add_subplot(121)
    line1, = ax1.plot(predicted_y, test_y, ".g", markersize=3)
    ax1.grid()
    ax1.set_xlabel('Area')
    ax1.set_ylabel('Predicted Area')
    ax1.set_title('Training with full sample')

    #training with smaller number of training samples
    #---------------------------------------------------------------------#
    # TODO: Train a model with reduced dataset size (e.g. every fourth
    # training sample).
    trainXones=trainXones[::4]
    training_y=training_y[::4]
    
    theta_smalldata,_ = reg.ls_solve(trainXones,training_y)
    
    predicted_y_smalldata=util.addones(test_x).dot(theta_smalldata)
    
    # Computation af the validation model with reduced samples
    E_test_reduced = np.sum((predicted_y_smalldata-test_y)**2)/np.shape(test_x)[0]
    print ('Error of model with reduced samples:' , E_test_reduced)
    #---------------------------------------------------------------------#

    # visualize the results
    ax2  = fig2.add_subplot(122)
    line2, = ax2.plot(predicted_y_smalldata, test_y, ".g", markersize=3)
    ax2.grid()
    ax2.set_xlabel('Area')
    ax2.set_ylabel('Predicted Area')
    ax2.set_title('Training with smaller sample')


def nuclei_classification(it, mu, batch, theta):
    #the inputs it, mu, batch, and theta are used to define the hyperparameters below
    
    ## dataset preparation
    fn = '../data/nuclei_data_classification.mat'
    mat = scipy.io.loadmat(fn)

    #reduction factor for the training set, set to 1 to use the full training set
    reduction = 1
    
    test_images = mat["test_images"] # (24, 24, 3, 20730)
    test_y = mat["test_y"] # (20730, 1)
    training_images = mat["training_images"][:,:,:,:int(14607*reduction)] # (24, 24, 3, 14607)
    training_y = mat["training_y"][:int(14607*reduction),:] # (14607, 1)
    validation_images = mat["validation_images"] # (24, 24, 3, 7307)
    validation_y = mat["validation_y"] # (7307, 1)

    ## dataset preparation
    imageSize = training_images.shape
    # every pixel is a feature so the number of features is:
    # height x width x color channels
    numFeatures = imageSize[0]*imageSize[1]*imageSize[2]
    training_x = training_images.reshape(numFeatures, training_images.shape[3]).T.astype(float)
    validation_x = validation_images.reshape(numFeatures, validation_images.shape[3]).T.astype(float)
    test_x = test_images.reshape(numFeatures, test_images.shape[3]).T.astype(float)

    # the training will progress much better if we
    # normalize the features
    meanTrain = np.mean(training_x, axis=0).reshape(1,-1)
    stdTrain = np.std(training_x, axis=0).reshape(1,-1)

    training_x = training_x - np.tile(meanTrain, (training_x.shape[0], 1))
    training_x = training_x / np.tile(stdTrain, (training_x.shape[0], 1))

    validation_x = validation_x - np.tile(meanTrain, (validation_x.shape[0], 1))
    validation_x = validation_x / np.tile(stdTrain, (validation_x.shape[0], 1))

    test_x = test_x - np.tile(meanTrain, (test_x.shape[0], 1))
    test_x = test_x / np.tile(stdTrain, (test_x.shape[0], 1))

    ## training linear regression model
    num_iterations = it
    mu = mu
    batch_size = batch
    Theta = theta*np.ones((training_x.shape[1]+1,1))

    xx = np.arange(num_iterations)
    loss = np.empty(*xx.shape)
    loss[:] = np.nan
    validation_loss = np.empty(*xx.shape)
    validation_loss[:] = np.nan
    test_loss = np.empty(*xx.shape)
    test_loss[:] = np.nan
    g = np.empty(*xx.shape)
    g[:] = np.nan

    fig = plt.figure(figsize=(8,8))
    ax2 = fig.add_subplot(111)
    ax2.set_xlabel('Iteration')
    ax2.set_ylabel('Loss (average per sample)')
    ax2.set_title('mu = '+str(mu))
    h1, = ax2.plot(xx, loss, linewidth=2) #'Color', [0.0 0.2 0.6],
    h2, = ax2.plot(xx, validation_loss, linewidth=2) #'Color', [0.8 0.2 0.8],
    ax2.set_ylim(0, 1.5)
    ax2.set_xlim(0, num_iterations)
    ax2.grid()

    text_str2 = 'iter.: {}, loss: {:.3f}, val. loss: {:.3f}'.format(0, 0, 0)
    txt2 = ax2.text(0.3, 0.95, text_str2, bbox={'facecolor': 'white', 'alpha': 1, 'pad': 10}, transform=ax2.transAxes)

    for k in np.arange(num_iterations):
        # pick a batch at random
        idx = np.random.randint(training_x.shape[0], size=batch_size)

        training_x_ones = util.addones(training_x[idx,:])
        validation_x_ones = util.addones(validation_x)
        test_x_ones = util.addones(test_x)

        # the loss function for this particular batch
        loss_fun = lambda Theta: cad.lr_nll(training_x_ones, training_y[idx], Theta)

        # gradient descent
        # instead of the numerical gradient, we compute the gradient with
        # the analytical expression, which is much faster
        Theta_new = Theta - mu*cad.lr_agrad(training_x_ones, training_y[idx], Theta).T
        
        #compute losses
        loss[k] = loss_fun(Theta_new)/batch_size
        validation_loss[k] = cad.lr_nll(validation_x_ones, validation_y, Theta_new)/validation_x.shape[0]
        test_loss[k] = cad.lr_nll(test_x_ones, test_y, Theta_new)/test_x.shape[0]
        
        # visualize the training
        h1.set_ydata(loss)
        h2.set_ydata(validation_loss)
        text_str2 = 'iter.: {}, loss: {:.3f}, val. loss={:.3f} '.format(k, loss[k], validation_loss[k])
        txt2.set_text(text_str2)

        Theta = None
        Theta = np.array(Theta_new)
        Theta_new = None
        tmp = None

        display(fig)
        clear_output(wait = True)
        plt.pause(.005)
        
    #save figure for future reference
    fig.savefig('results_cad/'+'it_'+str(it)+'_mu_'+str(mu)+'_batch_'+str(batch)+'_theta_'+str(theta)+'.png') 
    
    # save losses for future reference
    error_file = open("losses for each experiment.txt")
    error_list = error_file.readlines()
    error_file.close()
    
    error_list.append('it = '+str(it)+'   mu = '+str(mu)+'   batch = '+str(batch)+'   theta = '+str(theta)+'\n'+'  training loss: '+str(loss[-1])+'\n'+'  validation loss: '+str(validation_loss[-1])+'\n')
    error_list_new = "".join(error_list)
    
    error_file = open("losses for each experiment.txt", 'w')
    error_file.write(error_list_new)
    error_file.close()
    
    return test_loss[-1]
###############################################################################
####################### Experiments Logistic Regression #######################
# set standard values for parameters
mu = 0.001
batch = 30
theta = 1
it = 300

## The following lines of code were used to search for the optimal values of the hyperparameters
## The validation losses are written to the file 'losses for each experiment.txt'
## In this file, the validation losses were compared manually, and the parameter values for which the lowest 
## validation losses were computed, are selected
## These values should then be changed manually in this script

# rough search for optimal value of theta
for theta1 in [0.001, 0.01, 0.1, 0.2, 0.3, 0.6, 0.8, 1, 2, 3, 6, 8, 10]:
    nuclei_classification(it, mu, batch, theta1)
    
# fine search for optimal value of theta, searching based on results of rough search
for theta1 in [0.0001, 0.001, 0.002, 0.003, 0.004, 0.005, 0.006, 0.007, 0.008, 0.008, 0.3, 0.4, 0.5]:
    nuclei_classification(it, mu, batch, theta1)

# set standard value for theta to the found optimal value
theta = 0.003

# search for optimal value of the number of iterations
for it1 in [300, 350, 400, 450, 500, 550, 600, 650, 700]:
    nuclei_classification(it1, mu, batch, theta)
    
# set standard value for it to the found optimal value
it = 300;

# search for the optimal value of mu
for mu1 in [0.00001, 0.0001, 0.001, 0.01, 0.1]:
    nuclei_classification(it, mu1, batch, theta)

# set standard value for mu to the found optimal value
mu = 0.001

# search for optimal value of batch size
for batch1 in [10, 20, 30, 50, 75, 100]:
    nuclei_classification(it, mu, batch1, theta)

# set standard value of batch to the found optimal value
batch = 20

# perform the training with the selected optimal values, to get the classification accuracy, which is the test loss
print('The classificatoin accuracy for the final model is:')
print(nuclei_classification(it,mu,batch,theta))

##################### End of experiments Logistic Regression ##################
###############################################################################