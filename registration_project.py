"""
Registration project code.
"""

import numpy as np
import matplotlib.pyplot as plt
import registration as reg
import registration_util as util
from IPython.display import display, clear_output

# define data arrays that contain the unique parts of the filenames of the images
data_t1     = ['1_1_t1', '1_2_t1', '1_3_t1', 
               '2_1_t1', '2_2_t1', '2_3_t1',
               '3_1_t1', '3_2_t1', '3_3_t1']
data_t1_d   = ['1_1_t1_d', '1_2_t1_d', '1_3_t1_d', 
               '2_1_t1_d', '2_2_t1_d', '2_3_t1_d',
               '3_1_t1_d', '3_2_t1_d', '3_3_t1_d']
data_t2     = ['1_1_t2', '1_2_t2', '1_3_t2', 
               '2_1_t2', '2_2_t2', '2_3_t2',
               '3_1_t2', '3_2_t2', '3_3_t2']

def get_images(fixedID, movingID):
    # --------------------------------------------------------------------------------------
    # retrieves images and image paths
    #
    # Input:
    # fixedID:  unique part of filename of the fixed image
    # movingID: unique part of filename of the moving image
    #
    # Output:
    # I:        fixed image
    # I_path:   pathname of fixed image
    # Im:       moving image
    # Im_path:  pathname of moving image
    # --------------------------------------------------------------------------------------
    
    # combine strings to form the image path for each corresponding set of images in the fixed and moving set
    I_path  = 'C:/Users/20183816/Documents/BMT 2020-2021/8DC00 Medical Image Analysis/code/MIA-group-21/data/image_data/'+fixedID+'.tif'
    Im_path = 'C:/Users/20183816/Documents/BMT 2020-2021/8DC00 Medical Image Analysis/code/MIA-group-21/data/image_data/'+movingID+'.tif'
        
    # read in the fixed and moving images
    Im = plt.imread(Im_path)
    I = plt.imread(I_path)

    # return the images and paths
    return I, I_path, Im, Im_path

def get_error(T, X, Xm):
    # --------------------------------------------------------------------------------------
    # calculates generalization error of a registration (E = (TXm - X)^2)
    #
    # Input:
    # T:        transformation matrix that performs registration
    # X:        fixed image coordinates of target points
    # Xm:       moving image coordinates of target points
    #
    # Output:
    # E:        generalization error
    # --------------------------------------------------------------------------------------

    T = np.transpose(T)
    w1 = T[:,0]
    w2 = T[:,1]

    A = np.transpose(Xm)

    b1 = np.transpose(X[0,:])
    b2 = np.transpose(X[1,:])
    
    # calculate error for x- and y-coordinates separately
    E1 = np.transpose(A.dot(w1) - b1).dot(A.dot(w1) - b1)
    E2 = np.transpose(A.dot(w2) - b2).dot(A.dot(w2) - b2)    
    
    # calculate total error
    E = E1 + E2
    
    return E

def make_figure(If, Im, Imt):
    # --------------------------------------------------------------------------------------
    # makes a figure that contains subplots of fixed, moving and transformed image
    #
    # Input:
    # If:       fixed image
    # Im:       moving image
    # Imt:      transformed moving image
    #
    # Output:
    # fig:      figure with subplots
    # --------------------------------------------------------------------------------------
    
    fig, axes = plt.subplots(1,3)
    fig.figsize = [30,10]
    
    #plot the images
    axes[0].imshow(If)
    axes[0].set_title("Fixed image")
    
    axes[1].imshow(Im)
    axes[1].set_title("Moving image")
    
    axes[2].imshow(Imt)
    axes[2].set_title("Transformed moving image");
    
    return fig
        
def point_based_experiment(fixed, moving, exp, npairs = 3, dist = 'arb'):
    # ---------------------------------------------------------------------------------------
    # workflow per experiment to evaluate point-based registration
    # user input required: per set of images, first select a number of fiducial points, then select a number of target points
    # 
    # Input:
    # fixed:        array of unique parts of filenames of the fixed images, to be registered
    # moving:       array of unique parts of filenames of the moving images, to be registered
    # exp:          number of experiment
    # npairs:       number of fiducial and target point pairs to be selected, default value is 3
    # dist:         tells how far the selected points are from each other, default = 'arb' for arbitrary
    #
    # Output:
    # mean_error    mean regularization error of all sets of registered images
    # 
    # Files saved:
    # figure with fixed, moving and transformed image
    # list of regularization errors
    # ---------------------------------------------------------------------------------------
    
    # create list to store the regularization errors of all sets of images being registered
    error = []
    
    # for loop to carry out the experiment for each set of images in the dataset
    for i in range(len(fixed)):
        If, If_path, Im, Im_path = get_images(fixed[i], moving[i])
        
        # get fiducial points in both the fixed and moving image and store the coordinates in Xf and Xmf respectively
        Xf, Xfm = util.cpselect(If_path, Im_path, 'fiducial', npairs)
        # turn coordinates into homogeneous coordinates
        Xf = util.c2h(Xf)
        Xfm = util.c2h(Xfm)
        
        # get target points in both the fixed and moving image and store the coordinates in Xt and Xmt respectively
        Xt, Xtm = util.cpselect(If_path, Im_path, 'target', 3)
        # turn coordinates into homogeneous coordinates
        Xt = util.c2h(Xt)
        Xtm = util.c2h(Xtm)
        
        # define an affine homogeneous transformation matrix T based on the fiducial point pairs
        T, Ef = reg.ls_affine(Xf, Xfm)
        
        # singular matrices occur regularly when only 2 point pairs are selected, so to solve that problem:
        if npairs == 2:
            # add a small positive number to T to avoid numerical problems (singular matrices)
            EPSILON = 10e-10
            T += EPSILON
        
        # calculate regularization error and append it to list with errors of all sets of images
        E = get_error(T, Xt, Xtm)
        error.append(E)
        
        # transform the moving image
        Imt, Xft = reg.image_transform(Im, T)

        # create a figure to plot the results in. 
        fig = make_figure(If, Im, Imt)
        
        # title of figure describes experiment
        fig.suptitle("Point-based registration of "+moving[i]+" onto "+fixed[i]+", #point pairs = "+str(npairs))

        #save figure for future reference
        fig.savefig('results_point_based/'+moving[i]+'+'+fixed[i]+' nrPairs_'+str(npairs)+' dist_'+dist+'.png') 
    
    # save list error for future reference
    error_file = open("generalization errors for each experiment.txt")
    error_list = error_file.readlines()
    error_file.close()
    
    error_list.append('Experiment '+str(exp)+': '+str(error)+'\n')
    error_list_new = "".join(error_list)
    
    error_file = open("generalization errors for each experiment.txt", 'w')
    error_file.write(error_list_new)
    error_file.close()
    
    # calculate mean error and return it as result of the experiment
    mean_error = np.mean(error)
    return mean_error

############################################################################################
# WORKFLOW POINT_BASED REGISTRATION EXPERIMENTS

# experiment 1: registering T1 and T1 transformed, 2 fiducial and 3 target point pairs
print('Run experiment 1')
#error1 = point_based_experiment(data_t1, data_t1_d, 1, 2)
#print('The mean error of experiment 1 is: '+str(error1))

# experiment 2: registering T1 and T1 transformed, 3 fiducial and 3 target point pairs
print('Run experiment 2')
error2 = point_based_experiment(data_t1, data_t1_d, 2, 3)
print('The mean error of experiment 2 is: '+str(error2))

# experiment 3: registering T1 and T1 transformed, 4 fiducial and 3 target point pairs
print('Run experiment 3')
error3 = point_based_experiment(data_t1, data_t1_d, 3, 4)
print('The mean error of experiment 3 is: '+str(error3))

# experiment 4: registering T1 and T1 transformed, 3 fiducial and 3 target point pairs, fiducial points selected close to each other
print('Run experiment 4, select fiducial points close to each other')
error4 = point_based_experiment(data_t1, data_t1_d, 4, 3, dist = 'close')
print('The mean error of experiment 4 is: '+str(error4))

# experiment 5: registering T1 and T1 transformed, 3 fiducial and 3 target point pairs, fiducial points selected far away from each other
print('Run experiment 5, select fiducial points far away from each other')
error5 = point_based_experiment(data_t1, data_t1_d, 5, 3, dist = 'far')
print('The mean error of experiment 5 is: '+str(error5))

# experiment 6: registering T1 and T2, 3 fiducial and 3 target point pairs
print('Run experiment 6')
error6 = point_based_experiment(data_t1, data_t1_d, 6, 3)
print('The mean error of experiment 6 is: '+str(error6))

###########################################################################################

def intensity_based_registration_demo():
    # ---------------------------------
    # perfomes a intensity based registration
    # using different kinds of similarity measures
    # to images with the same -and/or different 
    # modalities.
    # ---------------------------------    
    
    # read the fixed and moving images
    # change these in order to read different images
    fileI='data/image_data/3_3_t1.tif'
    fileIM='data/image_data/3_3_t2.tif'
    
    I = plt.imread(fileI)
    Im = plt.imread(fileIM)

    # initial values for the parameters
    # we start with the identity transformation
    # most likely you will not have to change these
    #x = np.array([0., 0., 0.])

    # NOTE: for affine registration you have to initialize
    # more parameters and the scaling parameters should be
    # initialized to 1 instead of 0
    
    ###this means that x=np.array([0.,1.,1.,0.,0.,0.,0.])
    ### so make an if else statement
    
    similarity_function = reg.affine_corr #change this to one of the followings [reg.rigid_corr, reg.affine_corr, reg.affine_mi]
    
    
    if similarity_function == reg.rigid_corr:
        x = np.array([0., 0., 0.])
    else:
        x = np.array([0., 1., 1., 0., 0., 0., 0.])

    # the similarity function
    # this line of code in essence creates a version of rigid_corr()
    # in which the first two input parameters (fixed and moving image)
    # are fixed and the only remaining parameter is the vector x with the
    # parameters of the transformation
    #fun = lambda x: similarity_measure(I, Im, x, return_transform=False)
    fun = lambda x: similarity_function(I, Im, x)

    # the learning rate
    mu = 0.0004

    # number of iterations
    num_iter = 200

    iterations = np.arange(1, num_iter+1)
    similarity = np.full((num_iter, 1), np.nan)

    fig = plt.figure(figsize=(14,6))

    # fixed and moving image, and parameters
    ax1 = fig.add_subplot(121)

    # fixed image
    im1 = ax1.imshow(I)
    # moving image
    im2 = ax1.imshow(I, alpha=0.7)
    # parameters
    txt = ax1.text(0.3, 0.95,
        np.array2string(x, precision=5, floatmode='fixed'),
        bbox={'facecolor': 'white', 'alpha': 1, 'pad': 10},
        transform=ax1.transAxes)

    # 'learning' curve
    ax2 = fig.add_subplot(122, xlim=(0, num_iter), ylim=(0, 1))

    learning_curve, = ax2.plot(iterations, similarity, lw=2)
    #add titel to graph: 
    ax2.set_xlabel('Iteration')
    ax2.set_xlabel('Iteration')
    ax2.set_ylabel('Similarity')
    ax2.grid()

    # perform 'num_iter' gradient ascent updates
    for k in np.arange(num_iter):

        # gradient ascent
        g = reg.ngradient(fun, x)
        x += g*mu

        # for visualization of the result
        ## for both rigid and affine transformations --> make fun(x)
        S, Im_t, _ = fun(x) #(I, Im, x, return_transform=True) 

        clear_output(wait = True)

        # update moving image and parameters
        im2.set_data(Im_t)
        txt.set_text(np.array2string(x, precision=5, floatmode='fixed'))

        # update 'learning' curve
        similarity[k] = S
        learning_curve.set_ydata(similarity)

        display(fig)
        
        #add saving figure, so the graphics will be saved on your comp
        fig.savefig('3_3_t1+3_3_t2'+'mu='+str(mu)+'integer = '+str(num_iter)+'_affine_corr'+'.png')

