"""
Registration project code.
"""

import numpy as np
import matplotlib.pyplot as plt
import registration as reg
from IPython.display import display, clear_output
#add import util file, for cpselect
import registration_util as util


def point_based_registration_demo():
    # ---------------------------------
    # Append the following code in jupiter to run:
    #   %matplotlib inline
    #   import sys
    #   sys.path.append(..\code")
    #   from registration_project import point_based_registration_demo
    
    # point_based_registration_demo()
    # ---------------------------------
    
    # read the fixed and moving images, 2x T1 or T1&T2
    # change these in order to read different images
    I_path = '../data/image_data/1_1_t1.tif'
    Im_path = '../data/image_data/1_1_t1_d.tif'
    
    X, Xm = util.cpselect(I_path, Im_path)

    print('X:\n{}'.format(X))
    print('Xm:\n{}'.format(Xm))
    
    #gives a matrix: 
    #X:
    #[[x1 x2 x3]
    # [y1 y2 y3]]
    #Xm:
    #[[x1m x2m x3m]
    # [y1m y2m y3m]]
    
    #make homogenous coordinates of X and Xm
    X=util.c2h(X)
    Xm=util.c2h(Xm)


    #calculate the affine transformationmatrix and the fiducial registration error
    T, Efiducial = reg.ls_affine(X, Xm)
    
    #calculate the target registraion error
    #Etarget=point_based_error(I_path,Im_path,T)
    X, Xm = util.cpselect(I_path, Im_path)
    #make homogenous coordinates of X and Xm
    X=util.c2h(X)
    Xm=util.c2h(Xm)
    _, Etarget = reg.ls_affine(X,Xm)
    
    print(Efiducial)
    print(Etarget)
    
    #read image
    Im=plt.imread(Im_path)
    
    #Apply the affine transformation to the moving image
    It, Xt = reg.image_transform(Im, T)
    
    #plot figure
    fig = plt.figure(figsize=(12,5))
    fig, ax = plt.subplots()
    
    ax.set_title("Transformed image with error: Etarget={} Efiducial={}".format(Etarget,Efiducial))
    ax.set_xlabel('x')
    ax.set_ylabel('y')

    plt.imshow(It)
    
    

def intensity_based_registration_demo():
    # ---------------------------------
    # Append the following code in jupiter to run:
    #   %matplotlib inline
    #   import sys
    #   sys.path.append(..\code")
    #   from registration_project import intensity_based_registration_demo
    # intensity_based_registration_demo()
    # ---------------------------------    
    
    # read the fixed and moving images
    # change these in order to read different images
    I = plt.imread('../data/image_data/1_1_t1.tif')
    Im = plt.imread('../data/image_data/1_1_t1_d.tif')

    # initial values for the parameters
    # we start with the identity transformation
    # most likely you will not have to change these
    #x = np.array([0., 0., 0.])

    # NOTE: for affine registration you have to initialize
    # more parameters and the scaling parameters should be
    # initialized to 1 instead of 0
    
    ###this means that x=np.array([0.,1.,1.,0.,0.,0.,0.])
    ### so make an if else statement
    
    similarity_function = reg.rigid_corr #change this to one of the followings [reg.rigid_corr, reg.affine_corr, reg.affine_mi]
    
    
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
    mu = 0.001

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
        fig.savefig('name = '+str(mu)+'integer = '+ str(num_iter) + '.png')

