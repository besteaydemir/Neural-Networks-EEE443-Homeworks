import sys
import numpy as np
import h5py
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import random
question = sys.argv[1]


def beste_aydemir_21703033_hw1(question):
    if question == '1':
        ##Question 1
        print(question)


    elif question == '2':
        #Part B
        #Generate binary inputs
        nums = np.array(range(16))
        bin_nums = ((nums.reshape(-1,1) & (2**np.arange(4))) != 0).astype(int)
        X = bin_nums[:,::-1].T
        print("Binary inputs:")
        print(X)

        #True output vector for all possible inputs
        true = np.array([[0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1]])
        print("True output vector:")
        print(true)

        #Found weight vector
        W1 = np.array([[1, 0, 1, 1], [0, -1, 1, 1], [-1,1,-1,0], [-1,1,0,-1]])
        t1 = np.array([[2],[1], [0], [0]])
        W2 = np.array([[1,1,1,1]])
        t2 = 0

        place = 0.1 #To add to t1 and t2 and define their place in the interval

        #Evaluating performance for a chosen network
        theta1 = t1 + place
        theta2 = t2 + place
        out = output(X,W1,W2,theta1,theta2)
        print("Network output and accuracy for theta1 = [%f, %f, %f, %f] and theta2 = %f:" %(theta1[0],theta1[1],theta1[2],theta1[3], theta2))
        print(out)
        print(accu(out,true))

        #Part c
        #Most robust decision boundary
        place = 0.5
        theta1 = t1 + place
        theta2 = t2 + place
        out = output(X,W1,W2,theta1,theta2)
        print(true.shape)
        print("Network output and accuracy for theta1 = [%f, %f, %f, %f] and theta2 = %f:" %(theta1[0],theta1[1],theta1[2],theta1[3], theta2))
        print(out)
        print(accu(out,true))

        #Part d
        np.random.seed(0)
        #Generating 400 input samples by concating each vector 25 times next to each other
        X_ext = np.repeat(X, [25], axis=1)

        true_ext = np.repeat(true, [25], axis=1)
        noise = np.random.normal(0, 0.2, size=(4, 25*16))
        X_noise = X_ext + noise

        macc = 0
        for i in range (100):
          place = i/100
          theta1 = t1 + place
          theta2 = t2 + place
          out = output(X_noise,W1,W2,theta1,theta2)
          acc = accu(out,true_ext)
          if (acc > macc):
            mtheta1 = theta1
            mtheta2 = theta2
            macc = acc
          if (i==50 or i == 20):
            print("Network output and accuracy for theta1 = [%f, %f, %f, %f] and theta2 = %f:" %(theta1[0],theta1[1],theta1[2],theta1[3], theta2))
            print(acc)

        print("Best network output and accuracy: theta1 = [%f, %f, %f, %f] and theta2 = %f:" %(mtheta1[0],mtheta1[1],mtheta1[2],mtheta1[3], mtheta2))
        print(macc)


    elif question == '3':
        print(question)
        plt.close('all')

        # Getting the data from assign1_data1 file
        filename = "assign1_data1.h5"
        with h5py.File(filename, "r") as f:
            trainims = np.array(f['trainims'])
            trainlbls = np.array(f['trainlbls'])
            testims = np.array(f['testims'])
            testlbls = np.array(f['testlbls'])


        # Part A
        # Plotting sample images from each letter class
        n = 26 #Number of classes
        m = 28*28 #Size of images
        axes = []   
        fig = plt.figure(figsize=(14, 20), dpi=80)
      
        for a in range(n):
            axes.append(fig.add_subplot(7, 4, a + 1))
            im = plt.imshow(np.transpose(trainims[200*a,:,:])) #The image is transposed 
        
        cb_ax = fig.add_axes([1, 0.1, 0.02, 0.8])
        cbar = fig.colorbar(im, cax=cb_ax)
        #plt.savefig('samples.png', dpi=100)      
        plt.tight_layout()     
        plt.show()
        
        # Displaying correlations in matrix format
        sample_ims = np.zeros((n,m)) #Add the sample images to the rows of a matrix
        for i in range(n):
          sample_ims[i,:] = np.reshape(trainims[200*i,:,:], (1,m))
        
        corr = np.corrcoef(sample_ims)

        fig, ax = plt.subplots(figsize=(16, 16), dpi=80)
        im = ax.imshow(corr)

        # Loop over data dimensions and create text annotations
        for i in range(n):
            for j in range(n):
                text = ax.text(j, i, float(np.round(corr[i,j], 2)),
                              ha="center", va="center", color="w")

        ax.set_title("Correlation Coefficients between Sample Images")
        plt.colorbar(im);
        #plt.savefig('corr.png', dpi=100)
        plt.show()


        # Part B
        random.seed(1)

        # Hyperparameter tuning
        lambda1 = [1, 1.05]
        eta = list(range(40))
        W_array = []
        mse_min = 1
        mse_best = []
        l_best = 0
        e_best = 0

        # for l in lambda1:
        #   for e in eta:
        #     print("Lambda", l)
        #     print("eta", e/80)
        #     W, mse, W_array = train(trainims, trainlbls, l, e/80)


        #     print(mse_min)
        #     print(mse[-1])
        #     #Best MSE
        #     if (mse_min > mse[-1]):
        #       mse_min = mse[-1]
        #       mse_best = mse
        #       l_best = l
        #       e_best = e/80
              
        l_best = 1
        e_best = 0.325
        W_best, mse_best, W_array = train(trainims, trainlbls, l_best, e_best)
        print("Lambda best" , l_best)
        print("Eta best" , e_best)


        # Showing W
        axes = []   
        fig = plt.figure(figsize=(14, 20), dpi=80)
        # fig.suptitle('Sample Images from Letter Classes')
      
        for a in range(n):
            axes.append(fig.add_subplot(7, 4, a + 1))
            imag =  W_best[a,0:m]
            imag = np.reshape(imag,(28,28)).T
            im = plt.imshow(imag) #The image is transposed 
        
        cb_ax = fig.add_axes([1, 0.1, 0.02, 0.8])
        cbar = fig.colorbar(im, cax=cb_ax)
        # plt.savefig('test3png.png', dpi=1000)      
        plt.tight_layout()     
        plt.show()

        # Part c
        # Plotting MSE for eta_low and eta_high
        W_low, mse_low, W_array = train(trainims, trainlbls, lambda1 = 1, eta = 0.0001)
        W_high, mse_high, W_array = train(trainims, trainlbls, lambda1 = 1, eta = 20)

        fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(20, 5))
        axes[0].plot( mse_low)
        axes[1].plot( mse_best)
        axes[2].plot( mse_high)
        plt.xlabel('Iteration')
        plt.suptitle('MSE')
        plt.show()
        fig.tight_layout()

        #Part d
        print('Test performance')
        mse_low, accs_low = classification_acc(testims,testlbls,W_low, lambda1 = 1)
        mse_best, accs_best = classification_acc(testims,testlbls,W_best, lambda1 = 1)
        mse_high, accs_high = classification_acc(testims,testlbls,W_high, lambda1 = 1)
        print("Average mse (over samples) for eta_low:", mse_low)
        print("Test accuracy for eta_low:", accs_low)
        print("Average mse (over samples) for eta:", mse_best)
        print("Test accuracy for eta:", accs_best)
        print("Average mse (over samples) for eta_high:", mse_high)
        print("Test accuracy for eta_high:", accs_high)




#For Question 3

#This function finds a network's output for an imageset and 
def classification_acc(ims,lbls,W,lambda1):
  #Find the output of the network
  size = ims.shape[0]
  m = 784
  X = np.reshape(ims.transpose(1,2,0), (m,size))
  X = X.astype('double')
  #X *= 1.0/np.linalg.norm(X, axis=0)
  X *= 1.0/np.max(X, axis=0)
  X_wo = np.append(X, -np.ones((1,size)) ,axis = 0) 

  V = np.matmul(W,X_wo)
  O =  1 / ( 1 + np.exp( -V*lambda1 ))

  D = np.zeros((size,26))
  a = np.array(lbls).astype('int')
  a -= 1
  D[np.arange(a.size),a] = 1
  
  mseavg = (np.sum((D.T-O)*(D.T-O), axis = 0)).mean() /26

  classno = np.argmax(O, axis=0)
  acc = ((lbls-1) == classno).mean()*100

  return mseavg, acc

# This function trains the network according to lambda1 and eta
def train(trainims, trainlbls, lambda1, eta):
  # # Generating weight matrix(784,26) including the beta vector(26,)
  m = 28*28
  n = 26
  sigma = 0.01
  mu = 0
  W = np.random.normal(mu, sigma, (m+1,n)) 
  W_T = np.transpose(W)

  # Online training with 10000 iterations
  iter_no = 10000
  mse_arr = np.zeros((iter_no,))
  W_arr = np.zeros((iter_no,n,m+1))

  for a in range(iter_no):
    im_index = random.randint(0, 5200-1) #randomly selecting x
    im_x =  np.reshape(trainims[im_index,:,:], (m,)) #image (784,)
    im_x = im_x.astype('double')
    im_x *= 1.0/np.max(im_x, axis=0)
    #im_x *= 1.0/np.linalg.norm(im_x)
    x = np.append(im_x, -1) #append -1 for beta

    V = np.matmul(W_T,x) #(26,785)
    V2 = V*lambda1
    O = 1 / ( 1 + np.exp( -V2 ))
    
    O = np.reshape(O, (n,))
    
    D = np.zeros((n,))
    indexlbl = int(trainlbls[im_index]-1)
    D[indexlbl] = 1 

    miss = D - O 
    mse = (sum(miss*miss))/n 
    mse_arr[a] = mse
    f_prime = (lambda1/2) * (1-O*O)
    f_prime = O*(1-O)*lambda1
    
    learning_sig = np.multiply(miss,f_prime)

    x = np.reshape(x, (1,m+1)) 

    learning_sig_exp = np.tile(learning_sig, (m+1, 1)).T
    x_exp = np.tile(x, (n, 1))

    delta_W = eta * np.multiply(learning_sig_exp,x_exp)
    W_T = W_T + delta_W
    W_arr[a,:,:] = W_T 

  return W_T, mse_arr, W_arr

#For Question 2
#This function gives the logic output for X(4,samples) given W1, W2, t1, t2
def output(X, W1, W2, t1, t2):
  v1 = np.matmul(W1,X)
  o1 = ((v1 - t1) >= 0).astype('uint8')
  v2 = np.matmul(W2,o1) 
  o2 = ((v2 - t2) >= 0).astype('uint8')
  return o2

#This function returns the accuracy between two arrays as percent
def accu(o1,true):
  return np.sum((o1 == true).astype('uint8'))/(true.shape[1])*100


beste_aydemir_21703033_hw1(question)

