# -*- coding: utf-8 -*-
"""MDAProject2.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1Jc7E_hE_p0IoIe3hJlECgCifLuzMIm1q
"""

## Matrix Decompositions in Data Analysis
## Winter 2021
## Assignment file Project 2 Non Negative Matrix Decomposition
## Name:: Abhishek N. Singh
## Student ID:322079
#
#Date: 13th February 2021
#Email: abhishek.narain@iitdalumni.com
#Description: The script below does NMF analysis as part of the course Matrix Decomposition for Data Analysis at 
#University of Eastern Finland

#Task 1: ALS vs. multiplicative NMF
#Download the data (news.csv) and the template code files assignment2.{m, py, R} from the course
#Moodle page (provided as a single zip-file assignment2.zip). You can use code files as templates for your
#answers, and you can also find some helper functions and boilerplate code in there.
#Your first task is to implement three versions of the NMF algorithm:
#a) NMF based on alternating least squares
#b) Lee and Seung’s multiplicative NMF algorithm
#c) NMF via gradient descent using Oblique Projected Landweber (OPL) updates
#You can truncate negative values to zero. Your implementations should be reasonably efficient. As a
#convergence criterion, you can stop after 300 iterations. For initial solutions, you can use either matrices
#where elements are sampled uniformly at random form the unit interval, or use scaling based on the input
#matrix. But you have to use the same initial solution generation for all methods.
#The data is a sample of 2000 news articles of the 20-newsgroups dataset.1 Terms have been stemmed
#and very frequent and infrequent words have been removed. The data is given in form of an 2000 × 5136
#document-term matrix; entry (d, w) denotes the term frequency (tf) of word w in document d.
#Run the three NMF algorithms on the news data for k = 20. Compare the reconstruction errors and
#convergence rates. Notice that any two runs of the algorithm might result to very different outcomes,
#depending on the initial W and H. Also, the default 300 iterations might not be enough (or it might be too
#much) for the methods to converge. Play around with the number of re-starts and iterations.
#Analyse the convergence speed of the algorithms. Use either the number of iterations the algorithms take
#to reach error below some reasonable threshold (e.g. error that is less than 95 % of the best error you have
#got), or the wall-clock time it takes for them to reach that level. Is one of the methods clearly better than the
#other? Can you show statistically significantly faster (in iterations or in wall-clock time) convergence times?
#Give at least one plot of convergence rates (iteration or time vs. reconstruction error) for each method.
#Compare also the best reconstruction errors. Does any of the methods give statistically significantly
#lower reconstruction errors over different re-starts? In all these tests, use an appropriate statistical test.
#Based on your experiments, which one of the three methods you consider better for this data and why?
#Hint: The news data is reasonably large. It is advisable to start early enough with solving the assignment as
#the computations need some time to run. It’s also a good idea to start with a smaller sample to make sure
#your code actually works. Make sure you use the same computer if you do any wall-clock tests.

import numpy as np
from numpy.linalg import svd, norm, lstsq
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.metrics import normalized_mutual_info_score as nmi
from scipy.stats import zscore
from sklearn.decomposition import non_negative_factorization
from sklearn.decomposition import NMF

## Load the news data
A = np.genfromtxt('news.csv', delimiter=',', skip_header=1)

## To read the terms, just read the first line of news.csv
with open('news.csv') as f:
    header = f.readline()
    terms = [x.strip('"\n') for x in header.split(',')]

header

terms

## accelerated-Hierarchical Alternating Least Squares method for NMF (of Cichocki et al.)
W, H, n_iter = non_negative_factorization(A, n_components=2, init='random',solver='cd', random_state=0, verbose='int')

W

H

n_iter

#Now I use standerd ALS method
#Now I use the Alternating Least Square updates for W and H given A and an
#initiation random value of W and H which are updated using pseudo inverse and
# then the negative values are truncated. 
# Function that updates W and H using ALS
#LSTSQ: Return the least-squares solution to a linear matrix equation
#rcondfloat, optional Cut-off ratio for small singular values of a. 
#For the purposes of rank determination, singular values are treated as zero 
#if they are smaller than rcond times the largest singular value of a.
#The previous default of -1 will use the machine precision as rcond parameter
def nmf_als(A, W, H):
  # Update H
  # Solve the least squares problem: argmin_H ||WH - A||
  H = lstsq(W, A, rcond = -1)[0] #Note this is solving the same purpose
  # as that of H <- W(pseudo-inverse)*A
  # Set negative elements of H to 0 as we have been given permission for this
  H[H < 0] = 0
  # Update W
  # Solve the least squares problem: argmin_W.T ||H.TW.T - A.T||
  # This will do the same job as W <- A * H(pseudo-inverse)
  W = lstsq(H.T, A.T, rcond = -1)[0].T

  # Set negative elements of W to 0 as we have been given permission for this
  W[W < 0] = 0

  return (W, H)

#Now let me use the pseudo-inverse formula for updating H and W as in the 
#Lecture slides
def nmf_als2(A, W, H):
  # Update H
  H = np.linalg.pinv(W) @ A #PseudoInverse of W multiplied by A matrix
  #H = ( np.linalg.inv(W.T @ W) @ W.T ) @ A #Note this is solving the same purpose
  # as that of H <- W(pseudo-inverse)*A
  # Set negative elements of H to 0 as we have been given permission for this
  H[H < 0] = 0
  # Update W
  # This will do the same job as W <- A * H(pseudo-inverse)
  W = A @ np.linalg.pinv(H)

  # Set negative elements of W to 0 as we have been given permission for this
  W[W < 0] = 0

  return (W, H)

## Boilerplate for NMF
def nmf(A, k, optFunc=nmf_als, maxiter=300, repetitions=1):
    (n, m) = A.shape
    bestErr = np.Inf;
    for rep in range(repetitions):
        # Init W and H 
        W = np.random.rand(n, k) 
        H = np.random.rand(k, m)
        errs = [np.nan] * maxiter
        for i in range(maxiter):
            (W, H) = optFunc(A, W, H)
            currErr = norm(A - np.matmul(W, H), 'fro')**2
            errs[i] = currErr
        if currErr < bestErr:
            bestErr = currErr
            bestW = W
            bestH = H
            bestErrs = errs
    return (bestW, bestH, bestErrs)

## Boilerplate for NMF
def nmf4OPD(A, k, optFunc=nmf_opl, maxiter=300, repetitions=1):
    (n, m) = A.shape
    bestErr = np.Inf;
    for rep in range(repetitions):
        # Init W and H 
        W = np.random.rand(n, k) 
        H = np.random.rand(k, m)
        errs = [np.nan] * maxiter
        for i in range(maxiter):
            H = optFunc(A, W, H)
            W = (optFunc(A.T, H.T, W.T)).T
            currErr = norm(A - np.matmul(W, H), 'fro')**2
            errs[i] = currErr
        if currErr < bestErr:
            bestErr = currErr
            bestW = W
            bestH = H
            bestErrs = errs
    return (bestW, bestH, bestErrs)

#Now I call the function for NMF using ALS using k the inner dimensions of the matrices as 20
W_Als, H_Als, Errs = nmf(A, 20, optFunc=nmf_als, maxiter=300, repetitions=1)



W_Als

H_Als

Errs

#Now I call the function for NMF using ALS using k the inner dimensions of the matrices as 20
W_Als2, H_Als2, Errs2 = nmf(A, 20, optFunc=nmf_als2, maxiter=300, repetitions=1)

W_Als2

H_Als2

Errs2

#Lee and Seung's Multiplicative update

def LeeSeungMU(A, k, delta, num_iter, init_W=None, init_H=None, print_enabled=False):
    
    
    if print_enabled:
        print('Frobenius norm ||A - WH||_F')
        

    # Initialize W and H if we do not pass an initial value for W and H 
    if init_W is None:
        W = np.random.rand(np.size(A, 0), k)
    else:
        W = init_W

    if init_H is None:
        H = np.random.rand(k, np.size(A, 1))
    else:
        H = init_H
    errs = [np.nan] * num_iter
    # Decompose the input matrix
    for n in range(num_iter):
        # Note that .T is used to do a transpose of the matrix
        # Update H  
        #Numerator
        W_TA = W.T @ A # @= and @ are new operators introduced in Python 3.5 performing matrix multiplication. 
        #Denominator
        W_TWH = W.T @ W @ H + delta
        #Now multiplying each element at a time for H 
        for i in range(np.size(H, 0)): #Iterate through rows of H matrix
            for j in range(np.size(H, 1)): #Iterate throug columns of H matrix
                H[i, j] = H[i, j] * W_TA[i, j] / W_TWH[i, j] #Update every element of H matrix

        # Update W
        AH_T = A @ H.T #This is the numerator where H transpose is post multiplied to A matrix
        WHH_T =  W @ H @ H.T + delta #This is the denominator
        
        #Now multiplying each element at a time for W 
        for i in range(np.size(W, 0)): #Here I count the number of rows in W matrix
            for j in range(np.size(W, 1)): #Iterate through number of columns in W matrix
                W[i, j] = W[i, j] * AH_T[i, j] / WHH_T[i, j] #Update every element by multiplicative factor

        if print_enabled:
            errs[n] = np.linalg.norm(A - W @ H, 'fro')
            print("iteration " + str(n + 1) + ": " + str(errs[n]))

    return (W, H, errs)

W_ls, H_ls, errs = LeeSeungMU(A, 20, 0.000001, 300, init_W=None, init_H=None, print_enabled=True)

#We note that 300 iterations does not seem sufficient

W_ls, H_ls, errs = LeeSeungMU(A, 20, 0.000001, 3000, init_W=None, init_H=None, print_enabled=True)

#OPD Oblique Algorithm
def nmf_als2(A, W, H):
  # Update H
  H = np.linalg.pinv(W) @ A #PseudoInverse of W multiplied by A matrix
  #H = ( np.linalg.inv(W.T @ W) @ W.T ) @ A #Note this is solving the same purpose
  # as that of H <- W(pseudo-inverse)*A
  # Set negative elements of H to 0 as we have been given permission for this
  H[H < 0] = 0
  # Update W
  # This will do the same job as W <- A * H(pseudo-inverse)
  W = A @ np.linalg.pinv(H)

  # Set negative elements of W to 0 as we have been given permission for this
  W[W < 0] = 0

  return (W, H)

#The multiplicative update seems to be stuck at local minima as there are no major update in the drop in the forbenius score of the
#difference between A and the WH matrix product.

def nmf_opl(A, W, H):
  # Update H
  # Get step by NetaH
  NetaH = np.diag(1 / np.sum( (W.T @ W), axis = 1)) #Doing Row Sum and getting 
  #the reciprocal and then creating a diagonal matrix
  G = ( W.T @ W @ H ) - ( W.T @ A ) #This is the gradient matrix
  H = H - ( NetaH @ G ) #Updating H by NetaH steps times G
  # Set negative elements of H to 0 as we have been given permission for this
  H[H < 0] = 0
  # Update W
  # Get the step by NetaG
  #NetaW = np.diag(1 / np.sum( (H.T @ H), axis = 1)) #Doing Row Sum and getting
  # the reciprocal then getting the diagonal matrix
  #G = H.T @ ( ( H @ W) -  A ) #This is the gradient matrix
  #W = W - ( NetaW @ G )

  # Set negative elements of W to 0 as we have been given permission for this
  #W[W < 0] = 0

  return (H)

W_Opl, H_Opl, ErrsOpl = nmf4OPD(A, 20, optFunc=nmf_opl, maxiter=300, repetitions=1)

W_Opl

H_Opl

ErrsOpl

## To show the per-iteration error of Lee Seung Multiplicative Update
plt.plot(errs)
plt.xlabel('Iterations')
plt.ylabel('Squared Frobenius')
plt.title('Convergence of NMF Lee Seung MU')
plt.show()

## To show the per-iteration error of OPL
plt.plot(ErrsOpl)
plt.xlabel('Iterations')
plt.ylabel('Squared Frobenius')
plt.title('Convergence of NMF OPL')
plt.show()

## To show the per-iteration error of ALS
plt.plot(Errs2)
plt.xlabel('Iterations')
plt.ylabel('Squared Frobenius')
plt.title('Convergence of NMF ALS')
plt.show()

#Opinion: The method of Multiplicative Update by Lee and Seung gives the least error 
# in Forbenius norm between the matrix A and its factors W and H, and so it can be preferred.
# However, the steepest descent for convergence was seen in OPL.

# Test script 2D array   
arr = [[14, 17, 12, 33, 44],    
       [15, 6, 27, 8, 19],   
       [23, 2, 54, 1, 4,]]

print("Diagonal Matrix of 1 / RowSums: ", np.diag (1 / np.sum(arr, axis = 1)))

#Task 2: Analysing the data
#In this task we try to analyse the news data. Before proceeding further, normalize the data such that
#the sum of all entries in the data equals 1. Then use one of the methods you implemented in the first
#task to find k = 20 NMF of the data and study the top-10 terms of the right factor matrix H. Can you
#infer some “topics” based on these terms? Recall that the terms are stemmed. The topics can be very
#broad (e.g. “terms associated with sports”) and they might not be the ones of the newsgroups. Also,
#some factors might not correspond to any sensible topic. Argue why (or why not) you think the factors
#correspond to the topics you claim they do.
#Repeat the analysis with k = 5, 14, 32, 40. How do the results change with increased k? Can you name
#the single best rank for this data?
#Repeat the analysis, but this time using the generalized K–L divergence optimizing version of NMF
#(provided in utils.R). Do the results change? Are they better or worse? Is a different k better with K–L
#divergence than with Euclidean distance?

## Normalise the data before applying the NMF algorithms
B = A/sum(sum(A)) #

W_lsNorm, H_lsNorm, errsNorm = LeeSeungMU(B, 20, 0.000001, 300, init_W=None, init_H=None, print_enabled=True)



## To print the top-10 terms of the first row of H, we can do the following
h =  H_lsNorm[1,:]
ind = h.argsort()[::-1][:10]
for i in range(10): print("{}\t{}".format(terms[ind[i]], h[ind[i]]))

W_OplNorm, H_OplNorm, ErrsOplNorm = nmf4OPD(B, 20, optFunc=nmf_opl, maxiter=300, repetitions=1)

## To show the per-iteration error of Normalized Matrix using Opl Norm
plt.plot(ErrsOplNorm)
plt.xlabel('Iterations')
plt.ylabel('Squared Frobenius')
plt.title('Convergence of NMF OPL of Normalized Matrix')
plt.show()

## To print the top-10 terms of the first row of H, we can do the following
h =  H_OplNorm[1,:]
ind = h.argsort()[::-1][:10]
for i in range(10): print("{}\t{}".format(terms[ind[i]], h[ind[i]]))

"""**The topic over here seems to correspond to 'Study, Research, Tests, Production certainty of Food Science in which Steve was involved'.**"""

W_OplNormk5, H_OplNormk5, ErrsOplNormk5 = nmf4OPD(B, 5, optFunc=nmf_opl, maxiter=300, repetitions=1)

## To show the per-iteration error of Normalized Matrix using Opl Norm
plt.plot(ErrsOplNormk5)
plt.xlabel('Iterations')
plt.ylabel('Squared Frobenius')
plt.title('Convergence of NMF OPL of Normalized Matrix using k5')
plt.show()

## To print the top-10 terms of the first row of H, we can do the following
h =  H_OplNormk5[1,:]
ind = h.argsort()[::-1][:10]
for i in range(10): print("{}\t{}".format(terms[ind[i]], h[ind[i]]))





"""**The topic could be "Watch the 3rd sign on final night in a ride before getting hit by fan which can cause flame"**"""

W_OplNormk14, H_OplNormk14, ErrsOplNormk14 = nmf4OPD(B, 14, optFunc=nmf_opl, maxiter=300, repetitions=1)

## To print the top-10 terms of the first row of H, we can do the following
h =  H_OplNormk14[1,:]
ind = h.argsort()[::-1][:10]
for i in range(10): print("{}\t{}".format(terms[ind[i]], h[ind[i]]))



"""**The topic could be "Bike ride by while and black members"**"""

W_OplNormk32, H_OplNormk32, ErrsOplNormk32 = nmf4OPD(B, 32, optFunc=nmf_opl, maxiter=300, repetitions=1)

## To print the top-10 terms of the first row of H, we can do the following
h =  H_OplNormk32[1,:]
ind = h.argsort()[::-1][:10]
for i in range(10): print("{}\t{}".format(terms[ind[i]], h[ind[i]]))

"""**The topic could be "Research starting with letter W at a Unit has satellite signal on radio at green field"**"""

W_OplNormk40, H_OplNormk40, ErrsOplNormk40 = nmf4OPD(B, 40, optFunc=nmf_opl, maxiter=300, repetitions=1)

## To print the top-10 terms of the first row of H, we can do the following
h =  H_OplNormk40[1,:]
ind = h.argsort()[::-1][:10]
for i in range(10): print("{}\t{}".format(terms[ind[i]], h[ind[i]]))



"""**The topic could be "Canadian and American capital jet cost at Winnipeg during fall term and their insurance"**"""



"""With change in K the meanings change drastically. The best choice could be k=5 given that each words then have a much higher frequency. But then this is because k is small. Lets try with k =2 for instance. """

W_OplNormk2, H_OplNormk2, ErrsOplNormk2 = nmf4OPD(B, 2, optFunc=nmf_opl, maxiter=300, repetitions=1)

## To print the top-10 terms of the first row of H, we can do the following
h =  H_OplNormk2[1,:]
ind = h.argsort()[::-1][:10]
for i in range(10): print("{}\t{}".format(terms[ind[i]], h[ind[i]]))



"""**So we see that with decrease in k value, the frequency of each words increases. The best k rank to be chosen is very subjective. Typically, what can be done is that we can remove some elements from the A matrix and then calculate its factors W and H. Then, we can see that which of the W and H obtained via various values of k, help us get back the missing value in A. This would be computationally very expensive. However, in our case, the ground truth has been provided. So, we can look for that k value for which the ground truth can be obtained using the k that suits the best.**"""

import csv
with open('news_ground_truth.txt', 'r') as file:
    reader = csv.reader(file)
    for row in reader:
        print(row)

#Looks like we dont get much information from ground truth file, and so we cannot
#much of comment about the apt value for 'k'.



!pip install nimfa





#import nimfa #This package can be used to get K-L divergence and Euclidian divergence

#However, to get matrix decomposition, into W and H, we used K-L beta loss k=2
model = NMF(n_components=2, solver='mu', init='random', beta_loss='kullback-leibler', random_state=0)

W = model.fit_transform(B)

W

H = model.components_

H

## To print the top-10 terms of the first row of H, we can do the following
h =  H[1,:]
ind = h.argsort()[::-1][:10]
for i in range(10): print("{}\t{}".format(terms[ind[i]], h[ind[i]]))



"""**So using K-L divergence for matrix factorization we get a term that states about "American night for teams in war that stop other men"**"""

#However, to get matrix decomposition, into W and H, we used K-L beta loss now K =5
model = NMF(n_components=5, solver='mu', init='random', beta_loss='kullback-leibler', random_state=0)

W = model.fit_transform(B)

H = model.components_

W

H

## To print the top-10 terms of the first row of H, we can do the following
h =  H[1,:]
ind = h.argsort()[::-1][:10]
for i in range(10): print("{}\t{}".format(terms[ind[i]], h[ind[i]]))

"""**Now with k=5, we get a new meaning "Christ and Jew study of men with evidence"**"""

#However, to get matrix decomposition, into W and H, we used K-L beta loss now K =14
model = NMF(n_components=14, solver='mu', init='random', beta_loss='kullback-leibler', random_state=0)

W = model.fit_transform(B)

H = model.components_

## To print the top-10 terms of the first row of H, we can do the following
h =  H[1,:]
ind = h.argsort()[::-1][:10]
for i in range(10): print("{}\t{}".format(terms[ind[i]], h[ind[i]]))



"""**Now the meanings change "statements hold value with evidence under the context of truth of object and nature"**"""

#However, to get matrix decomposition, into W and H, we used K-L beta loss now K =32
model = NMF(n_components=32, solver='mu', init='random', beta_loss='kullback-leibler', random_state=0)

W = model.fit_transform(B)

H = model.components_

## To print the top-10 terms of the first row of H, we can do the following
h =  H[1,:]
ind = h.argsort()[::-1][:10]
for i in range(10): print("{}\t{}".format(terms[ind[i]], h[ind[i]]))





"""**Now we clearly have a medical document "Medical study and treatement of disease association of basic drug for known patient with medicine"**"""

#However, to get matrix decomposition, into W and H, we used K-L beta loss now K =40
model = NMF(n_components=40, solver='mu', init='random', beta_loss='kullback-leibler', random_state=0)

W = model.fit_transform(B)

H = model.components_

## To print the top-10 terms of the first row of H, we can do the following
h =  H[1,:]
ind = h.argsort()[::-1][:10]
for i in range(10): print("{}\t{}".format(terms[ind[i]], h[ind[i]]))





"""**Now we get kind of people "Turkish , Armenia, escape, mountain, pass Sardar, longer". Clearly, the ground reality will help us decide which k is better. The result for matrix factorization is different using different k and using different methods for factorization. This is simply because that non-negative matrix factorization does not lead to unique set of matrices, unlike SVD.**"""

#Is a different k better with K–L divergence than with Euclidean distance?



"""**Different K need not be better with K-L divergence than with Euclidean distance when it comes to non-negative matrix factorization. It simply depends on the data and the ground truth, as to which method and what value of k is most suitable to the problem in hand.**"""

#Task 3: Clustering and pLSA
#In this task, we study the use of pLSA as a dimensionality reduction tool, and compare it to Karhunen–
#Lóeve transformation. For pLSA, we use the normalized news data from the previous task; for Karhunen–Lóeve,
#you have to first normalize the data to z-scores. The documents of the data came from 20 newsgroups and
#we will use k = 20 factors in NMF/pLSA and in Karhunen–Lóeve. Our aim is to cluster the documents in
#such a way that the clusters correspond to the newsgroups. To evaluate the quality of the clustering, we use
#normalized mutual information (NMI).2 This takes values from [0, 1] and obtains value 0 for perfect match.
#Notice that NMI does not care about cluster labels or the ordering of the clusters.
#The template files have functions to compute the NMI. You will also need the file news_ground_truth.txt
#from the zip-package. Please remember that you must not use this file to guide your clustering, only to
#evaluate the results.
#To compute the pLSA, first compute the K–L divergence optimizing NMF (of the normalized data), and
#then normalize the columns of W to sum to one. To compute the Karhunen–Lóeve-transform (or PCA),
#normalize the data to z-scores, compute the SVD of the data (using existing implementations) and then do
#the transformation e.g. using the equation from slide set 4, slide 19.
#Cluster the normalized newsgroup data into 20 clusters using each of the methods below and compute
#the NMI. You can use existing implementations for k-means, but do re-start it multiple times and take the
#best solution. Try different ranks for the matrix factorizations. Which clustering(s) perform well, which do
#not? Why?
#a) k-means on the original data
#b) k-means on the first k principal components
#c) k-means on the W matrix of the NMF (using K–L divergence)

#To compute the pLSA, first compute the K–L divergence optimizing NMF (of the normalized data)
#To get matrix decomposition, into W and H, we used K-L beta loss now K = 20
model = NMF(n_components= 20, solver='mu', init='random', beta_loss='kullback-leibler', random_state=0)

W = model.fit_transform(B)

W

## Normalise W
Wnorm = W/sum(sum(W)) #

Wnorm

#Karhunen–Lóeve-transform (or PCA), normalize the data to z-scores
ZnormB = (B - np.mean(B))/(np.std(B))

ZnormB

#Better still we can use the in-built zscore function 
Z = zscore(A)

Z

#Computing the SVD of Z
U, S, V = svd(Z, full_matrices=False)

V

V.transpose()

V = V[0:20,:] #Taking the 20 topics

#Doing KL Transform or PCA
KL = np.matmul(Z, V.transpose())

KL

#Now doing Clustering for original data matrix A
  clustering = KMeans(n_clusters=20, n_init=20).fit(A)

## In Python, we can compute a slightly different normalized mutual information using scikit-learn's normalized_mutual_info_score (imported as nmi)
def nmi_news(x):
    gd = np.loadtxt('news_ground_truth.txt')
    return 1 - nmi(x, gd)



clustering.labels_

idx = clustering.labels_
## How good is this?
print("NMI for Originial Matrix = {}".format(nmi_news(idx)))

#Now doing Clustering for first k principal component
clusteringKPCS = KMeans(n_clusters=20, n_init=20).fit(KL)

idxKL = clusteringKPCS.labels_
## How good is this?
print("NMI for PCs = {}".format(nmi_news(idxKL)))

#Now doing Clustering for Wnorm which is the pLSA obtained by KL divergence optimization and then normalizing W
clusteringpLSA = KMeans(n_clusters=20, n_init=20).fit(Wnorm)

idxpLSA = clusteringpLSA.labels_
## How good is this?
print("NMI for pLSA = {}".format(nmi_news(idxpLSA)))

#Now trying different k=5 for matrix factorization to get W for pLSA via K-L divergence
model = NMF(n_components= 5, solver='mu', init='random', beta_loss='kullback-leibler', random_state=0)
W = model.fit_transform(B)
Wnorm = W/sum(sum(W))
clusteringpLSA = KMeans(n_clusters=20, n_init=20).fit(Wnorm)
idxpLSA = clusteringpLSA.labels_
## How good is this?
print("NMI for pLSA for k 5 is = {}".format(nmi_news(idxpLSA)))

#Now trying different k=10 for matrix factorization to get W for pLSA via K-L divergence
model = NMF(n_components= 10, solver='mu', init='random', beta_loss='kullback-leibler', random_state=0)
W = model.fit_transform(B)
Wnorm = W/sum(sum(W))
clusteringpLSA = KMeans(n_clusters=20, n_init=20).fit(Wnorm)
idxpLSA = clusteringpLSA.labels_
## How good is this?
print("NMI for pLSA for k 10 is = {}".format(nmi_news(idxpLSA)))

#Now trying different k=15 for matrix factorization to get W for pLSA via K-L divergence
model = NMF(n_components= 15, solver='mu', init='random', beta_loss='kullback-leibler', random_state=0)
W = model.fit_transform(B)
Wnorm = W/sum(sum(W))
clusteringpLSA = KMeans(n_clusters=20, n_init=20).fit(Wnorm)
idxpLSA = clusteringpLSA.labels_
## How good is this?
print("NMI for pLSA for k 15 is = {}".format(nmi_news(idxpLSA)))

#Now trying different k=25 for matrix factorization to get W for pLSA via K-L divergence
model = NMF(n_components= 25, solver='mu', init='random', beta_loss='kullback-leibler', random_state=0)
W = model.fit_transform(B)
Wnorm = W/sum(sum(W))
clusteringpLSA = KMeans(n_clusters=20, n_init=20).fit(Wnorm)
idxpLSA = clusteringpLSA.labels_
## How good is this?
print("NMI for pLSA for k 25 is = {}".format(nmi_news(idxpLSA)))

#Now trying different k=30 for matrix factorization to get W for pLSA via K-L divergence
model = NMF(n_components= 10, solver='mu', init='random', beta_loss='kullback-leibler', random_state=0)
W = model.fit_transform(B)
Wnorm = W/sum(sum(W))
clusteringpLSA = KMeans(n_clusters=20, n_init=20).fit(Wnorm)
idxpLSA = clusteringpLSA.labels_
## How good is this?
print("NMI for pLSA for k 30 is = {}".format(nmi_news(idxpLSA)))

#Now trying different k=35 for matrix factorization to get W for pLSA via K-L divergence
model = NMF(n_components= 35, solver='mu', init='random', beta_loss='kullback-leibler', random_state=0)
W = model.fit_transform(B)
Wnorm = W/sum(sum(W))
clusteringpLSA = KMeans(n_clusters=20, n_init=20).fit(Wnorm)
idxpLSA = clusteringpLSA.labels_
## How good is this?
print("NMI for pLSA for k 35 is = {}".format(nmi_news(idxpLSA)))

#Now trying different k=40 for matrix factorization to get W for pLSA via K-L divergence
model = NMF(n_components= 40, solver='mu', init='random', beta_loss='kullback-leibler', random_state=0)
W = model.fit_transform(B)
Wnorm = W/sum(sum(W))
clusteringpLSA = KMeans(n_clusters=20, n_init=20).fit(Wnorm)
idxpLSA = clusteringpLSA.labels_
## How good is this?
print("NMI for pLSA for k 40 is = {}".format(nmi_news(idxpLSA)))

#Now trying different k=100 for matrix factorization to get W for pLSA via K-L divergence
model = NMF(n_components= 100, solver='mu', init='random', beta_loss='kullback-leibler', random_state=0)
W = model.fit_transform(B)
Wnorm = W/sum(sum(W))
clusteringpLSA = KMeans(n_clusters=20, n_init=20).fit(Wnorm)
idxpLSA = clusteringpLSA.labels_
## How good is this?
print("NMI for pLSA for k 100 is = {}".format(nmi_news(idxpLSA)))













## Task 3
#########




# Clustering the KL matrix
clustering = KMeans(n_clusters=20, n_init=20).fit(KL)
idx = clustering.labels_
## How good is this?
print("NMI for KL = {}".format(nmi_news(idx)))


