# -*- coding: utf-8 -*-
"""kal2257_bml_hw2.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1PSGlVts5FhuSc9hmJByYBgLZoiVc52zF
"""

# imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.special as sci
from google.colab import drive
drive.mount('/content/gdrive')

X_set1 = pd.read_csv(f'/content/gdrive/My Drive/EECS6720Fall2023_hw2_data/X_set1.csv')
X_set2 = pd.read_csv(f'/content/gdrive/My Drive/EECS6720Fall2023_hw2_data/X_set2.csv')
X_set3 = pd.read_csv(f'/content/gdrive/My Drive/EECS6720Fall2023_hw2_data/X_set3.csv')
y_set1 = pd.read_csv(f'/content/gdrive/My Drive/EECS6720Fall2023_hw2_data/y_set1.csv')
y_set2 = pd.read_csv(f'/content/gdrive/My Drive/EECS6720Fall2023_hw2_data/y_set2.csv')
y_set3 = pd.read_csv(f'/content/gdrive/My Drive/EECS6720Fall2023_hw2_data/y_set3.csv')
z_set1 = pd.read_csv(f'/content/gdrive/My Drive/EECS6720Fall2023_hw2_data/z_set1.csv')
z_set2 = pd.read_csv(f'/content/gdrive/My Drive/EECS6720Fall2023_hw2_data/z_set2.csv')
z_set3 = pd.read_csv(f'/content/gdrive/My Drive/EECS6720Fall2023_hw2_data/z_set3.csv')


# variational inference algorithm for bayesian linear regression y ~ N(xTw, lambda) with unknown precision lambda and w ~ N (0,a1...ad) unknown precision a1...ad
def bayesian_regression(x_data,y_data,a0, b0 ,e0,f0,T = 500):

  # initializing parameters
  n, d = x_data.shape
  a =np.repeat([a0], d)
  b =np.repeat([b0], d)
  e, f = e0, f0
  VI_objectives =[]

  for t in range(T):
    # q (w): joint likelihood y ~ N(xTw, lambda) and w ~ N (0,a1...ad)
    E_ak =a/b
    E_lambda =e/f
    xTx = np.dot(x_data.T, x_data)*(1/E_lambda)
    w_covar =np.linalg.inv(np.identity(d)*(E_ak) + xTx)
    yx =np.dot(x_data.T,y_data)*(1/E_lambda)
    w_mean =np.dot(w_covar,yx)

    #q (a):joint likelihood w ~ N (0,a1...ad) and ak ~ Gamma (a0,b0)
    a =np.repeat([a0+1/2], d)
    b =b0 + np.diag(w_covar+w_mean**2)/2

    # q (lambda):joint likelihood y ~ N(xTw, lambda) and lambda ~ Gamma (e0,f0)
    e = e0 + n / 2
    sq_diff=(y_data.values-np.dot(x_data,w_mean))**2
    sq_diff=sq_diff.sum()
    xTcovar = np.dot(x_data, w_covar)
    xTcovarx = np.dot(xTcovar, x_data.T)
    xTcovarx = np.trace(xTcovarx)
    f=f0 + (sq_diff+xTcovarx)/2


    #VI Objective: assess for convergence
    py = n/2 * (sci.psi(e) - np.log(f)) - ((e/f) * (sq_diff+ xTcovarx))/2
    pw = ((sci.psi(a) - np.log(b)).sum() -np.dot(np.diag(w_covar+w_mean**2), (a/b)))/2
    pa = (((a0 - 1) * (sci.psi(a) - np.log(b))) - (b0 *(a/b))).sum()
    pl = ((e0 - 1) * (sci.psi(e) - np.log(f))) - (f0 *(e/f))
    qw = np.linalg.slogdet(w_covar)[1]/2         #np.log(np.linalg.det(w_covar))/2 –> gives error inf
    ql = sci.gammaln(e)+ (1 -e) * sci.psi(e) -np.log(f) + e
    qa = (sci.gammaln(a)+ (1 -a) * sci.psi(a) -np.log(b) + a).sum()
    VI = py + pw + pa + pl + qw + ql + qa
    VI_objectives+=[VI]
  y_pred=np.dot(x_data,w_mean)
  inv_Eak =1/(a/b)
  inv_Elambda =1/(e/f)
  return (y_pred,inv_Eak,inv_Elambda,VI_objectives)

# define priors
a0, b0 = 10e-16, 10e-16
e0, f0= 1,1


# dataset 1
y_pred1,inv_Eak1,inv_Elambda1,VI_objectives1 = bayesian_regression(X_set1,y_set1,a0, b0 ,e0,f0,T = 500)

# objective function
fig =plt.figure()
plt.plot(range(1, 501),VI_objectives1,color="red")
plt.title("VI Objective Function for Bayesian Linear Regression")
plt.xlabel("Iteration")
plt.ylabel("LL")
plt.show()

# 1/ E[ak] = b/a
fig =plt.figure()
plt.plot(range(1,inv_Eak1.shape[0] +1),inv_Eak1,color="red")
plt.title("Parameter w's Noise Precision")
plt.xlabel("Dimension")
plt.ylabel("Inverse Expected Value")
plt.show()

#1/ E[lambda] = f/e
print("1/ E[λ]:",inv_Elambda1)

# prediction accuracy
plt.plot(z_set1, y_pred1, color='red', label='$\hat{y}$')
plt.scatter(z_set1, y_set1, color='blue', s = 10)
plt.plot(z_set1, 10 * np.sinc(z_set1), label=u'y- ε')
plt.title('Predicted y vs Ground Truth')
plt.xlabel('z')
plt.ylabel('y')
plt.legend()
plt.show()


# dataset 2
y_pred2,inv_Eak2,inv_Elambda2,VI_objectives2 = bayesian_regression(X_set2,y_set2,a0, b0 ,e0,f0,T = 500)

# objective function
fig =plt.figure()
plt.plot(range(1, 501),VI_objectives2,color="red")
plt.title("VI Objective Function for Bayesian Linear Regression")
plt.xlabel("Iteration")
plt.ylabel("LL")
plt.show()

# 1/ E[ak] = b/a
fig =plt.figure()
plt.plot(range(1,inv_Eak2.shape[0] +1),inv_Eak2,color="red")
plt.title("Parameter w's Noise Precision")
plt.xlabel("Dimension")
plt.ylabel("Inverse Expected Value")
plt.show()

#1/ E[lambda] = f/e
print("1/ E[λ]:",inv_Elambda2)

# prediction accuracy
plt.plot(z_set2, y_pred2, color='red', label='$\hat{y}$')
plt.scatter(z_set2, y_set2, color='blue', s = 10)
plt.plot(z_set2, 10 * np.sinc(z_set2), label=u'y- ε')
plt.title('Predicted y vs Ground Truth')
plt.xlabel('z')
plt.ylabel('y')
plt.legend()
plt.show()


# dataset 3
y_pred3,inv_Eak3,inv_Elambda3,VI_objectives3 = bayesian_regression(X_set3,y_set3,a0, b0 ,e0,f0,T = 500)

# objective function
fig =plt.figure()
plt.plot(range(1, 501),VI_objectives3,color="red")
plt.title("VI Objective Function for Bayesian Linear Regression")
plt.xlabel("Iteration")
plt.ylabel("LL")

# 1/ E[ak] = b/a
fig =plt.figure()
plt.plot(range(1,inv_Eak3.shape[0] +1),inv_Eak3,color="red")
plt.title("Parameter w's Noise Precision")
plt.xlabel("Dimension")
plt.ylabel("Inverse Expected Value")
plt.show()

#1/ E[lambda] = f/e
print("1/ E[λ]:",inv_Elambda3)

# prediction accuracy
plt.plot(z_set3, y_pred3, color='red', label='$\hat{y}$')
plt.scatter(z_set3, y_set3, color='blue', s = 10)
plt.plot(z_set3, 10 * np.sinc(z_set3), label=u'y- ε')
plt.title('Predicted y vs Ground Truth')
plt.xlabel('z')
plt.ylabel('y')
plt.legend()
plt.show()