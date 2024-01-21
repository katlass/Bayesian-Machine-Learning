#!/usr/bin/env python
# coding: utf-8

from scipy.stats import norm
from numpy.random import multivariate_normal
from tabulate import tabulate
import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt

# imports
X_train = pd.read_csv("/Users/katelassiter/Downloads/bayesian_ml/EECS6720_data_hw1/ratings.csv",names=["user_id","movie_id","rating"])
X_test = pd.read_csv("/Users/katelassiter/Downloads/bayesian_ml/EECS6720_data_hw1/ratings_test.csv",names= ["user_id","movie_id","rating"])


u_ids=np.unique(X_train["user_id"])
m_ids=np.unique(X_train["movie_id"])
N =max(u_ids)
M =max(m_ids)
R =np.full((max(u_ids), max(m_ids)),np.nan)


# Filling matrix R with (i,j) pairs
for i in range(len(u_ids)):
    y_idx =X_train[X_train["user_id"]==u_ids[i]]["movie_id"].values-1
    R[i][y_idx]=X_train[X_train["user_id"]==u_ids[i]]["rating"].values 

Flowers yeah yeah what is the time anything else Play music she's
# confusion matrix
def confusion_matrix(ypred,ytrue):
    if ypred==1 and ytrue ==1:
        val="TP"
    elif ypred==1 and  ytrue ==-1:
        val="FP"
    elif ypred==-1 and  ytrue ==-1:
        val="TN"
    elif ypred==-1 and  ytrue ==1:
        val="FN"
    return(val)


def likelihood_EM(matrix1,matrix2,variance=1, lambda_=1):
    constant =2*((d/2) * np.log(1/(2*np.pi)))
    norm_1=sum(list(map(lambda x: np.linalg.norm(matrix1[:,x])**2,range(matrix1.shape[1]))))
    norm_2=sum(list(map(lambda x: np.linalg.norm(matrix2[:,x])**2,range(matrix2.shape[1]))))
    
    # calculate positive/negative probability where Rij is positive/negative
    UT =np.dot(matrix1.T,matrix2)
    UT_pos =UT[pos_x_loc,pos_y_loc]
    UT_neg =UT[neg_x_loc,neg_y_loc]
    LL_pos =sum(np.log(norm.cdf(UT_pos/s)))
    LL_neg =sum(np.log(1-norm.cdf(UT_neg/s)))

    likelihood =constant+LL_pos+LL_neg-lambda_/2 *(norm_1+  norm_2 )  
    return likelihood

#####################################################
# EM for Probit Matrix Factorization

d = 5
c = 1
var = 1
s=np.sqrt(var)
lambda_ =(1/c)*np.identity(d)


# getting indices of positive vs negative i,j pairs
pos_x_loc =np.argwhere(R==1)[:,0]
pos_y_loc =np.argwhere(R==1)[:,1]
neg_x_loc =np.argwhere(R==-1)[:,0]
neg_y_loc =np.argwhere(R==-1)[:,1]



T =100
confusion_results =[]
run_likelihoods =[]
likelihoods =[]

for iter_ in range(5):
    U =np.random.multivariate_normal([0]*d, 0.1*np.identity(d),N).T
    V =np.random.multivariate_normal([0]*d, 0.1*np.identity(d),M).T
    phi=np.zeros((N,M))
    
    for x in range(1,T +1):
        print(iter_,x)
        # EXPECTATION #
        # posterior calculated as truncated normal
        UT =np.dot(U.T,V) 
        z_UT=(UT*-1)/s
        pdf_UT =norm.pdf(z_UT)
        cdf_UT =norm.cdf(z_UT)
        pos_prob=(pdf_UT/(1-cdf_UT))*s 
        neg_prob=(-pdf_UT/cdf_UT)*s
        phi[pos_x_loc,pos_y_loc]=UT[pos_x_loc,pos_y_loc]+pos_prob[pos_x_loc,pos_y_loc]
        phi[neg_x_loc,neg_y_loc]=UT[neg_x_loc,neg_y_loc]+neg_prob[neg_x_loc,neg_y_loc]

        # MAXIMIZATION #
        # solve ridge regression solution using expectation matrix phi instead of true matrix R
        for i in range(N):
            ui =np.argwhere(np.isnan(R[i])== False).flatten()
            vTv = np.dot(V[:,ui], V[:,ui].T)*(1/var)
            covar =np.linalg.inv(lambda_  + vTv)
            Rv =np.dot(phi[i,list(ui)],V[:,ui].T)*(1/var)
            U[:,i] =np.dot(covar,Rv)

        for i in range(M):
            vj =np.argwhere(np.isnan(R[:,i])== False).flatten()
            uTu =np.dot(U[:,vj], U[:,vj].T)*(1/var)
            covar =np.linalg.inv(lambda_  + uTu)
            Ru =np.dot(phi[list(vj),i],U[:,vj].T)*(1/var)
            V[:,i]=np.dot(covar,Ru)
            
        # calculate log likelihood
        likelihoods=likelihoods +[likelihood_EM(U,V,variance=1, lambda_=c)]
    
    # calculate prediction
    u_T_v=np.diagonal(np.dot(U[:,X_test["user_id"].values-1].T, V[:,X_test["movie_id"].values-1]))
    y_hat=[-1 if i<0 else 1 for i in u_T_v]
    accuracy =list(map(lambda y_pred,y_true: confusion_matrix(y_pred,y_true),np.array(y_hat), X_test["rating"].values))
    confusion_results =confusion_results +[accuracy]
    run_likelihoods=run_likelihoods +[likelihoods]



# plotting objective 
fig =plt.figure()
plt.plot(range(1,100),run_likelihoods[0][1:100],color="red",label='Run 1')
plt.title("EM for Probit Matrix Factorization")
plt.ylabel("Log Likelihood")
plt.xlabel("Iteration")
plt.legend()
plt.show()


fig =plt.figure()
plt.plot(range(20,100),run_likelihoods[0][20:100],color="red",label='Run 1')
plt.plot(range(20,100),run_likelihoods[1][20:100],color="Blue",label='Run 2')
plt.plot(range(20,100),run_likelihoods[2][20:100],color="Green",label='Run 3')
plt.plot(range(20,100),run_likelihoods[3][20:100],color="Purple",label='Run 4')
plt.plot(range(20,100),run_likelihoods[4][20:100],color="Pink",label='Run 5')
plt.title("EM for Probit Matrix Factorization")
plt.ylabel("Log Likelihood")
plt.xlabel("Iteration")
plt.legend()
plt.show()


# confusion matrix results
confusion_results[0].sort()
conf_1=np.unique(confusion_results[0], return_counts = True)
confusion_results[1].sort()
conf_2=np.unique(confusion_results[1], return_counts = True)
confusion_results[2].sort()
conf_3=np.unique(confusion_results[2], return_counts = True)
confusion_results[3].sort()
conf_4=np.unique(confusion_results[3], return_counts = True)
confusion_results[4].sort()
conf_5=np.unique(confusion_results[4], return_counts = True)


conf_matrix=np.array([["   ",-1,1],[-1,conf_1[1][2],conf_1[1][0]],[1,conf_1[1][1],conf_1[1][3]]])
print(tabulate(conf_matrix))
print("prediction accuracy:",(conf_1[1][2]+conf_1[1][3])/sum(conf_1[1]))


conf_matrix=np.array([["   ",-1,1],[-1,conf_2[1][2],conf_2[1][0]],[1,conf_2[1][1],conf_2[1][3]]])
print(tabulate(conf_matrix))
print("prediction accuracy:",(conf_2[1][2]+conf_2[1][3])/sum(conf_2[1]))


conf_matrix=np.array([["   ",-1,1],[-1,conf_3[1][2],conf_3[1][0]],[1,conf_3[1][1],conf_3[1][3]]])
print(tabulate(conf_matrix))
print("prediction accuracy:",(conf_3[1][2]+conf_3[1][3])/sum(conf_3[1]))


conf_matrix=np.array([["   ",-1,1],[-1,conf_4[1][2],conf_4[1][0]],[1,conf_4[1][1],conf_4[1][3]]])
print(tabulate(conf_matrix))
print("prediction accuracy:",(conf_4[1][2]+conf_4[1][3])/sum(conf_4[1]))


conf_matrix=np.array([["   ",-1,1],[-1,conf_5[1][2],conf_5[1][0]],[1,conf_5[1][1],conf_5[1][3]]])
print(tabulate(conf_matrix))
print("prediction accuracy:",(conf_5[1][2]+conf_5[1][3])/sum(conf_5[1]))


confusion_results.sort()
confusion_results=np.unique(confusion_results, return_counts = True)


conf_matrix=np.array([["   ",-1,1],[-1,confusion_results[1][2],confusion_results[1][0]],[1,confusion_results[1][1],confusion_results[1][3]]])
print(tabulate(conf_matrix))
print("prediction accuracy:",(confusion_results[1][2]+confusion_results[1][3])/sum(confusion_results[1]))





#####################################################
# Gibbs Sampler for Probit Matrix Factorization

def likelihood_gibbs(A,matrix1,matrix2,x_locs,y_locs,variance=1, lambda_=1):
    diff_term=np.linalg.norm(A[x_locs,y_locs]-np.dot(matrix1.T,matrix2)[x_locs,y_locs])**2
    norm_1=sum(list(map(lambda x: np.linalg.norm(matrix1[:,x])**2,range(matrix1.shape[1]))))
    norm_2=sum(list(map(lambda x: np.linalg.norm(matrix2[:,x])**2,range(matrix2.shape[1]))))
    return -1*(diff_term*(1/(2*variance)))-lambda_/2 *(norm_1+  norm_2 )


T =1000
samples_drawn=0
confusion_results =[]
likelihoods =[]
lambda_ =(1/c)*np.identity(d)


U = np.zeros((d,N))
V = np.zeros((d,M))

# isolating (i,j) locations
x_locs =np.argwhere(np.isnan(R)==False)[:,0]
y_locs =np.argwhere(np.isnan(R)==False)[:,1]

# initializing an array of zeros that will hold the total sum of utv over every iteration
u_T_v_total=np.zeros((len(X_test)))

for x in range(1,T +1):
    # solve  ridge regression equation for each user
    for i in range(N):
        ui =np.argwhere(np.isnan(R[i])== False).flatten()
        vTv = np.dot(V[:,ui], V[:,ui].T)*(1/var)
        covar =np.linalg.inv(lambda_ + vTv)
        Rv =np.dot(R[i,list(ui)],V[:,ui].T)*(1/var)
        mean =np.dot(covar,Rv)
        U[:,i]= np.random.multivariate_normal(mean, covar) # generate random variable according to this distribution
    
    # solve ridge regression equation for each movie
    for i in range(M):
        vj =np.argwhere(np.isnan(R[:,i])== False).flatten()
        uTu =np.dot(U[:,vj], U[:,vj].T)*(1/var)
        covar =np.linalg.inv(lambda_ + uTu)
        Ru =np.dot(R[list(vj),i],U[:,vj].T)*(1/var)
        mean =np.dot(covar,Ru)
        V[:,i]= np.random.multivariate_normal(mean, covar)# generate random variable according to this distribution

    # solve log likelihood 
    likelihoods=likelihoods +[likelihood_gibbs(R,U,V,x_locs,y_locs,var, c)]
    
    # after burn out phase of 100 iter, take monte carlo sample every 25 iter to find average E[rij|R]
    if  x>=100 and x%25==0:
        print(x)
        samples_drawn=samples_drawn +1
        u_T_v=np.diagonal(np.dot(U[:,X_test["user_id"].values-1].T, V[:,X_test["movie_id"].values-1]))
        u_T_v_total=(u_T_v_total+u_T_v)
        MC_aprox=u_T_v_total/samples_drawn
        y_hat=[-1 if i<0 else 1 for i in MC_aprox]
        accuracy =list(map(lambda y_pred,y_true: confusion_matrix(y_pred,y_true),np.array(y_hat), X_test["rating"].values))
        confusion_results =confusion_results +[accuracy]
confusion_results.sort()
confusion_results=np.unique(confusion_results, return_counts = True)


# plotting objective
fig =plt.figure()
plt.plot(range(1,1001),likelihoods,color="red",label='Iration 1 to 1000')
plt.title("Gibbs Sampler Log Likelihood All Iterations")
plt.ylabel("LL")
plt.xlabel("Iteration")
plt.legend()
plt.show()


fig =plt.figure()
plt.plot(range(100,1000),likelihoods[99:999] ,color="red",label='Iteration 100 to 1000')
plt.title("Gibbs Sampler Log Likelihood Iteration 100 to 1000")
plt.ylabel("LL")
plt.xlabel("Iteration")
plt.legend()
plt.show()


# confusion matrix results
conf_matrix=np.array([["   ",-1,1],[-1,confusion_results[1][2],confusion_results[1][0]],[1,confusion_results[1][1],confusion_results[1][3]]])
print(tabulate(conf_matrix))
print("prediction accuracy:",(confusion_results[1][2]+confusion_results[1][3])/sum(confusion_results[1]))




accuracy =list(map(lambda y_pred,y_true: confusion_matrix(y_pred,y_true),np.array(y_hat), X_test["rating"].values))
accuracy.sort()
confusion_results=np.unique(accuracy, return_counts = True)
conf_matrix=np.array([["   ",-1,1],[-1,confusion_results[1][2],confusion_results[1][0]],[1,confusion_results[1][1],confusion_results[1][3]]])
print(tabulate(conf_matrix))
print("prediction accuracy:",(confusion_results[1][2]+confusion_results[1][3])/sum(confusion_results[1]))
