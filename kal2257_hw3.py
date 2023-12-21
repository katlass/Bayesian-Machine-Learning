#### binomial mixture model two approaches: expectation maximization and variational inference ####

# imports
from google.colab import drive
from scipy.special import comb
import scipy.special as sci
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
drive.mount('/content/gdrive')
x_train= pd.read_csv(f'/content/gdrive/My Drive/E6720Fall2023_data.csv', header=None)
np.random.seed(4)


######## Approach 1: Expectation Maximization ########
def EM_binomial(x_, K,iter= 50):
  n, d = x_.shape
  phi=np.empty((n, K))
  phi_tot=np.empty((n, K))
  theta =np.linspace(0.00000001,0.99999999, K)
  pi = np.random.uniform(0,1, K)
  LLs =[]
  for i in range(iter):
    # expectation: (binomial r.v * pi)/sumk(binomial r.v * pi)
    for y in range(n):
      val=0
      for k in range(K):
        phi[y,k]=np.exp(x_.values[y]*np.log(theta[k]) + (20 -x_.values[y])*np.log(1-theta[k])+np.log(pi[k]))
        val+=phi[y,k]
      phi[y,:]= phi[y,:]/val

    # maximization
    nj= np.sum(phi, axis=0)
    pi =nj/n
    theta=np.sum(phi*x_.values, axis=0)*(1/(20*nj))

    # calculate objective function: log likelihood
    # take unnormalized expectation, sum over k, take logarithm, then sum
    for y in range(n):
      val=0
      for k in range(K):
        phi_tot[y,k]=np.exp(x_.values[y]*np.log(theta[k]) + (20 -x_.values[y])*np.log(1-theta[k])+np.log(pi[k]))
    phi_tot=np.sum(phi_tot, axis=1)
    phi_tot=np.log(phi_tot)
    LLs += [phi_tot.sum()]
    phi_tot=np.empty((n, K))
  return LLs,phi

# binomial mixture model with expectation maximization
LL3,phi1=EM_binomial(x_train, 3,iter=50)
LL9,phi2=EM_binomial(x_train, 9,iter=50)
LL15,phi3=EM_binomial(x_train, 15,iter=50)

# plot objective function
fig =plt.figure()
plt.plot(range(2,51),LL3[1:50],color="red",label='3')
plt.plot(range(2,51),LL9[1:50],color="blue",label='9')
plt.plot(range(2,51),LL15[1:50],color="goldenrod",label='15')
ax = plt.gca()
ax.set_ylim([-22740, -22660])
plt.title("EM for Binomial Mixture Model")
plt.ylabel("Log Likelihood")
plt.xlabel("Iteration")
plt.legend()
plt.show()

# examine cluster assignments for x=1...x=20
qs =[]
for x in range (0,21):
  qs+=[phi1[np.where(x_train. values == x)[0]][-1]]
result=pd.DataFrame(np.round(np.array(qs).T, 3).tolist())
sns.heatmap(result)

qs =[]
for x in range (0,21):
  qs+=[phi2[np.where(x_train. values == x)[0]][-1]]
result=pd.DataFrame(np.round(np.array(qs).T, 3).tolist())
sns.heatmap(result)

qs =[]
for x in range (0,21):
  qs+=[phi3[np.where(x_train. values == x)[0]][-1]]
result=pd.DataFrame(np.round(np.array(qs).T, 3).tolist())
sns.heatmap(result)


######## Approach 2: Variational Inference ########
def VI_bmm(x_,K=3,iter=500,runs=1):
  n, d = x_.shape
  #initializing priors
  alpha0 = 0.1
  a0 = 0.5
  b0 = 0.5
  best_VI=-1e16
  best_LLs=[]
  for j in range(runs):
    # initializing variational parameters
    a_s = np.random.uniform(10,100, K)
    b_s = np.random.uniform(10,100, K)
    alphas = np.random.uniform(0,1, K)
    # initializing log expectations for parameters
    lnpi =sci.digamma(alphas) - sci.digamma(alphas.sum())
    ln_pos_theta = sci.digamma(a_s) -sci.digamma(a_s + b_s)
    ln_neg_theta = sci.digamma(b_s) -sci.digamma(a_s + b_s)

    LLs =[]
    for i in range(iter):
      # update q(ci)
      phi =np.exp (lnpi + x_.values*ln_pos_theta + (20 -x_.values)*ln_neg_theta +np.log(comb(20, x_.values)))
      phi = phi/ np.sum(phi, axis=1)[:,None]

      # update q(pi) q(theta) parameters
      nj= np.sum(phi, axis=0)
      alphas =alpha0 +nj
      a_s=a0 + np.sum(phi*x_.values, axis=0)
      b_s=b0 + np.sum(phi*(20-x_.values), axis=0)

      # update log expectations for parameters
      lnpi =sci.digamma(alphas) - sci.digamma(alphas.sum())
      ln_pos_theta = sci.digamma(a_s) -sci.digamma(a_s + b_s)
      ln_neg_theta = sci.digamma(b_s) -sci.digamma(a_s + b_s)

      # variational objective function
      VI =-(np.sum((alphas-1)*lnpi) + sci.gammaln(sum(alphas)) - sum(sci.gammaln(alphas)) )    -\
        np.sum(((a_s-1)* ln_pos_theta) + ((b_s-1)* ln_neg_theta) -sci.gammaln(a_s) -sci.gammaln(b_s) +sci.gammaln(a_s +b_s) )+\
        np.sum((alpha0-1)*lnpi) + ((a0-1)* ln_pos_theta.sum() + (b0-1)*ln_neg_theta.sum()) +\
        np.sum(phi*lnpi) + np.sum(phi* (x_.values*ln_pos_theta + (20 -x_.values)*ln_neg_theta +np.log(comb(20, x_.values)))) -np.sum(phi*np.log(phi))
      LLs+=[VI]
    # saving best run
    if best_VI<VI:
       best_VI=VI
       best_LLs=LLs
       best_phi=phi
  return best_LLs, best_phi

# binomial mixture model with variational inference
LL3,phi1=VI_bmm(x_train, 3,iter=500,runs=10)
LL9,phi2=VI_bmm(x_train, 9,iter=500,runs=10)
LL15,phi3=VI_bmm(x_train, 15,iter=500,runs=10)

# plot objective function
fig =plt.figure()
plt.plot(range(2,501),LL3[1:],color="red",label='3')
plt.plot(range(2,501),LL9[1:],color="blue",label='9')
plt.plot(range(2,501),LL15[1:],color="goldenrod",label='15')
plt.title("VI for Binomial Mixture Model")
plt.ylabel("Log Likelihood")
plt.xlabel("Iteration")
plt.legend()
plt.show()

# examine cluster assignments for x=1...x=20
qs =[]
for x in range (0,21):
  qs+=[phi1[np.where(x_train. values == x)[0]][-1]]
result=pd.DataFrame(np.round(np.array(qs).T, 3).tolist())
sns.heatmap(result)

qs =[]
for x in range (0,21):
  qs+=[phi2[np.where(x_train. values == x)[0]][-1]]
result=pd.DataFrame(np.round(np.array(qs).T, 3).tolist())
sns.heatmap(result)

qs =[]
for x in range (0,21):
  qs+=[phi3[np.where(x_train. values == x)[0]][-1]]
result=pd.DataFrame(np.round(np.array(qs).T, 3).tolist())
sns.heatmap(result)