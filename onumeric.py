import os
import numpy as np
from math import sqrt
from operator import add
from pyspark.mllib.clustering import GaussianMixture, GaussianMixtureModel, KMeans, KMeansModel
from pyspark.mllib.linalg import Vectors
from pyspark.mllib.linalg import DenseMatrix


def nearest_neighbors_filter(sequence, k=20):
    column = sequence.map(lambda x: float(x)).collect()
    column = np.array(column)
    
    mean_nn_dists = []
    for i in range(len(column)):
        """
        current = sequence.collect()[i]
        dist = sequence.map(lambda x: (x-current)**2).sortBy(lambda x: x).collect()[1:(k+1)]
        """
        current = column[i]
        dist = np.sort(np.power(column - current, 2))[1:(k+1)]
        mean_nn_dists.append(np.sum(dist)/k)
        
    mean_nn_dists = np.array(mean_nn_dists)
    m, std = np.mean(mean_nn_dists), np.std(mean_nn_dists)
    
    out = np.where(mean_nn_dists > m + 5*std)[0]
    if len(out) >= 1:
        outliers = list(mean_nn_dists[np.where(mean_nn_dists > m + 5*std)[0]])
    else:
        outliers = []
    filtered = sequence.filter(lambda x: x not in outliers)
    
    return outliers, filtered


def find_collective_outliers_KGaussians(sequence,k=2,proportion=0.5,ratio=10):
    df=sequence
    df_vector=df.map(lambda x: np.array(float(x)))
    gmm=GaussianMixture.train(df_vector,k)
    labels=gmm.predict(df_vector).collect()
    w=gmm.weights
    l=[]
    labels=np.array(labels)
    points=np.array(df_vector.collect())
    mus, sigmas=list(zip(*[(g.mu, g.sigma) for g in gmm.gaussians]))
    m=[]
    for i in range(k):
        m.append(float(mus[i].values))
    removed=[]
    not_removed=[]
    for i in range(len(w)):
        w_i=w[i]
        if w_i<proportion/k:
            removed.append(i)
        else:
            not_removed.append(i)
    m=np.array(m)
    if removed:
        m=m[removed]
    n=np.where(m==max(m))
    if n[0]:
        n=not_removed[n[0]]
        m=sorted(m)
        a=m[0]
        b=m[-2]
        c=m[-1]
        if b/a<ratio*c/b:
            l=l+list(points[labels==n])
        return l
    else:
        return []






def find_outliers_KMeans(sequence,k=2,proportion=0.95):
    #currently please take input as a list and return a list
    #for now, k=2
    df=sequence
    df_vector=df.map(lambda x: np.array(float(x)))

    clusters=KMeans.train(df_vector, k, maxIterations=10,initializationMode="random")
    center=clusters.centers

    if len(center)==1:
        return []
    else:
        l=np.array(df_vector.collect())
        l0=abs(l-center[0])
        l1=abs(l-center[1])

        n=len(l0)
        c0=sum(l1-l0>=0)
        c1=n-c0

        if c0/n>proportion:
            return list(l[l1<l0])
        elif c1/n>proportion:
            return list(l[l1>=l0])
        else:
            return []

def find_outliers_KGuaussians(sequence,k=2,proportion=0.95,distance_factor=3):
    #currently please take input as a list and return a list
    #for now, k=2
    df=sequence
    df_vector=df.map(lambda x: np.array(float(x)))

    gmm=GaussianMixture.train(df_vector,k)
    labels=gmm.predict(df_vector).collect()

    labels = np.array(labels)
    n=len(labels)
    c0=len(labels[labels==0])
    c1=n-c0

    mus, sigmas=list(zip(*[(g.mu, g.sigma) for g in gmm.gaussians]))

    m=[]
    s=[]
    for i in range(k):
        m.append(float(mus[i].values))
        s.append(float(sigmas[i].values))

        m=np.array(m)
        s=np.array(m)

        l=df_vector.collect()
        l=np.array(l)
        if abs(m[0]-m[1])>distance_factor*(sqrt(s[0])+sqrt(s[1])):
            if c0/n>proportion:
                return list(l[labels==1])
            elif c1/n>proportion:
                return list(l[labels==0])
            else:
                return []
                        
def find_outliers_Gaussian(sequence,distance_factor=6):
    df=sequence
    df_vector=df.map(lambda x: np.array(float(x)))

    gmm=GaussianMixture.train(df_vector,1)

    mu, sigma=list(zip(*[(g.mu, g.sigma) for g in gmm.gaussians]))

    m=mu[0].values
    s=sqrt(sigma[0].values)

    l=np.array(df_vector.collect())
    d=abs(l-m)
    return [m] + [s] + list(set(list(l[d>=distance_factor*s])))





def find_outliers(sequence, k=2,proportion=0.95,distance_factor=6):
    #l1=find_outliers_KMeans(sequence,k,proportion)
    #l2=find_outliers_KGuaussians(sequence,k,proportion)
    #l3=find_outliers_Gaussian(sequence,distance_factor)
    nn_outliers, filtered = nearest_neighbors_filter(sequence, k=20)
    #collective_outliers = find_collective_outliers_KGaussians(filtered,k=5,proportion=0.5,ratio=10)
    
    #outliers = nn_outliers + collective_outliers
    return nn_outliers

        
if __name__=='__main__':
    print("")
