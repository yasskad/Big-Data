import os
import numpy as np
from math import sqrt
from operator import add
from pyspark.mllib.clustering import GaussianMixture, GaussianMixtureModel
from pyspark.mllib.linalg import Vectors
from pyspark.mllib.linalg import DenseMatrix


def find_outliers_KMeans(sc, sequence,k=2,proportion=0.95):
	#currently please take input as a list and return a list
        #for now, k=2
        df=sc.parallelize(sequence)
        df_vector=df.map(lambda x: np.array(float(x)))
        
        clusters=KMeans.train(df_vector, maxIterations=10,initializationMode="random")
        center=clusters.centers

        l=df_vector.collect()
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

def find_outliers_KGuaussians(sc, sequence,k=2,proportion=0.95):
        #currently please take input as a list and return a list
        #for now, k=2
        df=sc.parallelize(sequence)
        df_vector=df.map(lambda x: np.array(float(x)))

        gmm=GaussianMixture.train(df_vector,k)
        labels=gmm.predict(df_vector).collect()

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
        if abs(m[0]-m[1])>2*(sqrt(s[0])+sqrt(s[1])):
                if c0/n>proportion:
                        return list(l[labels==1])
                elif c1/n>proportion:
                        return list(l[labels==0])
                else:
                        return []
        else:
                return []

def find_outliers(sc, sequence,k=2,proportion=0.95):
        l1=find_outliers_KMeans(sc,sequence,k,proportion)
        l2=find_outliers_KGuaussians(sc,sequence,k,proportion)
        return l1+l2

        
if __name__=='__main__':
	print("")	
