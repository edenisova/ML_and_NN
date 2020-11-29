#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_wine
from sklearn.datasets import load_breast_cancer
from sklearn.datasets.samples_generator import make_blobs
from sklearn import metrics
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import scale


def takePlot(clusterDest, data, k):
  for j in range(k):
    x = []
    y = []
    for i, line in enumerate(data):
      if (clusterDest[i] == j):
        x.append(line[0])
        y.append(line[1])
    plt.scatter(x, y)
  plt.show()


# In[2]:


print("Big distance between groups")
samples = 1000
density = 0.025
centers = [[10, 10], [-10, -10], [10, -10]]
n_clusters = len(centers)
data, labels_true = make_blobs(n_samples=samples, centers=centers, cluster_std=density)

plt.scatter(data[:,0],data[:,1], c=labels_true)


# In[4]:


clustering = AgglomerativeClustering(n_clusters = 3, linkage = "average", affinity = "manhattan").fit(data)

num_clusters = clustering.n_clusters_
clusterDest = clustering.labels_
takePlot(clusterDest, data, num_clusters)


# In[5]:


print("Completeness: %0.3f" % metrics.completeness_score(labels_true, clusterDest))
print("Homogeneity: %0.3f" % metrics.homogeneity_score(labels_true, clusterDest))
print("Adjusted Rand index: %0.3f" % metrics.adjusted_rand_score(labels_true, clusterDest))
print("Adjusted Mutual information: %0.3f" % metrics.adjusted_mutual_info_score(labels_true, clusterDest))


# In[6]:


print("Very small distance between groups")
samples = 1000
density = 0.4
centers = [[0, 1], [-1, -1], [1, -1]]
n_clusters = len(centers)
data, labels_true = make_blobs(n_samples=samples, centers=centers, cluster_std=density)
plt.scatter(data[:,0],data[:,1], c=labels_true)


# In[7]:


clustering = AgglomerativeClustering(n_clusters = 3, linkage = "average", affinity = "manhattan").fit(data)

num_clusters = clustering.n_clusters_
clusterDest = clustering.labels_
takePlot(clusterDest, data, num_clusters)


# In[8]:


print("Completeness: %0.3f" % metrics.completeness_score(labels_true, clusterDest))
print("Homogeneity: %0.3f" % metrics.homogeneity_score(labels_true, clusterDest))
print("Adjusted Rand index: %0.3f" % metrics.adjusted_rand_score(labels_true, clusterDest))
print("Adjusted Mutual information: %0.3f" % metrics.adjusted_mutual_info_score(labels_true, clusterDest))


# In[9]:


print("Average intersection area of classes is 10-20%")
samples = 1000
density = 0.5
centers = [[0, 1], [-1, -1], [1, -1]]
n_clusters = len(centers)
data, labels_true = make_blobs(n_samples=samples, centers=centers, cluster_std=density)

plt.scatter(data[:,0],data[:,1], c=labels_true)


# In[10]:


clustering = AgglomerativeClustering(n_clusters = 3, linkage = "average", affinity = "manhattan").fit(data)

num_clusters = clustering.n_clusters_
clusterDest = clustering.labels_
takePlot(clusterDest, data, num_clusters)


# In[11]:


print("Completeness: %0.3f" % metrics.completeness_score(labels_true, clusterDest))
print("Homogeneity: %0.3f" % metrics.homogeneity_score(labels_true, clusterDest))
print("Adjusted Rand index: %0.3f" % metrics.adjusted_rand_score(labels_true, clusterDest))
print("Adjusted Mutual information: %0.3f" % metrics.adjusted_mutual_info_score(labels_true, clusterDest))


# In[21]:


print("Average intersection area of classes is 50-70%")
samples = 1000
density = 0.7
centers = [[0, 1], [-1, -1], [1, -1]]
n_clusters = len(centers)
data, labels_true = make_blobs(n_samples=samples, centers=centers, cluster_std=density)

plt.scatter(data[:,0],data[:,1], c=labels_true)


# In[22]:


clustering = AgglomerativeClustering(n_clusters = 3, linkage = "average", affinity = "manhattan").fit(data)

num_clusters = clustering.n_clusters_
clusterDest = clustering.labels_
takePlot(clusterDest, data, num_clusters)


# In[23]:


print("Completeness: %0.3f" % metrics.completeness_score(labels_true, clusterDest))
print("Homogeneity: %0.3f" % metrics.homogeneity_score(labels_true, clusterDest))
print("Adjusted Rand index: %0.3f" % metrics.adjusted_rand_score(labels_true, clusterDest))
print("Adjusted Mutual information: %0.3f" % metrics.adjusted_mutual_info_score(labels_true, clusterDest))


# In[27]:


print("Dataset Breast cancer Wisconsin")
dataset = load_breast_cancer()
X = dataset.data
y = dataset.target
#X = scale(X)

clustering = AgglomerativeClustering(n_clusters = 2, linkage = "average", affinity = "manhattan").fit(X)

clusterDest = clustering.labels_

print("Completeness: %0.3f" % metrics.completeness_score(y, clusterDest))
print("Homogeneity: %0.3f" % metrics.homogeneity_score(y, clusterDest))
print("Adjusted Rand index: %0.3f" % metrics.adjusted_rand_score(y, clusterDest))
print("Adjusted Mutual information: %0.3f" % metrics.adjusted_mutual_info_score(y, clusterDest))


# In[28]:


print("Dataset Wine")
dataset = load_wine()
X = dataset.data
y = dataset.target
X = scale(X)

clustering = AgglomerativeClustering(n_clusters = 3, linkage = "average", affinity = "manhattan").fit(X)

clusterDest = clustering.labels_

print("Completeness: %0.3f" % metrics.completeness_score(y, clusterDest))
print("Homogeneity: %0.3f" % metrics.homogeneity_score(y, clusterDest))
print("Adjusted Rand index: %0.3f" % metrics.adjusted_rand_score(y, clusterDest))
print("Adjusted Mutual information: %0.3f" % metrics.adjusted_mutual_info_score(y, clusterDest))


# In[ ]:




