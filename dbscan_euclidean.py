#!/usr/bin/env python
# coding: utf-8

# In[39]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_wine
from sklearn.datasets import load_breast_cancer
from sklearn.datasets.samples_generator import make_blobs
from sklearn import metrics
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import scale


# In[40]:


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


# In[41]:


print("Big distance between groups")
samples = 1000
density = 0.025
centers = [[10, 10], [-10, -10], [10, -10]]
n_clusters = len(centers)
data, labels_true = make_blobs(n_samples=samples, centers=centers, cluster_std=density)

plt.scatter(data[:,0],data[:,1], c=labels_true)


# In[42]:


clustering = DBSCAN().fit(data)
clusters = clustering.labels_
takePlot(clusters, data, 3)


# In[43]:


print("Completeness: %0.3f" % metrics.completeness_score(labels_true, clusters))
print("Homogeneity: %0.3f" % metrics.homogeneity_score(labels_true, clusters))
print("Adjusted Rand index: %0.3f" % metrics.adjusted_rand_score(labels_true, clusters))
print("Adjusted Mutual information: %0.3f" % metrics.adjusted_mutual_info_score(labels_true, clusters))


# In[44]:


print("Very small distance between groups")
samples = 1000
density = 0.4
centers = [[0, 1], [-1, -1], [1, -1]]
n_clusters = len(centers)
data, labels_true = make_blobs(n_samples=samples, centers=centers, cluster_std=density)
plt.scatter(data[:,0],data[:,1], c=labels_true)


# In[49]:


clustering = DBSCAN(eps = 0.2).fit(data)
clusters = clustering.labels_
takePlot(clusters, data, 3)


# In[50]:


print("Completeness: %0.3f" % metrics.completeness_score(labels_true, clusters))
print("Homogeneity: %0.3f" % metrics.homogeneity_score(labels_true, clusters))
print("Adjusted Rand index: %0.3f" % metrics.adjusted_rand_score(labels_true, clusters))
print("Adjusted Mutual information: %0.3f" % metrics.adjusted_mutual_info_score(labels_true, clusters))


# In[51]:


print("Average intersection area of classes is 10-20%")
samples = 1000
density = 0.5
centers = [[0, 1], [-1, -1], [1, -1]]
n_clusters = len(centers)
data, labels_true = make_blobs(n_samples=samples, centers=centers, cluster_std=density)

plt.scatter(data[:,0],data[:,1], c=labels_true)


# In[55]:


clustering = DBSCAN(eps = 0.15).fit(data)
clusters = clustering.labels_
takePlot(clusters, data, 3)


# In[56]:


print("Completeness: %0.3f" % metrics.completeness_score(labels_true, clusters))
print("Homogeneity: %0.3f" % metrics.homogeneity_score(labels_true, clusters))
print("Adjusted Rand index: %0.3f" % metrics.adjusted_rand_score(labels_true, clusters))
print("Adjusted Mutual information: %0.3f" % metrics.adjusted_mutual_info_score(labels_true, clusters))


# In[57]:


print("Average intersection area of classes is 50-70%")
samples = 1000
density = 0.7
centers = [[0, 1], [-1, -1], [1, -1]]
n_clusters = len(centers)
data, labels_true = make_blobs(n_samples=samples, centers=centers, cluster_std=density)

plt.scatter(data[:,0],data[:,1], c=labels_true)


# In[62]:


clustering = DBSCAN(eps = 0.15).fit(data)
clusters = clustering.labels_
takePlot(clusters, data, 3)


# In[63]:


print("Completeness: %0.3f" % metrics.completeness_score(labels_true, clusters))
print("Homogeneity: %0.3f" % metrics.homogeneity_score(labels_true, clusters))
print("Adjusted Rand index: %0.3f" % metrics.adjusted_rand_score(labels_true, clusters))
print("Adjusted Mutual information: %0.3f" % metrics.adjusted_mutual_info_score(labels_true, clusters))


# In[ ]:





# In[ ]:





# In[ ]:




