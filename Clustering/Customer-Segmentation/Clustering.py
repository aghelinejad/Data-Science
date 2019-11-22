import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

da = pd.read_csv("Mall_Customers.csv")
x = da.iloc[:, 2:4].values

#from sklearn.preprocessing import LabelEncoder, OneHotEncoder
#encoder_1 = LabelEncoder()
#x[:, 0] = encoder_1.fit_transform(x[:, 0])
#encoder_2 = OneHotEncoder()
#x = encoder_2.fit_transform(x).toarray

from sklearn.cluster import KMeans
WCSS = []
for i in range(1, 11):
    cluster = KMeans(n_clusters = i, random_state = 0)
    cluster.fit_transform(x)
    WCSS.append(cluster.inertia_)
plt.plot(range(1, 11), WCSS)

cluster = KMeans(n_clusters = 5, random_state = 0)
y_pred_kmean = cluster.fit_predict(x)

import scipy.cluster.hierarchy as sch
cluster = sch.dendrogram(sch.linkage(x, method = "ward"))

from sklearn.cluster import AgglomerativeClustering
cluster = AgglomerativeClustering(n_clusters = 5)
y_pred_HC = cluster.fit_predict(x)

plt.scatter(x[:, 0], x[:, 1])

plt.scatter(x[y_pred_kmean == 0, 0], x[y_pred_kmean == 0, 1], color = "red")
plt.scatter(x[y_pred_kmean == 1, 0], x[y_pred_kmean == 1, 1], color = "blue")
plt.scatter(x[y_pred_kmean == 2, 0], x[y_pred_kmean == 2, 1], color = "yellow")
plt.scatter(x[y_pred_kmean == 3, 0], x[y_pred_kmean == 3, 1], color = "black")
plt.scatter(x[y_pred_kmean == 4, 0], x[y_pred_kmean == 4, 1], color = "green")
plt.show()

plt.scatter(x[y_pred_HC == 0, 0], x[y_pred_HC == 0, 1], color = "red")
plt.scatter(x[y_pred_HC == 1, 0], x[y_pred_HC == 1, 1], color = "blue")
plt.scatter(x[y_pred_HC == 2, 0], x[y_pred_HC == 2, 1], color = "yellow")
plt.scatter(x[y_pred_HC == 3, 0], x[y_pred_HC == 3, 1], color = "black")
plt.scatter(x[y_pred_HC == 4, 0], x[y_pred_HC == 4, 1], color = "green")
plt.show()