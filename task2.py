import pandas as pd
from sklearn.cluster import KMeans
data=pd.read_csv(r"C:\Users\Patoju Karthikeya\Downloads\Mall_Customers.csv")
data.head(10)
CustomerID	Gender	Age	Annual Income (k$)	Spending Score (1-100)
0	1	Male	19	15	39
1	2	Male	21	15	81
2	3	Female	20	16	6
3	4	Female	23	16	77
4	5	Female	31	17	40
5	6	Female	22	17	76
6	7	Female	35	18	6
7	8	Female	23	18	94
8	9	Male	64	19	3
9	10	Female	30	19	72
X=data[["Spending Score (1-100)"]]
k=3
kmeans=KMeans(n_clusters=k,random_state=42)
kmeans.fit(X)
KMeans(n_clusters=3, random_state=42)
In a Jupyter environment, please rerun this cell to show the HTML representation or trust the notebook.
On GitHub, the HTML representation is unable to render, please try loading this page with nbviewer.org.
cluster_labels=kmeans.labels_
data["Cluster"]=cluster_labels
new_data=data.drop(["Gender"],axis=1)
print(new_data.groupby("Cluster").mean())
         CustomerID        Age  Annual Income (k$)  Spending Score (1-100)
Cluster                                                                   
0         84.483871  42.247312           54.215054               48.709677
1        113.482759  30.000000           65.293103               82.068966
2        115.530612  42.877551           67.000000               15.306122
