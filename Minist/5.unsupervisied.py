
from Data import DataExtract
from sklearn.mixture import GaussianMixture
from Data.DataTransform import purity, DBI

X_train, _, Y_train, _ =  DataExtract.load_minist_csv()


model = GaussianMixture(n_components=10)
model.fit(X_train)
# VxD
M = model.means_
var = model.covariances_
# 根据数据预测各组分的后验概率。
# NxV
R = model.predict_proba(X_train)

# 分类后的
print("Purity:", purity(Y_train, R))
# 2个聚类间std偏差和/聚类均值间距离的比值  越低越好
print("DBI:", DBI(X_train, M, R))
