from Data import DataExtract

X_train, X_test, Y_train, Y_test =  DataExtract.load_minist_csv()

# 测试 LogisticRegression RandomForest xgboost
from sklearn.linear_model import LogisticRegressionCV
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

models = [('LogisticRegression', LogisticRegressionCV(Cs=10, cv=5)),  # Cs正则约束，越小越强  cv:交叉验证
          ('RandomForest', RandomForestClassifier(n_estimators=50)),    # , criterion='gini'
          ('XGBoost', XGBClassifier(max_depth=3, n_estimators=50, silent=True, objective='multi:softmax'))]
for name, model in models:
	model.fit(X_train, Y_train)
	print(name, '训练集正确率：', model.score(X_train, Y_train))
	print(name, '测试集正确率：', model.score(X_test, Y_test))