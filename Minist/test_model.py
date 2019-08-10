# logistic model
from Minist.Common.Model import LogisticModel
from Data.DataExtract import load_minist_csv
Xtrain, Xtest, Ytrain, Ytest = load_minist_csv()
model = LogisticModel()
model.fit(Xtrain, Ytrain, show_fig=True)


# numpy ANN + momentum + l1 regularizatrion
from Minist.Common.Model import ANNModel
from Data.DataExtract import load_minist_csv
Xtrain, Xtest, Ytrain, Ytest = load_minist_csv()
model = ANNModel()
model.fit(Xtrain, Ytrain, show_fig=True)


# tensorflow ANN + batch norm
from Minist.Common.Model import ANN
from Data.DataExtract import load_minist_csv
Xtrain, Xtest, Ytrain, Ytest = load_minist_csv()
model = ANN([500, 300])
model.fit(Xtrain, Ytrain, show_fig=True)


# tensorflow ANN + l2 regularization
from Minist.Common.Model import ANN_without_batch_normalization
from Data.DataExtract import load_minist_csv
Xtrain, Xtest, Ytrain, Ytest = load_minist_csv()
model = ANN_without_batch_normalization([500, 300])
model.fit(Xtrain, Ytrain, show_fig=True)


