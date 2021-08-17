from getEmbeddings import getEmbeddings
import numpy as numpy
from sklearn.svm import SVC
import matplotlib.pyplot as pyplot
import scikitplot.plotters as scikitplot

def plot_cmat(yte, ypred):
	'''plotting confusion matrix'''
	skplt.plot_confusion_matrix(yte,ypred)
	plt.show

xtr,xte,ytr,yte = getEmbeddings ("datasets/train.csv")
np.save('./xtr',xtr)
np.save('./xte',xtr)
np.save('./ytr',xtr)
np.save('./yte',xtr)

xtr = np.load('./xtr.npy')
xte = np.load('./xte.npy')
ytr = np.load('./ytr.npy')
yte = np.load('./yte.npy')

gnb=GaussianNB()
gnb.fit(xtr,ytr)
y_pred = gnb.predict(xte)
m = yte.shape[0]
n = (yte != y_pred).sum()
print ("Accuracy = " + format((m-n)/m*100, '.2f') + "%")	#72.94%
plot_cmat(yte,y_pred)