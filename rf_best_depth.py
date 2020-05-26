
from data_parpare import  *
from plot_validation_curve import *
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import validation_curve

[X,Y]=data_parpre()
np.random.seed(1)
clf=RandomForestClassifier(n_estimators=35)
max_depths=[3,4,5,6,7,9,12,15,18,21,24,27,30,33]
#print('Training {} models'.format(len(max_depths)))
train_scores,test_scores= validation_curve(estimator=clf, X=X, y=Y, param_name="max_depth",param_range=max_depths,cv=5)
print(train_scores,test_scores,sep='\n\n')
plot_validation_curve(0.9,train_scores,test_scores,max_depths,xlabel='max_depth')
plt.xlim(3,33)
plt.show()
