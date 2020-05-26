from data_parpare import  *
from plot_validation_curve import *
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import validation_curve

[X,Y]=data_parpre()
np.random.seed(1)
clf=RandomForestClassifier(max_depth=18)
N_estimators = np.arange(5,50,1)
print(N_estimators)
train_scores,test_scores= validation_curve(estimator=clf, X=X, y=Y, param_name="n_estimators",param_range=N_estimators,cv=5)
print(train_scores,test_scores,sep='\n\n')



plot_validation_curve(0.94,train_scores,test_scores,N_estimators,xlabel='n_estimators')
plt.xlim(5,49)
plt.show()
