from data_parpare import  *
from plot_validation_curve import *
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import validation_curve

[X,Y]=data_parpre()#数据集提取

np.random.seed(1)
clf=RandomForestClassifier(max_depth=18,n_estimators=17)
Min_sample_leaf = np.arange(1,30,1)
#print(Min_sample_leaf)

train_scores,test_scores= validation_curve(estimator=clf, X=X, y=Y, param_name="min_samples_leaf",param_range=Min_sample_leaf,cv=5)
#print(train_scores,test_scores,sep='\n\n')#训练模型


plot_validation_curve(0.9,train_scores,test_scores,Min_sample_leaf,xlabel='min_samples_leaf')
plt.xlim(1,30)
plt.show()#可视化结果
