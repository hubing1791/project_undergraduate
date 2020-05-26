from data_parpare import *
from data_pca import *
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.decomposition import PCA
from sklearn import svm
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import validation_curve
from plot_validation_curve import *



[X_bf,Y]=data_parpre()
[X_af,Y]=data_pca()

'''
#随机森林的降维前后对比
clf1=RandomForestClassifier( max_depth=18)
#scores_1= cross_val_score(estimator=clf1,X=X_bf,y=Y,cv=10)
#clf2=RandomForestClassifier(max_depth=18)
#scores_2= cross_val_score(estimator=clf2,X=X_af,y=Y,cv=10)
N_estimators = np.arange(5,50,1)
train_scores_1,test_scores_1= validation_curve(estimator=clf1,X=X_af,y=Y, param_name="n_estimators",param_range=N_estimators,cv=5)
train_scores_2,test_scores_2= validation_curve(estimator=clf1, X=X_af,y=Y,param_name="n_estimators",param_range=N_estimators,cv=5)
test_mean_1 = np.mean(test_scores_1, axis=1)
test_std_1 = np.std(test_scores_1, axis=1)
test_mean_2 = np.mean(test_scores_2, axis=1)
test_std_2 = np.std(test_scores_2, axis=1)
fig = plt.figure()
plt.plot(N_estimators, test_mean_1,
             color=sns.color_palette('Set1')[0], linestyle='--',
             marker='s', markersize=5,
             label='before PCA')

plt.fill_between(N_estimators,
                     test_mean_1 + test_std_1,
                     test_mean_1 - test_std_1,
                     alpha=0.15, color=sns.color_palette('Set1')[0])
plt.plot(N_estimators, test_mean_2,
             color=sns.color_palette('Set1')[1], linestyle='--',
             marker='s', markersize=5,
             label='after PCA')

plt.fill_between(N_estimators,
                     test_mean_2 + test_std_2,
                     test_mean_2 - test_std_2,
                     alpha=0.15, color=sns.color_palette('Set1')[1])
plt.xlabel('n_estimators')
plt.legend(loc='lower right')
plt.ylabel('Accuracy')
plt.ylim(0.94, 1.0)
plt.xlim(5,49)
plt.show()

#print('accruacy {} +/- {}'.format(scores_1.mean(axis=0),scores_1.std(axis=0)))
#print('accruacy {} +/- {}'.format(scores_2.mean(axis=0),scores_2.std(axis=0)))
'''


'''
#支持向量机-线性
clf3 = svm.LinearSVC(penalty='l2',dual=False,class_weight='balanced',max_iter=1000000)
scores_3= cross_val_score(estimator=clf3,X=X_bf,y=Y,cv=10)
print('accruacy {} +/- {}'.format(scores_3.mean(axis=0),scores_3.std(axis=0)))

clf4 = svm.LinearSVC(penalty='l2',dual=False,class_weight='balanced',max_iter=1000000)
scores_4= cross_val_score(estimator=clf4,X=X_af,y=Y,cv=10)
print('accruacy {} +/- {}'.format(scores_4.mean(axis=0),scores_4.std(axis=0)))
'''


svm_kernel = ['linear','poly', 'rbf', 'sigmoid']
gamma_ = np.arange(0.1,2,0.1)
para_c = np.arange(0.01,0.21,0.01)
poly_degree = np.arange(1,10,1)
rbf_pol_sig_gamma =['scale','auto']
clf5 = svm.SVC(cache_size=1000,class_weight='balanced',max_iter=1000000,kernel=svm_kernel[0])
#clf5 = svm.SVC(cache_size=1000,class_weight='balanced',max_iter=1000000,kernel=svm_kernel[1],degree=3,C=1)
#clf5 = svm.SVC(cache_size=1000,class_weight='balanced',max_iter=1000000,kernel=svm_kernel[2])
#clf5 = svm.SVC(cache_size=1000,class_weight='balanced',max_iter=1000000,kernel=svm_kernel[3],gamma = 0.007)
#scores_5= cross_val_score(estimator=clf5,X=X_bf,y=Y,cv=5)
#print('accruacy {} +/- {}'.format(scores_5.mean(axis=0),scores_5.std(axis=0)))
'''
param_test ={
    'gamma':gamma_,
    'C':para_c
}
gsearch = GridSearchCV(estimator=clf5 , param_grid = param_test, scoring='roc_auc', cv=5 )
gsearch.fit(X=X_bf,y=Y)
print(gsearch.best_params_,gsearch.best_score_)
'''
#'''
train_scores,test_scores= validation_curve(estimator=clf5, X=X_af, y=Y, param_name="C",param_range=para_c,cv=5)
plot_validation_curve(0.915,0.930,train_scores,test_scores,para_c,xlabel='para_c')
plt.xlim(0.01,0.2)
plt.show()
#'''




'''
clf6 = svm.SVC(dual=False,class_weight='balanced',max_iter=10000)
scores_6= cross_val_score(estimator=clf6,X=X_af,y=Y,cv=10)
print('accruacy {} +/- {}'.format(scores_6.mean(axis=0),scores_6.std(axis=0)))
'''