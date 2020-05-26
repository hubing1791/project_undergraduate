import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier

fpath = "E:/phishingdata/phishing.csv"
pdata = pd.read_csv(fpath)
features = ['UsingIP', 'LongURL', 'ShortURL', 'Symbol@', 'Redirecting//',
                'PrefixSuffix-', 'SubDomains', 'HTTPS', 'DomainRegLen', 'Favicon',
                'NonStdPort', 'HTTPSDomainURL', 'RequestURL', 'AnchorURL',
                'LinksInScriptTags', 'ServerFormHandler', 'InfoEmail', 'AbnormalURL',
                'WebsiteForwarding', 'StatusBarCust', 'DisableRightClick',
                'UsingPopupWindow', 'IframeRedirection', 'AgeofDomain', 'DNSRecording',
                'WebsiteTraffic', 'PageRank', 'GoogleIndex', 'LinksPointingToPage',
                'StatsReport']
X = pdata[features].values
Y = pdata['class'].values


np.random.seed(1)
clf=RandomForestClassifier(max_depth=18,n_estimators=200)
clf.fit(X=X,y=Y)

importance = (clf.feature_importances_,pdata.columns)
list(zip(*importance))
print(pd.Series(clf.feature_importances_, name='feature_importance', index=pdata[features].columns).sort_values())
#pd.Series(clf.feature_importances_, name='feature_importance', index=pdata[features].columns).sort_values().plot.barh()
plt.show()