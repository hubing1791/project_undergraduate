import pandas as pd

from sklearn.decomposition import PCA

def data_pca():
    fpath = "E:/phishingdata/phishing.csv"
    pdata = pd.read_csv(fpath)
    features_low =['DisableRightClick','NonStdPort','IframeRedirection','StatusBarCust','Redirecting//','Favicon','AbnormalURL',
                'UsingPopupWindow','StatsReport','Symbol@','ShortURL','WebsiteForwarding','InfoEmail','HTTPSDomainURL',
                'LongURL']
    X_reduce=pdata[features_low]
    pca=PCA(n_components=5)
    pca.fit(X_reduce)
    X_pca=pca.transform(X_reduce)
    pdata['1_pca']=X_pca.T[0]
    pdata['2_pca']=X_pca.T[1]
    pdata['3_pca']=X_pca.T[2]
    pdata['4_pca']=X_pca.T[3]
    pdata['5_pca']=X_pca.T[4]
    features_high = ['PageRank','DNSRecording','UsingIP','GoogleIndex','AgeofDomain','DomainRegLen','LinksPointingToPage',
                  'RequestURL','ServerFormHandler','LinksInScriptTags','PrefixSuffix-','SubDomains','WebsiteTraffic',
                  'AnchorURL','HTTPS','1_pca','2_pca','3_pca','4_pca','5_pca']
    X_af=pdata[features_high].values
    Y = pdata['class'].values
    return [X_af,Y]