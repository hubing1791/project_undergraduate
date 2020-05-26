import pandas as pd


def data_parpre():
    fpath = "E:/phishingdata/phishing.csv"
    pdata = pd.read_csv(fpath)
    # print(pdata.columns)
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
    return[X,Y]