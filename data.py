import pandas as pd
fpath = "E:/phishingdata/phishing.csv"
ratings = pd.read_csv(fpath)
he5 = ratings.head(5)
sha = ratings.shape
col = ratings.columns
itype = ratings.dtypes
print(he5,sha,col,itype,sep='\n')
