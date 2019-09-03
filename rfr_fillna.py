import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestRegressor as RFR
from sklearn.model_selection import cross_val_score as cvs
from tqdm import tqdm

from sklearn.preprocessing import MinMaxScaler


def rfr_fillna(df_all):
    # 将数据分段，选择好要进行预测的因变量和自变量
    user_id = df_all.iloc[:,0]
    X = df_all.iloc[:,1:-1]
    Y = df_all.iloc[:,-1]

    X1 = X.copy()
    Y2 = X1.iloc[:,43:]
    sex = X1.iloc[:,0]
    X2 = X1.iloc[:,1:43]

    # 量纲归一化
    MinMax_1st= MinMaxScaler().fit(X2)
    X2.iloc[:,:] = MinMax_1st.transform(X2)
    
    X2 = pd.concat([sex,X2],axis=1)
    # 对于模型进行筛选
    model = {}
    krange = range(4,30)
    for k in tqdm(list(Y2)):
        X_train = X2[Y2[k].notnull()]
        X_test = X2[Y2[k].isnull()]
        Y_train = Y2[k][Y2[k].notnull()]
        score = []
        for i in krange:
            rfr = RFR(min_samples_split=i,n_jobs = -1)
            score_each = cvs(rfr,X_train,Y_train,cv=3,n_jobs=-1).mean()
            score.append(score_each)
        best_choose = list(krange)[np.argmax(score)]
        rfr = RFR(min_samples_split=best_choose,n_jobs = -1)
        rfr = rfr.fit(X_train,Y_train)
        model[k]=rfr
        Y2[k][Y2[k].isnull()] = rfr.predict(X_test)

    # 对银行流水表再次量纲归一化
    MinMax_2nd = MinMaxScaler().fit(Y2)
    Y2.iloc[:,:] = MinMax_2nd.transform(Y2)
    
    df_adda = pd.concat([X2,Y2],axis = 1)
    

    df_adda = pd.concat([user_id,df_adda,Y],axis = 1)
    return df_adda,model,MinMax_1st,MinMax_2nd