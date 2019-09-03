import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier as RFC
from sklearn.model_selection import GridSearchCV as GSV
import time,datetime

def get_df_dummies(df,feature_list):
    '''
    功能:将函数的特征列构造成为哑变量
    paramas: Datafram & 需要修改的列名
    返回值:构建好的哑变量列表
    '''
    df1 = df.copy()
    for var in feature_list:
        temp= pd.get_dummies(df1[var], prefix=var, prefix_sep='_')
        del df1[var]
        df1 = pd.concat([df1,temp], axis=1)
    return df1


def timestamp_translate(df,out_format="%Y-%m-%d %H:%M:%S"):
    '''
    功能:将时间戳转化为标准的时间格式
    参数：需要变化的Dataframe 以及 输出的日期时间格式
    返回值：改变后的矩阵
    '''
    df1 = df.copy()
    df1.timestamp = df1.timestamp.apply(lambda x:time.strftime(out_format,time.localtime(x)) if x!=0 and not np.isnan(x) else 0)
    return df1


def data_translate(user_info,bank_detail,browse_history,bill_detail,model,MinMax_1st,MinMax_2nd):
    # 这段是变的user_info表
    # (这个地方需要注意，在训练过程中，如果性别未知就删除了，但是这里，在之后需要填补上缺失值)
    user_info1 = user_info.copy()
    user_info1.sex[user_info.sex==0]=np.nan
    user_info1['sex'][user_info1['sex']==2]=0

    feature_list = list(user_info1)[2:6]
    user_info1 = get_df_dummies(user_info1,feature_list)


    # bank_detail
    bank_detail1 = bank_detail.copy()
    bank_detail1.timestamp[bank_detail1['timestamp'] ==0]=np.nan
    bank_detail1 = bank_detail1.dropna()

    bank_detail1= timestamp_translate(bank_detail1,'%Y-%m')

    bank_detail1['income']=bank_detail1['expend']=0
    bank_detail1['income'] = bank_detail1['trade_amount']*(1-bank_detail1['trade_type'])
    bank_detail1['expend'] = bank_detail1['trade_amount']*(bank_detail1['trade_type'])

    bank_detail1 = bank_detail1.drop(['trade_type','trade_amount'],axis=1)

    bank_detail2 = bank_detail1.groupby(by=['user_id','timestamp']).agg({'user_id':'count','income':'sum','expend':'sum','salary_label':'sum'})
    bank_detail2['trade_amount']=bank_detail2['income']-bank_detail2['expend']
    # 生成新的特征变量，判别每月的交易正负情况，用于之后计数

    bank_detail2['trade_labels_1']=bank_detail2['trade_amount'].apply(lambda x:1 if x>0 else 0)
    bank_detail2['trade_labels_-1']=bank_detail2['trade_amount'].apply(lambda x:1 if x<0 else 0)
    bank_detail2.columns = ['comsumption_freq','income','expend','salary_label','trade_amount','trade_labels_1','trade_labels_-1']

    #再次根据用户ID对于数据进行聚合
    bank_detail2 = bank_detail2.groupby('user_id').agg({'comsumption_freq':'sum',
                                                        'income':'sum',
                                                        'expend':'sum',
                                                        'trade_amount':['sum','count'],
                                                        'salary_label':'mean',
                                                        'trade_labels_1':'sum',
                                                        'trade_labels_-1':'sum'})
    bank_detail2.columns = ['comsumption_freq',
                            'income', 
                            'expend', 
                            'trade_amount',
                            'month_',
                            'salary_label',
                            'trade_labels_1', 
                            'trade_labels_-1']

    # 用用户总量以及流水月份还有消费频数来进行计算月均和单笔消费的情况
    bank_detail2['monthly_mean'] = bank_detail2['trade_amount']/bank_detail2['month_']
    bank_detail2['each_mean'] = bank_detail2['trade_amount']/bank_detail2['comsumption_freq']

    bank_detail2 = bank_detail2.drop(['comsumption_freq','month_','trade_amount'],axis=1)

    # 重置索引
    bank_detail2 = bank_detail2.reset_index()


    # browse_history

    browse_history1 = browse_history.groupby(by =['user_id','behavior_num']).count()
    browse_history1=browse_history1.pivot_table(index='user_id',columns='behavior_num',values='timestamp' )
    browse_col=list(browse_history1)

    # bill_detail

    bill_detail1 = bill_detail.drop(['timestamp',
                                     'bank_id','min_C_bill_R',
                                     'available_amount','cash_advance_limit'],axis=1)
    bill_detail2 = bill_detail1.copy()
    # 基于之前的观察，时间戳并没有任何的作用，并且银行id，也没有作用。
    # 最低还款额，可用金额，预借现金额度与其他的特征在经济意义上都都有一定的相关性。因此。对以上特征进行删除。
    bill_detail2['L_F'] = bill_detail2['L_BA']-bill_detail2['L_RA']
    bill_detail2 = bill_detail2.drop(['L_BA','L_RA'],axis=1)

    # 新生成一列特征，表示当期已经还款的金额
    # 本期已还款金额 = 上期结转金额 + 本期账单金额 + 本期调整金额 + 循环利息 - 账单余额
    bill_detail2['C_ped']=bill_detail2['L_F']+bill_detail2['C_bill_amount']+bill_detail2['adjust_amount']+bill_detail2['compound_interest']-bill_detail2['C_bill_balance']

    bill_detail2 = bill_detail2.drop(['adjust_amount','compound_interest'],axis=1)

    # 新生成两列数据分别记录是否拖欠了上月的账单
    bill_detail2['L_F_1'] = bill_detail2['L_F'].apply(lambda x:1 if x>0 else 0)
    bill_detail2['L_F_-1'] = bill_detail2['L_F'].apply(lambda x:1 if x<0 else 0)

    # 对于处理的数据进行聚合
    bill_detail3 = bill_detail2.groupby('user_id').agg({'credit_card_limit':'mean',
                                             'C_bill_balance':'mean',
                                             'number_of_consumption':'sum',
                                             'C_bill_amount':['sum','mean'],
                                             'R_status':'mean',
                                             'L_F':'mean',
                                             'C_ped':'mean',
                                             'L_F_1':'mean',
                                             'L_F_-1':'mean'}).reset_index()
    bill_detail3.columns= ['user_id',
                           'credit_card_limit',
                           'C_bill_balance',
                            'number_of_consumption',
                            'C_bill_amount_sum',
                            'C_bill_amount_mean_period',
                            'R_status',
                            'L_F',
                            'C_ped',
                            'L_F_1',
                            'L_F_-1']
    bill_detail_col = list(bill_detail3)

    # 整合
    df_all = pd.merge(user_info1,browse_history1,how='left',on='user_id')
    df_all = pd.merge(df_all,bill_detail3,how='left',on='user_id')
    df_all = pd.merge(df_all,bank_detail2,how='left',on='user_id')

    #用0填补两组的缺失值
    df_all[bill_detail_col] = df_all[bill_detail_col].fillna(0)
    df_all[browse_col] = df_all[browse_col].fillna(0)

    # 随机森林填补性别
    X = df_all.iloc[:,2:-7]
    
    X.iloc[:,:] = MinMax_1st.transform(X)
    
    Y = df_all.sex
    X_train = X[Y.notnull()]
    Y_train = Y[Y.notnull()]
    X_test = X[Y.isnull()]

    rfc = RFC(n_estimators=500,
              criterion='entropy',
              min_samples_split=10,
              max_depth=6,
              min_samples_leaf=1,n_jobs=-1,class_weight='balanced')
    rfc = rfc.fit(X_train,Y_train)
    Y[Y.isnull()]= rfc.predict(X_test)
    df_all.sex = Y
    df_all[list(X)]= X

    # 新建一个备份表用于进行模型的银行流水表的填充
    df_all2 = df_all.iloc[:,1:]

    for i in list(model.keys()):
        X_test = df_all2[df_all2[i].isnull()].iloc[:,:43]
        rfr = model[i]
        df_all2[i][df_all2[i].isnull()]=rfr.predict(X_test)
    
    # 归一化
    df_all2.iloc[:,-7:] = MinMax_2nd.transform(df_all2.iloc[:,-7:])
    # 加入user_id这个编号
    df_all.iloc[:,1:]=df_all2    
    
    
    return df_all