import xgboost as xgb
import lightgbm as lgb
import pandas as pd
import numpy as np
from datetime import date
import datetime
import warnings
warnings.filterwarnings("ignore")

flow_train = pd.read_csv('./data/flow_train.csv')
tran_train = pd.read_csv('./data/transition_train.csv')

def year(date):
    date = str(date)
    return int(date[0:4])
def month(date):
    date = str(date)
    return int(date[4:6])
def day(date):
    date = str(date)
    return int(date[6:8])
flow_train['year'] = flow_train['date_dt'].apply(year)
flow_train['month'] = flow_train['date_dt'].apply(month)
flow_train['day'] = flow_train['date_dt'].apply(day)

flow_train['address'] = flow_train['city_code']+':'+flow_train['district_code']
address = list(set(flow_train['address']))

flow_train_1 = flow_train[flow_train['date_dt'] < 20180214]
flow_test_1 = flow_train[flow_train['date_dt'] >= 20180214]

label = ['dwell','flow_in','flow_out']
feature = ['year','month','day']

def rmsle(y_true, y_pred):
    return 'RMSLE', np.sqrt(np.mean(np.power(np.log1p(y_pred) - np.log1p(y_true), 2))), False

star = '20180301'
dates = []
for i in range(1,16):
    date_format = datetime.datetime.strptime(star,'%Y%m%d')
    fut_date = date_format + datetime.timedelta(days=i)
    dates.append(int(datetime.datetime.strftime(fut_date,'%Y%m%d')))
test_df = pd.DataFrame({'date_dt':dates})
test_df['year'] = test_df['date_dt'].apply(year)
test_df['month'] = test_df['date_dt'].apply(month)
test_df['day'] = test_df['date_dt'].apply(day)
test_df['date_dt'] = test_df['date_dt'].astype(str)

result = pd.DataFrame(columns=['date_dt', 'city_code', 'district_code', 'dwell', 'flow_in', 'flow_out'])

for ad in address:
    ad_split = ad.split(':')
    test_df['city_code'] = ad_split[0]
    test_df['district_code'] = ad_split[1]
    for y in label:
        train_x = flow_train[flow_train['address']==ad][feature]
        train_y = flow_train[flow_train['address']==ad][y]
        test_x = flow_test_1[flow_test_1['address']==ad][feature]
        test_y = flow_test_1[flow_test_1['address']==ad][y]
        gbm = lgb.LGBMRegressor(num_leaves=50,
                        learning_rate=0.05,
                        n_estimators=1000)
        gbm.fit(train_x, train_y, 
                eval_set=[(test_x, test_y)],
                eval_metric='l1',
                early_stopping_rounds=5)
        # predict
        y_pred = gbm.predict(test_x, num_iteration=gbm.best_iteration_)
        # eval
        print('The rmsle of prediction is:', rmsle(test_y, y_pred)[1])
        test_df[y] = gbm.predict(test_df[feature])
    result = pd.concat([result,test_df[result.columns]])

result['date_dt'] = result['date_dt'].astype(int)
result = result[['date_dt', 'city_code', 'district_code', 'dwell', 'flow_in', 'flow_out']]
result.to_csv('prediction.csv', index=False, header=None)