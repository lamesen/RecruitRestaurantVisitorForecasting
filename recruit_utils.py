"""
Contributions from:
DSEverything - Mean Mix - Math, Geo, Harmonic (LB 0.493)
https://www.kaggle.com/dongxu027/mean-mix-math-geo-harmonic-lb-0-493
JdPaletto - Surprised Yet? - Part2 - (LB: 0.503)
https://www.kaggle.com/jdpaletto/surprised-yet-part2-lb-0-503
hklee - weighted mean comparisons, LB 0.497, 1ST
https://www.kaggle.com/zeemeen/weighted-mean-comparisons-lb-0-497-1st

Also all comments for changes, encouragement, and forked scripts rock

Keep the Surprise Going
"""

import pandas as pd
import numpy as np
from sklearn import preprocessing, metrics, cluster
from sklearn.model_selection import TimeSeriesSplit
import statsmodels.api as sm


def import_data():
    # Importing data
    data = {
        'tra': pd.read_csv('./data/air_visit_data.csv'),
        'as': pd.read_csv('./data/air_store_info.csv'),
        'hs': pd.read_csv('./data/hpg_store_info.csv'),
        'ar': pd.read_csv('./data/air_reserve.csv'),
        'hr': pd.read_csv('./data/hpg_reserve.csv'),
        'id': pd.read_csv('./data/store_id_relation.csv'),
        'tes': pd.read_csv('./data/sample_submission.csv'),
        'hol': pd.read_csv('./data/date_info.csv').rename(columns={'calendar_date': 'visit_date'})
    }

    data['hr'] = pd.merge(data['hr'], data['id'], how='inner', on=['hpg_store_id'])

    # Date/Time transformations
    for df in ['ar', 'hr']:
        data[df]['visit_datetime'] = pd.to_datetime(data[df]['visit_datetime'])
        data[df]['visit_datetime'] = data[df]['visit_datetime'].dt.date
        data[df]['reserve_datetime'] = pd.to_datetime(data[df]['reserve_datetime'])
        data[df]['reserve_datetime'] = data[df]['reserve_datetime'].dt.date
        data[df]['reserve_datetime_diff'] = data[df].apply(lambda r: (r['visit_datetime'] - r['reserve_datetime']).days,
                                                           axis=1)
        tmp1 = data[df].groupby(['air_store_id', 'visit_datetime'], as_index=False)[
            ['reserve_datetime_diff', 'reserve_visitors']].sum().rename(
            columns={'visit_datetime': 'visit_date', 'reserve_datetime_diff': 'rs1', 'reserve_visitors': 'rv1'})
        tmp2 = data[df].groupby(['air_store_id', 'visit_datetime'], as_index=False)[
            ['reserve_datetime_diff', 'reserve_visitors']].mean().rename(
            columns={'visit_datetime': 'visit_date', 'reserve_datetime_diff': 'rs2', 'reserve_visitors': 'rv2'})
        data[df] = pd.merge(tmp1, tmp2, how='inner', on=['air_store_id', 'visit_date'])

    # Feature engineering: log(visitors), day of week, week of month, year, month, day, etc.
    data['tra']['visit_date'] = pd.to_datetime(data['tra']['visit_date'])
    data['tra']['log_visitors'] = data['tra']['visitors'].apply(np.log)
    data['tra']['dow'] = data['tra']['visit_date'].dt.dayofweek
    data['tra']['wom'] = (data['tra']['visit_date'].dt.day - 1) // 7 + 1
    data['tra']['year'] = data['tra']['visit_date'].dt.year
    data['tra']['month'] = data['tra']['visit_date'].dt.month
    data['tra']['day'] = data['tra']['visit_date'].dt.day
    data['tra']['visit_date'] = data['tra']['visit_date'].dt.date
    data['tra']['large_party'] = np.where(data['tra']['visitors'] >= 120, 1., 0.)

    data['tes']['visit_date'] = data['tes']['id'].map(lambda x: str(x).split('_')[2])
    data['tes']['air_store_id'] = data['tes']['id'].map(lambda x: '_'.join(x.split('_')[:2]))
    data['tes']['visit_date'] = pd.to_datetime(data['tes']['visit_date'])
    data['tes']['log_visitors'] = 0
    data['tes']['dow'] = data['tes']['visit_date'].dt.dayofweek
    data['tes']['wom'] = (data['tes']['visit_date'].dt.day - 1) // 7 + 1
    data['tes']['year'] = data['tes']['visit_date'].dt.year
    data['tes']['month'] = data['tes']['visit_date'].dt.month
    data['tes']['day'] = data['tes']['visit_date'].dt.day
    data['tes']['visit_date'] = data['tes']['visit_date'].dt.date
    data['tes']['large_party'] = np.where(data['tes']['visitors'] >= 120, 1., 0.)

    # Manual differencing by me!
    data['tra']['subset'] = 'train'
    data['tes']['subset'] = 'test'
    combined = pd.concat([data['tra'], data['tes']])
    combined.sort_values(by=['air_store_id', 'visit_date'], inplace=True)
    combined['visitors_lag1'] = combined.groupby(['air_store_id'])['log_visitors'].shift()
    combined['visitors_diff1'] = combined.groupby(['air_store_id'])['log_visitors'].diff()
    combined['visitors_lag2'] = combined.groupby(['air_store_id'])['log_visitors'].shift(2)
    combined['visitors_diff2'] = combined.groupby(['air_store_id'])['log_visitors'].diff(2)
    combined['visitors_lag3'] = combined.groupby(['air_store_id'])['log_visitors'].shift(3)
    combined['visitors_diff3'] = combined.groupby(['air_store_id'])['log_visitors'].diff(3)
    combined['visitors_lag4'] = combined.groupby(['air_store_id'])['log_visitors'].shift(4)
    combined['visitors_diff4'] = combined.groupby(['air_store_id'])['log_visitors'].diff(4)
    combined['visitors_lag5'] = combined.groupby(['air_store_id'])['log_visitors'].shift(5)
    combined['visitors_diff5'] = combined.groupby(['air_store_id'])['log_visitors'].diff(5)
    combined['visitors_lag6'] = combined.groupby(['air_store_id'])['log_visitors'].shift(6)
    combined['visitors_diff6'] = combined.groupby(['air_store_id'])['log_visitors'].diff(6)
    combined['visitors_lag7'] = combined.groupby(['air_store_id'])['log_visitors'].shift(7)
    combined['visitors_diff7'] = combined.groupby(['air_store_id'])['log_visitors'].diff(7)

    data['tra'] = combined[combined['subset'] == 'train']
    data['tra'].drop('subset', axis=1)
    data['tes'] = combined[combined['subset'] == 'test']
    data['tes'].drop('subset', axis=1)

    # Create unique store data frame for future feature engineering
    unique_stores = data['tes']['air_store_id'].unique()
    stores = pd.concat([pd.DataFrame({'air_store_id': unique_stores,
                                      'dow': [i] * len(unique_stores)}) for i in range(7)],
                       axis=0, ignore_index=True).reset_index(drop=True)

    # Add descriptive statistics by store and day of week: min, mean, median, max and count of visitors
    tmp = data['tra'].groupby(['air_store_id', 'dow'], as_index=False)['visitors'].min().rename(
        columns={'visitors': 'min_visitors'})
    stores = pd.merge(stores, tmp, how='left', on=['air_store_id', 'dow'])
    tmp = data['tra'].groupby(['air_store_id', 'dow'], as_index=False)['visitors'].mean().rename(
        columns={'visitors': 'mean_visitors'})
    stores = pd.merge(stores, tmp, how='left', on=['air_store_id', 'dow'])
    tmp = data['tra'].groupby(['air_store_id', 'dow'], as_index=False)['visitors'].median().rename(
        columns={'visitors': 'median_visitors'})
    stores = pd.merge(stores, tmp, how='left', on=['air_store_id', 'dow'])
    tmp = data['tra'].groupby(['air_store_id', 'dow'], as_index=False)['visitors'].max().rename(
        columns={'visitors': 'max_visitors'})
    stores = pd.merge(stores, tmp, how='left', on=['air_store_id', 'dow'])
    tmp = data['tra'].groupby(['air_store_id', 'dow'], as_index=False)['visitors'].count().rename(
        columns={'visitors': 'count_observations'})
    stores = pd.merge(stores, tmp, how='left', on=['air_store_id', 'dow'])

    stores = pd.merge(stores, data['as'], how='left', on=['air_store_id'])

    # NEW FEATURES FROM Georgii Vyshnia (not that useful)
    stores['air_genre_name'] = stores['air_genre_name'].map(lambda x: str(str(x).replace('/', ' ')))
    stores['air_area_name'] = stores['air_area_name'].map(lambda x: str(str(x).replace('-', ' ')))

    lbl = preprocessing.LabelEncoder()
    for i in range(10):
        stores['air_genre_name' + str(i)] = lbl.fit_transform(
            stores['air_genre_name'].map(lambda x: str(str(x).split(' ')[i]) if len(str(x).split(' ')) > i else ''))
        stores['air_area_name' + str(i)] = lbl.fit_transform(
            stores['air_area_name'].map(lambda x: str(str(x).split(' ')[i]) if len(str(x).split(' ')) > i else ''))
    stores['air_genre_name'] = lbl.fit_transform(stores['air_genre_name'])
    stores['air_area_name'] = lbl.fit_transform(stores['air_area_name'])

    # Location clustering by me!
    cluster_stores = cluster_regions(stores[['longitude', 'latitude']], 8)
    stores['cluster'] = cluster_stores.predict(stores[['longitude', 'latitude']].as_matrix())

    data['hol']['visit_date'] = pd.to_datetime(data['hol']['visit_date'])
    data['hol']['day_of_week'] = lbl.fit_transform(data['hol']['day_of_week'])
    data['hol']['visit_date'] = data['hol']['visit_date'].dt.date

    return data, stores


def create_train_test(data, stores, clean=False, predict_large_party=False):
    train = pd.merge(data['tra'], data['hol'], how='left', on=['visit_date'])
    test = pd.merge(data['tes'], data['hol'], how='left', on=['visit_date'])

    train = pd.merge(train, stores, how='left', on=['air_store_id', 'dow'])
    test = pd.merge(test, stores, how='left', on=['air_store_id', 'dow'])

    for df in ['ar', 'hr']:
        train = pd.merge(train, data[df], how='left', on=['air_store_id', 'visit_date'])
        test = pd.merge(test, data[df], how='left', on=['air_store_id', 'visit_date'])

    train['id'] = train.apply(lambda r: '_'.join([str(r['air_store_id']), str(r['visit_date'])]), axis=1)

    train['total_reserv_sum'] = train['rv1_x'] + train['rv1_y']
    train['total_reserv_mean'] = (train['rv2_x'] + train['rv2_y']) / 2
    train['total_reserv_dt_diff_mean'] = (train['rs2_x'] + train['rs2_y']) / 2

    test['total_reserv_sum'] = test['rv1_x'] + test['rv1_y']
    test['total_reserv_mean'] = (test['rv2_x'] + test['rv2_y']) / 2
    test['total_reserv_dt_diff_mean'] = (test['rs2_x'] + test['rs2_y']) / 2

    # NEW FEATURES FROM JMBULL
    train['date_int'] = train['visit_date'].apply(lambda x: x.strftime('%Y%m%d')).astype(int)
    test['date_int'] = test['visit_date'].apply(lambda x: x.strftime('%Y%m%d')).astype(int)
    train['var_max_lat'] = train['latitude'].max() - train['latitude']
    train['var_max_long'] = train['longitude'].max() - train['longitude']
    test['var_max_lat'] = test['latitude'].max() - test['latitude']
    test['var_max_long'] = test['longitude'].max() - test['longitude']

    # NEW FEATURES FROM Georgii Vyshnia
    train['lon_plus_lat'] = train['longitude'] + train['latitude']
    test['lon_plus_lat'] = test['longitude'] + test['latitude']

    lbl = preprocessing.LabelEncoder()
    train['air_store_id2'] = lbl.fit_transform(train['air_store_id'])
    test['air_store_id2'] = lbl.transform(test['air_store_id'])

    if clean:
        train, test = train_clean(train, test)

    train = train.fillna(-1)
    test = test.fillna(-1)

    if predict_large_party:
        train['large_party'] = np.where(train['visitors'] >= 120, 1., 0.)

        subset = ['dow', 'wom', 'year', 'month', 'day', 'day_of_week', 'holiday_flg',
                  'air_genre_name', 'air_area_name', 'air_store_id2', 'cluster',
                  'min_visitors', 'mean_visitors', 'median_visitors', 'max_visitors',
                  'count_observations', 'rs1_x', 'rv1_x', 'rs2_x', 'rv2_x', 'rs1_y',
                  'rv1_y', 'rs2_y', 'rv2_y', 'total_reserv_sum', 'total_reserv_mean',
                  'total_reserv_dt_diff_mean']

        y = train['large_party']
        X = train[subset]

        lr = sm.Logit(y, X)
        model = lr.fit()
        test['large_party'] = model.predict(test[subset])
        test['large_party'] = np.where(test['large_party'] >= 0.5, 1., 0)

    return train, test


def train_clean(train, test):
    '''
    Remove observations from the training set with categorical_vars that aren't in the test
    set
    '''

    categorical_vars = ['air_genre_name', 'air_area_name', 'air_store_id2', 'cluster']
    for var in categorical_vars:
        new_train = train[train[var].isin(test[var].unique())]
    return new_train, test


def cluster_regions(df, n_clusters):
    X = df[['longitude', 'latitude']].as_matrix()
    kmeans = cluster.KMeans(n_clusters=n_clusters, init='k-means++', n_init=25, max_iter=1000).fit(X)
    return kmeans


def RMSLE(y, pred):
    return metrics.mean_squared_error(y, pred) ** 0.5


def time_series_cv(y, X, input_model):
    tscv = TimeSeriesSplit(n_splits=5)
    results = []

    for train_index, test_index in tscv.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        model = input_model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        results.append(RMSLE(y_test, predictions))

    return results


def score_predictions(predictions, name, log=True):
    # Score the file and save
    scored_df = pd.read_csv('./data/sample_submission.csv')
    temp_series = pd.Series(predictions)
    temp_series[temp_series < 0] = 0
    if log:
        scored_df['visitors'] = np.exp(temp_series)
    else:
        scored_df['visitors'] = temp_series
    scored_df.to_csv('./Output/' + name + '.csv', index=False)


def predict_iter(train, test, model):
    train.to_csv('./tmp/train_in.csv')
    test.to_csv('./tmp/test_in.csv')
    train['subset'] = 'train'
    test['subset'] = 'test'
    combined = pd.concat([train, test])
    combined.sort_values(by=['air_store_id', 'visit_date'], inplace=True)

    for _ in test.valid_date.unique():
        combined['predictions'] = model.predict(combined)
        combined['log_visitors'] = combined['predictions'].where(combined['subset'] == 'test',
                                                                 combined['log_visitors'])

        combined['visitors'] = np.exp(combined['predictions'].where(combined['subset'] == 'test',
                                                                    combined['visitors']))

        combined['visitors_lag1'] = combined.groupby(['air_store_id'])['log_visitors'].shift()
        combined['visitors_diff1'] = combined.groupby(['air_store_id'])['log_visitors'].diff()
        combined['visitors_lag2'] = combined.groupby(['air_store_id'])['log_visitors'].shift(2)
        combined['visitors_diff2'] = combined.groupby(['air_store_id'])['log_visitors'].diff(2)
        combined['visitors_lag3'] = combined.groupby(['air_store_id'])['log_visitors'].shift(3)
        combined['visitors_diff3'] = combined.groupby(['air_store_id'])['log_visitors'].diff(3)
        combined['visitors_lag4'] = combined.groupby(['air_store_id'])['log_visitors'].shift(4)
        combined['visitors_diff4'] = combined.groupby(['air_store_id'])['log_visitors'].diff(4)
        combined['visitors_lag5'] = combined.groupby(['air_store_id'])['log_visitors'].shift(5)
        combined['visitors_diff5'] = combined.groupby(['air_store_id'])['log_visitors'].diff(5)
        combined['visitors_lag6'] = combined.groupby(['air_store_id'])['log_visitors'].shift(6)
        combined['visitors_diff6'] = combined.groupby(['air_store_id'])['log_visitors'].diff(6)
        combined['visitors_lag7'] = combined.groupby(['air_store_id'])['log_visitors'].shift(7)
        combined['visitors_diff7'] = combined.groupby(['air_store_id'])['log_visitors'].diff(7)

    combined.drop('predictions', axis=1)
    train = combined[combined['subset'] == 'train']
    train.drop('subset', axis=1)
    test = combined[combined['subset'] == 'test']
    test.drop('subset', axis=1)
    train.to_csv('./tmp/train_out.csv')
    test.to_csv('./tmp/test_out.csv')

    return train, test
