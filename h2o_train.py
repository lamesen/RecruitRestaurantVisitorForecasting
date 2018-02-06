import recruit_utils
import pandas as pd
import numpy as np
import h2o
from h2o.automl import H2OAutoML


def convert_columns_as_factor(hdf, column_list):
    list_count = len(column_list)
    if list_count is 0:
        return "Error: You don't have a list of binary columns."
    if (len(hdf.columns)) is 0:
        return "Error: You don't have any columns in your data frame."
    for i in range(list_count):
        try:
            hdf[column_list[i]] = hdf[column_list[i]].asfactor()
            print('Column ' + column_list[i] + " is converted into factor/catagorical.")
        except ValueError:
            print('Error: ' + str(column_list[i]) + " not found in the data frame.")


# Import data into pandas data frames
data, stores = recruit_utils.import_data()

# Create train and test set
train, test = recruit_utils.create_train_test(data, stores, clean=True, predict_large_party=True)

# Initialize h2o cluster
h2o.init()

# Drop un-needed ID variables and target variable transformations
train_subset = train.drop(['air_store_id', 'visit_date', 'id', 'air_store_id2', 'visitors'], axis=1)

# Create an H2O data frame (HDF) from the pandas data frame
h2o_train = h2o.H2OFrame(train_subset)

# User-defined function to convert columns to HDF factors
convert_columns_as_factor(h2o_train, ['dow', 'wom', 'year', 'month', 'day', 'day_of_week',
                                      'holiday_flg', 'air_genre_name', 'air_area_name',
                                      'air_genre_name0', 'air_genre_name1', 'air_genre_name2',
                                      'air_genre_name3', 'air_genre_name4', 'air_genre_name5',
                                      'air_genre_name6', 'air_genre_name7', 'air_genre_name8',
                                      'air_genre_name9', 'air_area_name0', 'air_area_name1', 'air_area_name2',
                                      'air_area_name3', 'air_area_name4', 'air_area_name5',
                                      'air_area_name6', 'air_area_name7', 'air_area_name8',
                                      'air_area_name9', 'cluster', 'large_party'])

# Setup Auto ML to run for approximately 10 hours
aml = H2OAutoML(max_runtime_secs=36000)

# Train Auto ML
aml.train(y="log_visitors",
          training_frame=h2o_train)

# Save results
model_path = h2o.save_model(model=aml.leader, path="./tmp/mymodel2", force=True)
saved_model = h2o.load_model(model_path)

# Output leaderboard
lb = aml.leaderboard

# Save original train/test data
train.to_csv('./tmp/train_in.csv')
test.to_csv('./tmp/test_in.csv')

# Mark train/test sets
train['subset'] = 'train'
test['subset'] = 'test'
combined = pd.concat([train, test])
combined.sort_values(by=['air_store_id', 'visit_date'], inplace=True)

for _ in test.visit_date.unique():
    # Set up testing HDF
    h2o_test = h2o.H2OFrame(combined)
    convert_columns_as_factor(h2o_test, ['dow', 'wom', 'year', 'month', 'day', 'day_of_week',
                                     'holiday_flg', 'air_genre_name', 'air_area_name',
                                     'air_genre_name0', 'air_genre_name1', 'air_genre_name2',
                                     'air_genre_name3', 'air_genre_name4', 'air_genre_name5',
                                     'air_genre_name6', 'air_genre_name7', 'air_genre_name8',
                                     'air_genre_name9', 'air_area_name0', 'air_area_name1', 'air_area_name2',
                                     'air_area_name3', 'air_area_name4', 'air_area_name5',
                                     'air_area_name6', 'air_area_name7', 'air_area_name8',
                                     'air_area_name9', 'cluster', 'large_party'])

    # Generate predictions
    preds = aml.leader.predict(h2o_test)
    temp_df = h2o.as_list(preds)
    temp_series = temp_df['predict']
    temp_series[temp_series < 0] = 0
    combined['predictions'] = temp_series

    combined['log_visitors'] = combined['predictions'].where(combined['subset'] == 'test',
                                                             combined['log_visitors'])

    combined['visitors'] = np.exp(combined['predictions'].where(combined['subset'] == 'test',
                                                                combined['log_visitors']))

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

# Score the file and save
scored_df = pd.read_csv('./data/sample_submission.csv')
scored_final_df = pd.merge(scored_df[['id']], combined[['id', 'visitors']], on='id')
scored_final_df.to_csv('./Output/scored_automl.csv', index=False)
