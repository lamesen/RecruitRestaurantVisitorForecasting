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
train, test = recruit_utils.create_train_test(data, stores)

# Initialize h2o cluster
h2o.init()

# Drop un-needed ID variables and target variable transformations
train_subset = train.drop(['air_store_id', 'visit_date', 'id', 'air_store_id2',
                           'visitors', 'visitor_diff', 'log_visitor_diff'], axis=1)

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
                                      'air_area_name9', 'cluster'])

# Setup Auto ML to run for approximately 7 hours
aml = H2OAutoML(max_runtime_secs=25000)

# Train Auto ML
aml.train(y="log_visitors",
          training_frame=h2o_train)

# Output leaderboard
lb = aml.leaderboard

# Set up testing HDF
h2o_test = h2o.H2OFrame(test)

# Generate predictions
preds = aml.predict(h2o_test)

# Score the file and save
scored_df = pd.read_csv('./data/sample_submission.csv')
temp_df = h2o.as_list(preds)
temp_series = temp_df['predict']
temp_series[temp_series < 0] = 0
scored_df['visitors'] = np.exp(temp_series)
scored_df.to_csv('scored_yymmdd.csv', index=False)
