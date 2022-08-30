import pandas as pd
import numpy as np
import keras
import tensorflow as tf
import os
from keras.utils import to_categorical


file_dir = "C:/Users/Pratz/Desktop/Master Thesis/bbdc_2020_public_data/bbdc_2020/train/emg"
entries = os.listdir(file_dir)
subjects = []
def load_files(files,sub_data,dir_st):
    for subject_file in files:
        sub_data.append(pd.read_csv(dir_st + "/" + subject_file))
    return sub_data
subjects = load_files(entries,subjects,file_dir)

#Label
label_file = "C:/Users/Pratz/Desktop/Master Thesis/bbdc_2020_public_data/bbdc_2020/train/labels.train.csv"
label_df = pd.read_csv(label_file)
subjects_hand_unique = list(label_df['Subject Hand'].unique())
subjects_hand_left = [hand for hand in subjects_hand_unique if not hand.endswith('ra')]
subjects_hand_right = [hand for hand in subjects_hand_unique if not hand.endswith('la')]
label_subjects_left = []
label_subjects_right = []
for lab in subjects_hand_left:
    temp_df = label_df[label_df['Subject Hand'] == lab]
    temp_index = [temp_row for temp_row in range(temp_df.shape[0])]
    temp_df.index = temp_index
    label_subjects_left.append(temp_df)
    del temp_df
    del temp_index
for lab in subjects_hand_right:
    temp_df = label_df[label_df['Subject Hand'] == lab]
    temp_index = [temp_row for temp_row in range(temp_df.shape[0])]
    temp_df.index = temp_index
    label_subjects_right.append(temp_df)
    del temp_df
    del temp_index
label_mapping_left = {'la-nothing':0,'la-object-orient':1,'la-object-switch-hands':2,'la-object-place':3,'la-object-carry':4,'la-object-pick':5}
label_mapping_right = {'ra-nothing':0,'ra-object-orient':1,'ra-object-switch-hands':2,'ra-object-place':3,'ra-object-carry':4,'ra-object-pick':5}

#Feature Extraction
subject_features_left = []
subject_features_left_zero = []
subject_features_left_one = []
subject_target_left = []
subject_features_right = []
subject_features_right_zero = []
subject_features_right_one = []
subject_target_right = []

def feature_extraction(subject_data, label_subject, lab_map):
    target_variable = []
    feature_variables = pd.DataFrame()
    temp_df = pd.DataFrame()
    one_row = pd.DataFrame()
    for label_ob_row in range(0,len(label_subject)):
        
        start_time = label_subject.loc[label_ob_row,'Start Time']
        end_time = label_subject.loc[label_ob_row, 'End Time']
        hand_activity = label_subject.loc[label_ob_row, 'Hand Activity']

        subset_subject_data = subject_data.loc[(subject_data['ts'] >= start_time) & (subject_data['ts'] <= end_time)]
        subset_subject_data = subset_subject_data.drop(columns = ['ts'],axis = 1)
        subset_subject_data = subset_subject_data.dropna()

        #Collecting Rows Having Zero in ANY of it's columns
        temp_data = subset_subject_data[(subset_subject_data == 0).all(1)]
        temp_data['Label'] = lab_map[hand_activity]
        temp_df = temp_df.append(temp_data)
        
        #Collecting Subset Data having 1 observation ONLY
        if(subset_subject_data.shape[0] == 1):
            sub_df = subset_subject_data
            sub_df['Label'] = lab_map[hand_activity]
            one_row = one_row.append(sub_df)
        else:
            subset_subject_data = subset_subject_data[(subset_subject_data != 0).all(1)]
            mean_df = pd.DataFrame(subset_subject_data.mean()).transpose()
            max_df = pd.DataFrame(subset_subject_data.max()).transpose()
            min_df = pd.DataFrame(subset_subject_data.min()).transpose()
            median_df = pd.DataFrame(subset_subject_data.median()).transpose()
            variance_df = pd.DataFrame(subset_subject_data.var()).transpose()
            result_df = pd.concat([mean_df,median_df,min_df,max_df,variance_df], axis= 1, join= 'inner')
            feature_variables = feature_variables.append(result_df)
            target_variable.append(lab_map[hand_activity])       
    if(feature_variables.empty == False):
        feature_variables = index_assign(feature_variables)
    if(temp_df.empty == False):
        temp_df = index_assign(temp_df)
    if(one_row.empty == False):
        one_row = index_assign(one_row)
    return feature_variables, target_variable, temp_df, one_row

def index_assign(feature_data):
    new_index = [index_row for index_row in range(feature_data.shape[0])]
    feature_data.index = new_index
    new_columns = [index_col for index_col in range(feature_data.shape[1])]
    feature_data.columns = new_columns
    del new_index
    del new_columns
    return feature_data


for list_number in range(len(subjects)):
    features_train,targets_train,zeroes_df,ones_df  = feature_extraction(subjects[list_number], label_subjects_left[list_number], label_mapping_left)
    if(features_train.empty == False & len(targets_train) > 0):
        subject_features_left.append(features_train)
        subject_target_left.append(targets_train)
    subject_features_left_zero.append(zeroes_df)
    subject_features_left_one.append(ones_df)
    del features_train
    del targets_train
    del zeroes_df 
    del ones_df

for list_number in range(len(subjects)):
    features_train,targets_train,zeroes_df,ones_df = feature_extraction(subjects[list_number], label_subjects_right[list_number], label_mapping_right)
    if(features_train.empty == False & len(targets_train) > 0):
        subject_features_right.append(features_train)
        subject_target_right.append(targets_train)
    subject_features_right_zero.append(zeroes_df)
    subject_features_right_one.append(ones_df)
    del features_train
    del targets_train
    del zeroes_df 
    del ones_df



#subject_features_left = (pd.concat(subject_features_left)).to_numpy()
#subject_target_left = [element for elements in subject_target_left for element in elements]
#subject_target_left = to_categorical(subject_target_left)

#subject_features_right = (pd.concat(subject_features_right)).to_numpy()
#subject_target_right = [element for elements in subject_target_right for element in elements]
#subject_target_right = to_categorical(subject_target_right)
