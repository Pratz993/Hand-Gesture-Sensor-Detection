#Libraries
library(dplyr)
library(readr)
library(keras)
library(mltools)
library(tensorflow)
library(kerasR)
library(reticulate)
use_python('C:/Program Files/Python38/python.exe',required = TRUE)
#File Test Path
file_test_emg <- "C:/Users/Pratz/Desktop/Master Thesis/bbdc_2020_public_data/bbdc_2020/test/emg"

#Read Files Test Data
data_all_emg_test <- list.files(path = file_test_emg, pattern = '*.csv', full.names = TRUE)

#EMG Test Data
data_sub6_t1_test <- read.csv(data_all_emg_test[1])
data_sub6_t2_test <- read.csv(data_all_emg_test[2])
data_sub6_t3_test <- read.csv(data_all_emg_test[3])
data_sub6_t4_test <- read.csv(data_all_emg_test[4])
data_sub6_t5_test <- read.csv(data_all_emg_test[5])

#File Train Path
file_emg <- "C:/Users/Pratz/Desktop/Master Thesis/bbdc_2020_public_data/bbdc_2020/train/emg"
file_label <- "C:/Users/Pratz/Desktop/Master Thesis/bbdc_2020_public_data/bbdc_2020/train/labels.train.csv"

#Read File Train Data
data_all_emg <- list.files(path = file_emg, pattern = '*.csv', full.names = TRUE)
label_data_train <- read.csv(file_label)

#EMG Train Data
data_sub1_t1 <- read.csv(data_all_emg[1])
data_sub1_t2 <- read.csv(data_all_emg[2])
data_sub1_t3 <- read.csv(data_all_emg[3])
data_sub1_t4 <- read.csv(data_all_emg[4])
data_sub1_t5 <- read.csv(data_all_emg[5])
data_sub1_t6 <- read.csv(data_all_emg[6])
data_sub2_t1 <- read.csv(data_all_emg[7])
data_sub2_t2 <- read.csv(data_all_emg[8])
data_sub2_t3 <- read.csv(data_all_emg[9])
data_sub2_t4 <- read.csv(data_all_emg[10])
data_sub2_t5 <- read.csv(data_all_emg[11])
data_sub2_t6 <- read.csv(data_all_emg[12])
data_sub3_t1 <- read.csv(data_all_emg[13])
data_sub3_t2 <- read.csv(data_all_emg[14])
data_sub3_t3 <- read.csv(data_all_emg[15])
data_sub3_t4 <- read.csv(data_all_emg[16])
data_sub3_t5 <- read.csv(data_all_emg[17])
data_sub4_t1 <- read.csv(data_all_emg[18])
data_sub5_t2 <- read.csv(data_all_emg[19])
data_sub5_t6 <- read.csv(data_all_emg[20])

#Label EMG with N/A values
data_sub1_t1$label <- "N/A"
data_sub1_t2$label <- "N/A"
data_sub1_t3$label <- "N/A"
data_sub1_t4$label <- "N/A"
data_sub1_t5$label <- "N/A"
data_sub1_t6$label <- "N/A"
data_sub2_t1$label <- "N/A"
data_sub2_t2$label <- "N/A"
data_sub2_t3$label <- "N/A"
data_sub2_t4$label <- "N/A"
data_sub2_t5$label <- "N/A"
data_sub2_t6$label <- "N/A"
data_sub3_t1$label <- "N/A"
data_sub3_t2$label <- "N/A"
data_sub3_t3$label <- "N/A"
data_sub3_t4$label <- "N/A"
data_sub3_t5$label <- "N/A"
data_sub4_t1$label <- "N/A"
data_sub5_t2$label <- "N/A"
data_sub5_t6$label <- "N/A"

#Label Dataframe for Encoding
label_y <- list(1,2,3,4,5,6)
label_df <- data.frame(Activity = c("la-nothing","la-object-pick","la-object-carry","la-object-place","la-object-switch-hands","la-object-orient"), Number = c(1,2,3,4,5,6))
label_df_right <- data.frame(Activity = c("ra-nothing","ra-object-pick","ra-object-carry","ra-object-place","ra-object-switch-hands","ra-object-orient"), Number = c(1,2,3,4,5,6))

#Adding Labels
#preprocess_emg <- function(data_emg, label_emg, sub_hand){
  for(row_series in 1:nrow(label_emg)){
    sub_activity <- label_emg[row_series, "Subject.Hand"]
    start_time <- label_emg[row_series, "Start.Time"]
    end_time <- label_emg[row_series, "End.Time"]
    label_col <- label_emg[row_series, "Hand.Activity"]
    if(sub_activity == sub_hand){
      for(row_series_2 in 1:nrow(data_emg)){
        time_emg <- data_emg[row_series_2, "ts"]
        label_col_emg <- data_emg[row_series_2, "label"]
        if((time_emg >= start_time)&(time_emg < end_time)){
          label_col_emg <- label_col
        }
        data_emg[row_series_2, "label"] <- label_col_emg
      }
    }
  }
  return(data_emg)
#}

label_df_one_hot <- one_hot(label_y) 




#Label Encoding
label_encoding <- function(data_emg, label_data){
  for(variable in 1:nrow(data_emg)){
    temp <- data_emg[variable,"label"]
    for(variable2 in 1:nrow(label_data)){
      temp_activity <- label_data[variable2, "Activity"]
      temp_label <- label_data[variable2,"Number"]
      if(temp == temp_activity){
        temp <- temp_label
      }
    }
    data_emg[variable, "label"] <- temp
  }
  return(data_emg)
}

#Left Hand
#Data with Label for EDA(Left Hand)
data_sub1_t1_lab_left <- preprocess_emg(data_sub1_t1, label_data_train,"s01t01.la")
data_sub1_t2_lab_left <- preprocess_emg(data_sub1_t2, label_data_train,"s01t02.la")
data_sub1_t3_lab_left <- preprocess_emg(data_sub1_t3, label_data_train,"s01t03.la")
data_sub1_t4_lab_left <- preprocess_emg(data_sub1_t4, label_data_train,"s01t04.la")
data_sub1_t5_lab_left <- preprocess_emg(data_sub1_t5, label_data_train,"s01t05.la")
data_sub1_t6_lab_left <- preprocess_emg(data_sub1_t6, label_data_train,"s01t06.la")
data_sub2_t1_lab_left <- preprocess_emg(data_sub2_t1, label_data_train,"s02t01.la")
data_sub2_t2_lab_left <- preprocess_emg(data_sub2_t2, label_data_train,"s02t02.la")
data_sub2_t3_lab_left <- preprocess_emg(data_sub2_t3, label_data_train,"s02t03.la")
data_sub2_t4_lab_left <- preprocess_emg(data_sub2_t4, label_data_train,"s02t04.la")
data_sub2_t5_lab_left <- preprocess_emg(data_sub2_t5, label_data_train,"s02t05.la")
data_sub2_t6_lab_left <- preprocess_emg(data_sub2_t6, label_data_train,"s02t06.la")
data_sub3_t1_lab_left <- preprocess_emg(data_sub3_t1, label_data_train,"s03t01.la")
data_sub3_t2_lab_left <- preprocess_emg(data_sub3_t2, label_data_train,"s03t02.la")
data_sub3_t3_lab_left <- preprocess_emg(data_sub3_t3, label_data_train,"s03t03.la")
data_sub3_t4_lab_left <- preprocess_emg(data_sub3_t4, label_data_train,"s03t04.la")
data_sub3_t5_lab_left <- preprocess_emg(data_sub3_t5, label_data_train,"s03t05.la")
data_sub4_t1_lab_left <- preprocess_emg(data_sub4_t1, label_data_train,"s04t01.la")
data_sub5_t2_lab_left <- preprocess_emg(data_sub5_t2, label_data_train,"s05t02.la")
data_sub5_t6_lab_left <- preprocess_emg(data_sub5_t6, label_data_train,"s05t06.la")

#Remove Rows with Label = "N/A"(Left Hand)
data_sub1_t1_lab_left <- data_sub1_t1_lab_left[!(data_sub1_t1_lab_left$label == "N/A"), ]
data_sub1_t2_lab_left <- data_sub1_t2_lab_left[!(data_sub1_t2_lab_left$label == "N/A"), ]
data_sub1_t3_lab_left <- data_sub1_t3_lab_left[!(data_sub1_t3_lab_left$label == "N/A"), ]
data_sub1_t4_lab_left <- data_sub1_t4_lab_left[!(data_sub1_t4_lab_left$label == "N/A"), ]
data_sub1_t5_lab_left <- data_sub1_t5_lab_left[!(data_sub1_t5_lab_left$label == "N/A"), ]
data_sub1_t6_lab_left <- data_sub1_t6_lab_left[!(data_sub1_t6_lab_left$label == "N/A"), ]
data_sub2_t1_lab_left <- data_sub2_t1_lab_left[!(data_sub2_t1_lab_left$label == "N/A"), ]
data_sub2_t2_lab_left <- data_sub2_t2_lab_left[!(data_sub2_t2_lab_left$label == "N/A"), ]
data_sub2_t3_lab_left <- data_sub2_t3_lab_left[!(data_sub2_t3_lab_left$label == "N/A"), ]
data_sub2_t4_lab_left <- data_sub2_t4_lab_left[!(data_sub2_t4_lab_left$label == "N/A"), ]
data_sub2_t5_lab_left <- data_sub2_t5_lab_left[!(data_sub2_t5_lab_left$label == "N/A"), ]
data_sub2_t6_lab_left <- data_sub2_t6_lab_left[!(data_sub2_t6_lab_left$label == "N/A"), ]
data_sub3_t1_lab_left <- data_sub3_t1_lab_left[!(data_sub3_t1_lab_left$label == "N/A"), ]
data_sub3_t2_lab_left <- data_sub3_t2_lab_left[!(data_sub3_t2_lab_left$label == "N/A"), ]
data_sub3_t3_lab_left <- data_sub3_t3_lab_left[!(data_sub3_t3_lab_left$label == "N/A"), ]
data_sub3_t4_lab_left <- data_sub3_t4_lab_left[!(data_sub3_t4_lab_left$label == "N/A"), ]
data_sub3_t5_lab_left <- data_sub3_t5_lab_left[!(data_sub3_t5_lab_left$label == "N/A"), ]
data_sub4_t1_lab_left <- data_sub4_t1_lab_left[!(data_sub4_t1_lab_left$label == "N/A"), ]
data_sub5_t2_lab_left <- data_sub5_t2_lab_left[!(data_sub5_t2_lab_left$label == "N/A"), ]
data_sub5_t6_lab_left <- data_sub5_t6_lab_left[!(data_sub5_t6_lab_left$label == "N/A"), ]

#Left Hand Label Encoding
data_sub1_t1_lab_left <- label_encoding(data_sub1_t1_lab_left, label_df)
data_sub1_t2_lab_left <- label_encoding(data_sub1_t2_lab_left, label_df)
data_sub1_t3_lab_left <- label_encoding(data_sub1_t3_lab_left, label_df)
data_sub1_t4_lab_left <- label_encoding(data_sub1_t4_lab_left, label_df)
data_sub1_t5_lab_left <- label_encoding(data_sub1_t5_lab_left, label_df)
data_sub1_t6_lab_left <- label_encoding(data_sub1_t6_lab_left, label_df)
data_sub2_t1_lab_left <- label_encoding(data_sub2_t1_lab_left, label_df)
data_sub2_t2_lab_left <- label_encoding(data_sub2_t2_lab_left, label_df)
data_sub2_t3_lab_left <- label_encoding(data_sub2_t3_lab_left, label_df)
data_sub2_t4_lab_left <- label_encoding(data_sub2_t4_lab_left, label_df)
data_sub2_t5_lab_left <- label_encoding(data_sub2_t5_lab_left, label_df)
data_sub2_t6_lab_left <- label_encoding(data_sub2_t6_lab_left, label_df)
data_sub3_t1_lab_left <- label_encoding(data_sub3_t1_lab_left, label_df)
data_sub3_t2_lab_left <- label_encoding(data_sub3_t2_lab_left, label_df)
data_sub3_t3_lab_left <- label_encoding(data_sub3_t3_lab_left, label_df)
data_sub3_t4_lab_left <- label_encoding(data_sub3_t4_lab_left, label_df)
data_sub3_t5_lab_left <- label_encoding(data_sub3_t5_lab_left, label_df)
data_sub4_t1_lab_left <- label_encoding(data_sub4_t1_lab_left, label_df)
data_sub5_t2_lab_left <- label_encoding(data_sub5_t2_lab_left, label_df)
data_sub5_t6_lab_left <- label_encoding(data_sub5_t6_lab_left, label_df)

#Right Hand
#Data with Label for EDA(Right Hand)
data_sub1_t1_lab_right <- preprocess_emg(data_sub1_t1, label_data_train,"s01t01.ra")
data_sub1_t2_lab_right <- preprocess_emg(data_sub1_t2, label_data_train,"s01t02.ra")
data_sub1_t3_lab_right <- preprocess_emg(data_sub1_t3, label_data_train,"s01t03.ra")
data_sub1_t4_lab_right <- preprocess_emg(data_sub1_t4, label_data_train,"s01t04.ra")
data_sub1_t5_lab_right <- preprocess_emg(data_sub1_t5, label_data_train,"s01t05.ra")
data_sub1_t6_lab_right <- preprocess_emg(data_sub1_t6, label_data_train,"s01t06.ra")
data_sub2_t1_lab_right <- preprocess_emg(data_sub2_t1, label_data_train,"s02t01.ra")
data_sub2_t2_lab_right <- preprocess_emg(data_sub2_t2, label_data_train,"s02t02.ra")
data_sub2_t3_lab_right <- preprocess_emg(data_sub2_t3, label_data_train,"s02t03.ra")
data_sub2_t4_lab_right <- preprocess_emg(data_sub2_t4, label_data_train,"s02t04.ra")
data_sub2_t5_lab_right <- preprocess_emg(data_sub2_t5, label_data_train,"s02t05.ra")
data_sub2_t6_lab_right <- preprocess_emg(data_sub2_t6, label_data_train,"s02t06.ra")
data_sub3_t1_lab_right <- preprocess_emg(data_sub3_t1, label_data_train,"s03t01.ra")
data_sub3_t2_lab_right <- preprocess_emg(data_sub3_t2, label_data_train,"s03t02.ra")
data_sub3_t3_lab_right <- preprocess_emg(data_sub3_t3, label_data_train,"s03t03.ra")
data_sub3_t4_lab_right <- preprocess_emg(data_sub3_t4, label_data_train,"s03t04.ra")
data_sub3_t5_lab_right <- preprocess_emg(data_sub3_t5, label_data_train,"s03t05.ra")
data_sub4_t1_lab_right <- preprocess_emg(data_sub4_t1, label_data_train,"s04t01.ra")
data_sub5_t2_lab_right <- preprocess_emg(data_sub5_t2, label_data_train,"s05t02.ra")
data_sub5_t6_lab_right <- preprocess_emg(data_sub5_t6, label_data_train,"s05t06.ra")

#Remove Rows with Label = "N/A"(Right Hand)
data_sub1_t1_lab_right <- data_sub1_t1_lab_right[!(data_sub1_t1_lab_right$label == "N/A"), ]
data_sub1_t2_lab_right <- data_sub1_t2_lab_right[!(data_sub1_t2_lab_right$label == "N/A"), ]
data_sub1_t3_lab_right <- data_sub1_t3_lab_right[!(data_sub1_t3_lab_right$label == "N/A"), ]
data_sub1_t4_lab_right <- data_sub1_t4_lab_right[!(data_sub1_t4_lab_right$label == "N/A"), ]
data_sub1_t5_lab_right <- data_sub1_t5_lab_right[!(data_sub1_t5_lab_right$label == "N/A"), ]
data_sub1_t6_lab_right <- data_sub1_t6_lab_right[!(data_sub1_t6_lab_right$label == "N/A"), ]
data_sub2_t1_lab_right <- data_sub2_t1_lab_right[!(data_sub2_t1_lab_right$label == "N/A"), ]
data_sub2_t2_lab_right <- data_sub2_t2_lab_right[!(data_sub2_t2_lab_right$label == "N/A"), ]
data_sub2_t3_lab_right <- data_sub2_t3_lab_right[!(data_sub2_t3_lab_right$label == "N/A"), ]
data_sub2_t4_lab_right <- data_sub2_t4_lab_right[!(data_sub2_t4_lab_right$label == "N/A"), ]
data_sub2_t5_lab_right <- data_sub2_t5_lab_right[!(data_sub2_t5_lab_right$label == "N/A"), ]
data_sub2_t6_lab_right <- data_sub2_t6_lab_right[!(data_sub2_t6_lab_right$label == "N/A"), ]
data_sub3_t1_lab_right <- data_sub3_t1_lab_right[!(data_sub3_t1_lab_right$label == "N/A"), ]
data_sub3_t2_lab_right <- data_sub3_t2_lab_right[!(data_sub3_t2_lab_right$label == "N/A"), ]
data_sub3_t3_lab_right <- data_sub3_t3_lab_right[!(data_sub3_t3_lab_right$label == "N/A"), ]
data_sub3_t4_lab_right <- data_sub3_t4_lab_right[!(data_sub3_t4_lab_right$label == "N/A"), ]
data_sub3_t5_lab_right <- data_sub3_t5_lab_right[!(data_sub3_t5_lab_right$label == "N/A"), ]
data_sub4_t1_lab_right <- data_sub4_t1_lab_right[!(data_sub4_t1_lab_right$label == "N/A"), ]
data_sub5_t2_lab_right <- data_sub5_t2_lab_right[!(data_sub5_t2_lab_right$label == "N/A"), ]
data_sub5_t6_lab_right <- data_sub5_t6_lab_right[!(data_sub5_t6_lab_right$label == "N/A"), ]

#Right Hand Label Encoding
data_sub1_t1_lab_right <- label_encoding(data_sub1_t1_lab_right, label_df_right)
data_sub1_t2_lab_right <- label_encoding(data_sub1_t2_lab_right, label_df_right)
data_sub1_t3_lab_right <- label_encoding(data_sub1_t3_lab_right, label_df_right)
data_sub1_t4_lab_right <- label_encoding(data_sub1_t4_lab_right, label_df_right)
data_sub1_t5_lab_right <- label_encoding(data_sub1_t5_lab_right, label_df_right)
data_sub1_t6_lab_right <- label_encoding(data_sub1_t6_lab_right, label_df_right)
data_sub2_t1_lab_right <- label_encoding(data_sub2_t1_lab_right, label_df_right)
data_sub2_t2_lab_right <- label_encoding(data_sub2_t2_lab_right, label_df_right)
data_sub2_t3_lab_right <- label_encoding(data_sub2_t3_lab_right, label_df_right)
data_sub2_t4_lab_right <- label_encoding(data_sub2_t4_lab_right, label_df_right)
data_sub2_t5_lab_right <- label_encoding(data_sub2_t5_lab_right, label_df_right)
data_sub2_t6_lab_right <- label_encoding(data_sub2_t6_lab_right, label_df_right)
data_sub3_t1_lab_right <- label_encoding(data_sub3_t1_lab_right, label_df_right)
data_sub3_t2_lab_right <- label_encoding(data_sub3_t2_lab_right, label_df_right)
data_sub3_t3_lab_right <- label_encoding(data_sub3_t3_lab_right, label_df_right)
data_sub3_t4_lab_right <- label_encoding(data_sub3_t4_lab_right, label_df_right)
data_sub3_t5_lab_right <- label_encoding(data_sub3_t5_lab_right, label_df_right)
data_sub4_t1_lab_right <- label_encoding(data_sub4_t1_lab_right, label_df_right)
data_sub5_t2_lab_right <- label_encoding(data_sub5_t2_lab_right, label_df_right)
data_sub5_t6_lab_right <- label_encoding(data_sub5_t6_lab_right, label_df_right)


#Training Data
#Left Hand
data_emg_left_hand <- rbind(data_emg_left_hand,data_sub5_t6_lab_left)
Y_label_left <- data.frame(Label = as.numeric(data_emg_left_hand$label))
X_train_left <- subset(data_emg_left_hand,select = -label)

###############
#Right Hand
data_emg_right_hand <- rbind(data_emg_right_hand,data_sub5_t6_lab_right)
Y_label_right <- data.frame(Label = as.numeric(data_emg_right_hand$label))
Y_label_right_one_hot <- Y_label_right.to
X_train_right <- subset(data_emg_right_hand,select = -label)
###############

#Test Data
data_emg_test <- rbind(data_emg_test,data_sub6_t5_test)

#Writing csv
write.csv(X_train_left,file = "C:/Users/Pratz/Desktop/Master Thesis/bbdc_2020_public_data/bbdc_2020/preprocessed/train_data_left.csv",col.names = TRUE)
write.csv(Y_label_left, file = "C:/Users/Pratz/Desktop/Master Thesis/bbdc_2020_public_data/bbdc_2020/preprocessed/train_label_left.csv", col.names = TRUE)
write.csv(data_emg_test,file = "C:/Users/Pratz/Desktop/Master Thesis/bbdc_2020_public_data/bbdc_2020/preprocessed/test_data.csv", col.names = TRUE)
write.csv(X_train_right,file = "C:/Users/Pratz/Desktop/Master Thesis/bbdc_2020_public_data/bbdc_2020/preprocessed/train_data_right.csv", col.names = TRUE)
write.csv(Y_label_right, file = "C:/Users/Pratz/Desktop/Master Thesis/bbdc_2020_public_data/bbdc_2020/preprocessed/train_label_right.csv", col.names = TRUE)

#Model
#input_train <- layer_input(shape = 3)
model <- keras_model_sequential()
model %>% layer_dense(input_shape = dim(X_train_left)[2:3],units = 32)
model %>% bidirectional(layer_lstm(units = 64,return_sequences = TRUE, activation = 'relu'))
model %>% layer_dense(units = 32, activation = 'relu')
model %>% layer_dense(units = 1, activation = 'softmax')
summary(model)
model %>% compile(loss = 'categorical_crossentropy', optimizer = 'RMSprop', metrics = c('accuracy'))

#Parameters
batch_size_train = 10
epochs_train = 1000

#Train Model
trained_model <- model %>% fit(X_train_left,Y_label_left,batch_size = batch_size_train,epochs = epochs_train, validation_split = 0.1)
