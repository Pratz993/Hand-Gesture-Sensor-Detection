#Libraries
library(dplyr)
library(readr)
library(keras)
library(mltools)
library(tensorflow)

#File Train Path
file_mocap <- "C:/Users/Pratz/Desktop/Master Thesis/bbdc_2020_public_data/bbdc_2020/train/mocap"
file_label <- "C:/Users/Pratz/Desktop/Master Thesis/bbdc_2020_public_data/bbdc_2020/train/labels.train.csv"

#Read Train Data
data_all_mocap <- list.files(path = file_mocap, pattern = '*.csv', full.names = TRUE)

#Motion Capture Train Data
data_sub1_t1_mocap <- read.csv(data_all_mocap[1])
data_sub1_t2_mocap <- read.csv(data_all_mocap[2])
data_sub1_t3_mocap <- read.csv(data_all_mocap[3])
data_sub1_t4_mocap <- read.csv(data_all_mocap[4])
data_sub1_t5_mocap <- read.csv(data_all_mocap[5])
data_sub1_t6_mocap <- read.csv(data_all_mocap[6])
data_sub2_t1_mocap <- read.csv(data_all_mocap[7])
data_sub2_t2_mocap <- read.csv(data_all_mocap[8])
data_sub2_t3_mocap <- read.csv(data_all_mocap[9])
data_sub2_t4_mocap <- read.csv(data_all_mocap[10])
data_sub2_t5_mocap <- read.csv(data_all_mocap[11])
data_sub2_t6_mocap <- read.csv(data_all_mocap[12])
data_sub3_t1_mocap <- read.csv(data_all_mocap[13])
data_sub3_t2_mocap <- read.csv(data_all_mocap[14])
data_sub3_t3_mocap <- read.csv(data_all_mocap[15])
data_sub3_t4_mocap <- read.csv(data_all_mocap[16])
data_sub3_t5_mocap <- read.csv(data_all_mocap[17])
data_sub4_t1_mocap <- read.csv(data_all_mocap[18])
data_sub5_t2_mocap <- read.csv(data_all_mocap[19])
data_sub5_t6_mocap <- read.csv(data_all_mocap[20])

