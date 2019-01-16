rm(list=ls(all=T))
setwd("E:/Subject/edwisor")

# Importing the dataset

dataset = read.csv('day.csv')
dataset1 = dataset[3:16]

# Building the optimal model using Backward Elimination

regressor = lm(formula = cnt ~ 
      season+ yr+mnth+holiday+weekday+workingday+weathersit+temp+atemp+hum+windspeed+casual+registered,
               data = dataset1)
summary(regressor)

regressor = lm(formula = cnt ~ casual+registered,
               data = dataset1)
summary(regressor)

final_dataset=regressor[["model"]]
final_dataset1=as.vector(final_dataset[2:3])

#Ploting outliers
boxplot(final_dataset1$casual,horizontal = T)
hist(final_dataset1$casual)

boxplot(final_dataset1$registered,horizontal = T)

quantiles =quantile(final_dataset$casual, probs = c(.25, .75))
range =1.5 * IQR(final_dataset$casual)
pre_process_data = subset(final_dataset,
                     final_dataset$casual > (quantiles[1] - range) & 
                       final_dataset$casual < (quantiles[2] + range))

boxplot(pre_process_data,horizontal = T)
hist(pre_process_data$casual)

#change order of column number 
pre_process_data=pre_process_data[c(2,3,1)]


# Splitting the dataset into the Training set and Test set
# install.packages('caTools')
library(caTools)
set.seed(123)
split = sample.split(pre_process_data$cnt, SplitRatio = 0.8)
training_set = subset(pre_process_data, split == TRUE)
test_set = subset(pre_process_data, split == FALSE)

# Feature Scaling
training_set[-3] = scale(training_set[-3])
test_set[-3] = scale(test_set[-3])

# Fitting Random Forest Classification to the Training set
#install.packages('randomForest')
library(randomForest)
set.seed(123)
classifier = randomForest(x = training_set[-3],
                          y = training_set$cnt,
                          ntree = 500)

# Predicting the Test set results

y_pred = data.frame(predict(classifier, newdata = test_set[-3]))
colnames(y_pred)=c("Prediction")

#Merge test case and prediction to visualize result
res=data.frame(test_set$cnt,round(y_pred,digits=0))

colnames(res)=c("test_case","prediction")
MAE(y_test,y_pred)
