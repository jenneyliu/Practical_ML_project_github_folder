Practical Machine Learning Project
========================================================


In this project, we use data from accelerometers on the belt, forearm, arm, and dumbell of 6 participants to predict the manner in which participants did the exercise.   This is the "classe" variable in the training set. 









##  Load package, load and preprocessing data


```r
library(caret)
```


```r
# Downloading the data
dataUrl  <- "https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv"
file  <- "pml-training.csv"
download.file(url=dataUrl, destfile=file, method="curl")

# load data
trainRawData <- read.csv("pml-training.csv",na.strings=c("NA",""))
```

There are missing data in the dataset, we remove the columns which contain missing data.

```r
# discard NAs
col_NumNA <- apply(trainRawData,2,function(x) {sum(is.na(x))}) 
# col_NumNA <- apply(trainRawData,2,function(x) {length(which((is.na(x))))}) 
rmNAData <- trainRawData[, which(col_NumNA == 0)]
```


Separate the data into two data sets, we will train models on one dataset ( trainData ), and estimate the out of sample error on the other ( testErrorEstimateData ).  



```r
trainIndex <- createDataPartition(y = rmNAData$classe, p=0.2,list=FALSE) 
trainData <- rmNAData[trainIndex,]
testErrorEstimateData <- rmNAData[-trainIndex,]
```



##  Cross validation setting
We set 10-fold cross validation to tune the parameters of the prediction models.


```r
fitControl <- trainControl("cv", number=10, repeats=10, classProbs=TRUE, savePred=T) 
```

##  Built the  model

In this project, we tried two models,  Recursive Partitioning and Regression Trees (method="rpart"), and Random forest (method="rf").  

R code and result for the Recursive Partitioning and Regression Trees model (method="rpart") are as below. 

```r
modFit_rpart <- train(trainData$classe ~., data = trainData, method="rpart", trControl=fitControl)
```


```r
modFit_rpart
```

```
## 3927 samples
##   59 predictors
##    5 classes: 'A', 'B', 'C', 'D', 'E' 
## 
## No pre-processing
## Resampling: Cross-Validation (10 fold) 
## 
## Summary of sample sizes: 3534, 3535, 3533, 3536, 3535, 3535, ... 
## 
## Resampling results across tuning parameters:
## 
##   cp   Accuracy  Kappa  Accuracy SD  Kappa SD
##   0.2  0.7       0.7    0.09         0.1     
##   0.3  0.6       0.5    0.09         0.1     
##   0.3  0.3       0.1    0.09         0.2     
## 
## Accuracy was used to select the optimal model using  the largest value.
## The final value used for the model was cp = 0.2.
```

```r
train_pred <- predict(modFit_rpart, newdata=trainData)
train_error_rpart <- confusionMatrix(data=trainData$classe, train_pred)
train_error_rpart
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1116    0    0    0    0
##          B    0  760    0    0    0
##          C    0    0    0    0  685
##          D    0    0    0    0  644
##          E    0    0    0    0  722
## 
## Overall Statistics
##                                         
##                Accuracy : 0.662         
##                  95% CI : (0.647, 0.676)
##     No Information Rate : 0.522         
##     P-Value [Acc > NIR] : <2e-16        
##                                         
##                   Kappa : 0.569         
##  Mcnemar's Test P-Value : NA            
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity             1.000    1.000       NA       NA    0.352
## Specificity             1.000    1.000    0.826    0.836    1.000
## Pos Pred Value          1.000    1.000       NA       NA    1.000
## Neg Pred Value          1.000    1.000       NA       NA    0.585
## Prevalence              0.284    0.194    0.000    0.000    0.522
## Detection Rate          0.284    0.194    0.000    0.000    0.184
## Detection Prevalence    0.284    0.194    0.174    0.164    0.184
```


R code and result for the Random forest model (method="rf") are as below. 

```r
#   Recursive Partitioning and Regression Trees (method="rpart")
modFit_rf <- train(trainData$classe ~., data = trainData, method="rf", trControl=fitControl)
```


```r
modFit_rf
```

```
## 3927 samples
##   59 predictors
##    5 classes: 'A', 'B', 'C', 'D', 'E' 
## 
## No pre-processing
## Resampling: Cross-Validation (10 fold) 
## 
## Summary of sample sizes: 3535, 3535, 3534, 3534, 3536, 3534, ... 
## 
## Resampling results across tuning parameters:
## 
##   mtry  Accuracy  Kappa  Accuracy SD  Kappa SD
##   2     1         1      0.006        0.007   
##   40    1         1      8e-04        0.001   
##   80    1         1      0.001        0.002   
## 
## Accuracy was used to select the optimal model using  the largest value.
## The final value used for the model was mtry = 41.
```

```r
train_pred <- predict(modFit_rf, newdata=trainData)
train_rf <- confusionMatrix(data=trainData$classe, train_pred)
train_rf
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1116    0    0    0    0
##          B    0  760    0    0    0
##          C    0    0  685    0    0
##          D    0    0    0  644    0
##          E    0    0    0    0  722
## 
## Overall Statistics
##                                     
##                Accuracy : 1         
##                  95% CI : (0.999, 1)
##     No Information Rate : 0.284     
##     P-Value [Acc > NIR] : <2e-16    
##                                     
##                   Kappa : 1         
##  Mcnemar's Test P-Value : NA        
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity             1.000    1.000    1.000    1.000    1.000
## Specificity             1.000    1.000    1.000    1.000    1.000
## Pos Pred Value          1.000    1.000    1.000    1.000    1.000
## Neg Pred Value          1.000    1.000    1.000    1.000    1.000
## Prevalence              0.284    0.194    0.174    0.164    0.184
## Detection Rate          0.284    0.194    0.174    0.164    0.184
## Detection Prevalence    0.284    0.194    0.174    0.164    0.184
```


From the error on the train data, we found that Random forest model has much higher accuracy than the Recursive Partitioning and Regression Trees model.  Specially, the trainning accuracy of the Random forest model is 100%, much larger than that of the Recursive Partitioning and Regression Trees model which is 66.2%.  This can also be seen from the following bar-chart.  So we use the Random forest model as the final model.


![plot of chunk unnamed-chunk-12](figure/unnamed-chunk-12.png) 



##  The expected out of sample error
We will compute the error on the left-out datset that we did not perform any training on it, so that we will get relatively realistic estimate of the out of sample error.


```r
testErrorEstimate_pred <- predict(modFit_rf, newdata=testErrorEstimateData)
testErrorEstimate_rf <- confusionMatrix(data=testErrorEstimateData$classe, testErrorEstimate_pred)
testErrorEstimate_rf
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 4464    0    0    0    0
##          B    0 3034    3    0    0
##          C    0    0 2737    0    0
##          D    0    0    0 2572    0
##          E    0    0    0    0 2885
## 
## Overall Statistics
##                                     
##                Accuracy : 1         
##                  95% CI : (0.999, 1)
##     No Information Rate : 0.284     
##     P-Value [Acc > NIR] : <2e-16    
##                                     
##                   Kappa : 1         
##  Mcnemar's Test P-Value : NA        
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity             1.000    1.000    0.999    1.000    1.000
## Specificity             1.000    1.000    1.000    1.000    1.000
## Pos Pred Value          1.000    0.999    1.000    1.000    1.000
## Neg Pred Value          1.000    1.000    1.000    1.000    1.000
## Prevalence              0.284    0.193    0.175    0.164    0.184
## Detection Rate          0.284    0.193    0.174    0.164    0.184
## Detection Prevalence    0.284    0.194    0.174    0.164    0.184
```

From this R code result, the expected out of sample error is very small and the expected out of sample accuracy is very high, which is 100%.


##  Prediction of 20 different test cases

Finally, we will use the Random forest model to predict 20 different test cases.


```r
dataUrl  <- "https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv"
file  <- "pml-testing.csv"
download.file(url=dataUrl, destfile=file, method="curl")



testRawData <- read.csv("pml-testing.csv",na.strings=c("NA",""))
problem_id <- testRawData$problem_id
testRawData$classe <- NA
colind <- which(colnames(testRawData) %in% colnames(trainData))
testData <- testRawData[, colind]

# identical(colnames(trainData), colnames(testData))
# [1] TRUE


test_pred <- predict(modFit_rf, newdata=testData)
answers <- as.character( test_pred )

pml_write_files = function(x){
  n = length(x)
  for(i in 1:n){
    filename = paste0("./answers/problem_id_",i,".txt")
    write.table(x[i],file=filename,quote=FALSE,row.names=FALSE,col.names=FALSE)
  }
}

pml_write_files(answers)
```

After submitting this result to the Coursera programming assignment, the feedback confirm that the accruracy on these 20 different test cases is 100%.

