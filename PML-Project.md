---
title: "PML Project"
author: "JN Gabra"
date: "4/15/2021"
output: 
      html_document:
            keep_md: true
---
# Executive Summary
Machine learning models are used to predict the classe of weight lifting data from Velloso et al. 2013. Specifically, he data is from acceleromters on the participants belt, forearm, arm, and dumbell as the participants do curl exercies. In this particular data set, thee weight was lifted correctly (Classe = A) or incorreclty in 4 different manners (Classes B, C, D, E). There are 2 data sets, training and validation. The training data set is further divided into a training and a testing data set. The data is pre-processed to handle missing values and zero variance predictors. Various models are built using the training data set and tested for accuracy against the testing data set. The best model was the Gradient Boost Machine model that was then used to predict the classe of the validation data set. The final model accuratley predicted validation classes 100% of the time.

# Project Information

Project data comes from:   
Velloso, E.; Bulling, A.; Gellersen, H.; Ugulino, W.; Fuks, H. Qualitative Activity Recognition of Weight Lifting Exercises. Proceedings of 4th International Conference in Cooperation with SIGCHI (Augmented Human '13) . Stuttgart, Germany: ACM SIGCHI, 2013 

http://web.archive.org/web/20161224072740/http:/groupware.les.inf.puc-rio.br/har 

The data is from acceleromters on the participants belt, forearm, arm, and dumbell as the participants do curl exercies. In this particular data set, thee weight was lifted correctly (Classe = A) or incorreclty in 4 different manners (Classes B, C, D, E)

There are two data sets provided: **Training Data set** and **Testing Data**. *The testing data does not have the classe and is rather a validation data set.*

### Loading the Data Sets and Necessary Libraries  
This requires the use of the caret package. The data to be used is from the hyperlinks below but comes from Velloso et al. 2013. There is a training data set and a testing data set. It should be noted that the testing data set is acually a *testing validation* data set without the classe indication. The *"training data"* set is divided into an *acutal training data* set and a *test data* set. The training data set will be used to train the different models while the test data set will be used to evaluate the accuracy of each model to ultimately pick the best model to be used against the validation data set. 

Cross-validation method used is the random subsampling method to divide the "training data" set into an actual training data set and a testing data set. Specifically, 70% of the original training data is kept for model training while the remaining 30% is used for model testing.


```r
require(caret)
require(knitr)
training_data<-read.csv("https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv")
testing_validation<-read.csv("https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv")

set.seed(71890) #Cross-validation follows
inTrain = createDataPartition(training_data$classe, p = 0.7)[[1]]
training = training_data[ inTrain,-c(1:7)]
testing = training_data[-inTrain,]

training$classe<-as.factor(training$classe)
testing$classe<-as.factor(testing$classe)
```



### Data Pre-Processing  
First, we will do a barchart on the training data classe. Although not included for brevity, the training data was further explored for trends. It was revealed that there were a lot of missing values and for particular variables. These variables are removed from the training data set used in the models. In addition, the predictors were also analyzed to see if they are near zero variance predictors. If this was true, they were removed fromt the training data set used in model development.  

```r
barchart(training$classe)
```

![](PML-Project_files/figure-html/unnamed-chunk-2-1.png)<!-- -->

```r
na_count <-sapply(training, function(y) sum(length(which(is.na(y)))))
temp<-na_count==0
training_temp<-training[,temp]
nsv<-nearZeroVar(training_temp, saveMetrics = TRUE)
a<-nsv$nzv==FALSE
training_temp2<-training_temp[,a]
```
In the code above, the final training data used for model developement is "training_temp2".

### Building and Testing Models
The expected model error is about 30% (i.e. 70% accuracy) due to the fact that the classes are either correct motion (Classe A) or some variation in motion (Classes B through E). This is a very conservative estimate. 

The models used are all classifiers since our outcome is a factor variable. The models chosen are: classification and regression trees (rpart), linear discriminant analysis (lda), and gradient boosting machine (gbm), naive bayes (nb), and k-nearest neighbor (knn). Other models were run (random forest model) but are not included in this r markdown file for brevity as they take a while to run and do not produce better results. This was also true for principal component pre-processing included. Lastly, I also tried 2 different combinations to see if it would produce a better model than its subcomponents. Howver, they are not run in this document. The code is commented out.  

Each model is run against the testing data set to compare them to each other.


```r
set.seed(71890)
fit_rpart<-train(classe~.,method="rpart",data=training_temp2)
confusionMatrix(testing$classe,predict(fit_rpart,testing))
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1536   22  111    0    5
##          B  473  392  274    0    0
##          C  463   32  531    0    0
##          D  422  187  355    0    0
##          E  147  148  295    0  492
## 
## Overall Statistics
##                                           
##                Accuracy : 0.5014          
##                  95% CI : (0.4886, 0.5143)
##     No Information Rate : 0.5167          
##     P-Value [Acc > NIR] : 0.9909          
##                                           
##                   Kappa : 0.3486          
##                                           
##  Mcnemar's Test P-Value : NA              
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.5051  0.50192  0.33908       NA  0.98994
## Specificity            0.9515  0.85364  0.88539   0.8362  0.89050
## Pos Pred Value         0.9176  0.34416  0.51754       NA  0.45471
## Neg Pred Value         0.6426  0.91804  0.78699       NA  0.99896
## Prevalence             0.5167  0.13271  0.26610   0.0000  0.08445
## Detection Rate         0.2610  0.06661  0.09023   0.0000  0.08360
## Detection Prevalence   0.2845  0.19354  0.17434   0.1638  0.18386
## Balanced Accuracy      0.7283  0.67778  0.61224       NA  0.94022
```

```r
fit_lda<-train(classe~.,method="lda",data=training_temp2)
confusionMatrix(testing$classe,predict(fit_lda,testing))
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1373   41  132  122    6
##          B  173  738  123   45   60
##          C   90  101  680  128   27
##          D   48   42  116  716   42
##          E   46  187  107   95  647
## 
## Overall Statistics
##                                          
##                Accuracy : 0.7059         
##                  95% CI : (0.694, 0.7175)
##     No Information Rate : 0.294          
##     P-Value [Acc > NIR] : < 2.2e-16      
##                                          
##                   Kappa : 0.6279         
##                                          
##  Mcnemar's Test P-Value : < 2.2e-16      
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.7936   0.6655   0.5872   0.6474   0.8274
## Specificity            0.9276   0.9160   0.9268   0.9481   0.9148
## Pos Pred Value         0.8202   0.6479   0.6628   0.7427   0.5980
## Neg Pred Value         0.9152   0.9218   0.9016   0.9207   0.9719
## Prevalence             0.2940   0.1884   0.1968   0.1879   0.1329
## Detection Rate         0.2333   0.1254   0.1155   0.1217   0.1099
## Detection Prevalence   0.2845   0.1935   0.1743   0.1638   0.1839
## Balanced Accuracy      0.8606   0.7908   0.7570   0.7977   0.8711
```

```r
fit_gbm<-train(classe~.,method="gbm",data=training_temp2,verbose=FALSE)
confusionMatrix(testing$classe,predict(fit_gbm,testing))
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1654   16    1    1    2
##          B   37 1067   32    2    1
##          C    0   29  984   12    1
##          D    2    1   26  932    3
##          E    1   18   10   10 1043
## 
## Overall Statistics
##                                           
##                Accuracy : 0.9652          
##                  95% CI : (0.9602, 0.9697)
##     No Information Rate : 0.2879          
##     P-Value [Acc > NIR] : < 2.2e-16       
##                                           
##                   Kappa : 0.9559          
##                                           
##  Mcnemar's Test P-Value : 7.593e-06       
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.9764   0.9434   0.9345   0.9739   0.9933
## Specificity            0.9952   0.9849   0.9913   0.9935   0.9919
## Pos Pred Value         0.9881   0.9368   0.9591   0.9668   0.9640
## Neg Pred Value         0.9905   0.9865   0.9858   0.9949   0.9985
## Prevalence             0.2879   0.1922   0.1789   0.1626   0.1784
## Detection Rate         0.2811   0.1813   0.1672   0.1584   0.1772
## Detection Prevalence   0.2845   0.1935   0.1743   0.1638   0.1839
## Balanced Accuracy      0.9858   0.9641   0.9629   0.9837   0.9926
```


```r
## Unused data chunks for reference
fit_nb<-train(classe~.,method="nb",data=training_temp2)
confusionMatrix(testing$classe,predict(fit_nb,testing))
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1490   35   40   98   11
##          B  246  754   83   43   13
##          C  232   65  669   60    0
##          D  193    3  121  605   42
##          E   63  108   46   28  837
## 
## Overall Statistics
##                                           
##                Accuracy : 0.74            
##                  95% CI : (0.7286, 0.7512)
##     No Information Rate : 0.3779          
##     P-Value [Acc > NIR] : < 2.2e-16       
##                                           
##                   Kappa : 0.6671          
##                                           
##  Mcnemar's Test P-Value : < 2.2e-16       
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.6700   0.7813   0.6976   0.7254   0.9269
## Specificity            0.9497   0.9217   0.9275   0.9289   0.9508
## Pos Pred Value         0.8901   0.6620   0.6520   0.6276   0.7736
## Neg Pred Value         0.8257   0.9555   0.9403   0.9535   0.9863
## Prevalence             0.3779   0.1640   0.1630   0.1417   0.1534
## Detection Rate         0.2532   0.1281   0.1137   0.1028   0.1422
## Detection Prevalence   0.2845   0.1935   0.1743   0.1638   0.1839
## Balanced Accuracy      0.8099   0.8515   0.8126   0.8272   0.9389
```


```r
fit_knn<-train(classe~.,method="knn",data=training_temp2)
confusionMatrix(testing$classe,predict(fit_knn,testing))
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1616   20   17   17    4
##          B   54  983   52   23   27
##          C   12   26  941   29   18
##          D   19    6   67  859   13
##          E   18   52   33   28  951
## 
## Overall Statistics
##                                           
##                Accuracy : 0.9091          
##                  95% CI : (0.9015, 0.9163)
##     No Information Rate : 0.2921          
##     P-Value [Acc > NIR] : < 2.2e-16       
##                                           
##                   Kappa : 0.8849          
##                                           
##  Mcnemar's Test P-Value : 1.949e-12       
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.9401   0.9043   0.8477   0.8985   0.9388
## Specificity            0.9861   0.9675   0.9822   0.9787   0.9731
## Pos Pred Value         0.9654   0.8630   0.9172   0.8911   0.8789
## Neg Pred Value         0.9755   0.9781   0.9652   0.9803   0.9871
## Prevalence             0.2921   0.1847   0.1886   0.1624   0.1721
## Detection Rate         0.2746   0.1670   0.1599   0.1460   0.1616
## Detection Prevalence   0.2845   0.1935   0.1743   0.1638   0.1839
## Balanced Accuracy      0.9631   0.9359   0.9150   0.9386   0.9560
```




```r
data<-data.frame(Model=c("rpart","lda","gbm","nb","knn"),Accuracy=c(confusionMatrix(testing$classe,predict(fit_rpart,testing))$overall[1],confusionMatrix(testing$classe,predict(fit_lda,testing))$overall[1],confusionMatrix(testing$classe,predict(fit_gbm,testing))$overall[1],confusionMatrix(testing$classe,predict(fit_nb,testing))$overall[1],confusionMatrix(testing$classe,predict(fit_knn,testing))$overall[1]))  
#data<-data.frame(Model=c("rpart","lda","gbm","nb","knn"),Accuracy=c(confusionMatrix(testing$classe,predict(fit_rpart,testing))$overall[1],confusionMatrix(testing$classe,predict(fit_lda,testing))$overall[1],confusionMatrix(testing$classe,predict(fit_gbm,testing))$overall[1],0.74,0.9076))
data[,2]<-round(data[,2],2)*100
names(data)[2]<-"Accuracy [%]"
kable(data)
```



|Model | Accuracy [%]|
|:-----|------------:|
|rpart |           50|
|lda   |           71|
|gbm   |           97|
|nb    |           74|
|knn   |           91|
From the Model vs Accuracy table, we can see that the gradient boost machine model produces the best accuracy at 96%. This model will be used to run against the *testing validation* data set (i.e. the same data set for the quiz).

### Testing the final model on the validation data set

```r
Quiz_Answers<-predict(fit_gbm,testing_validation)
#Quiz_Answers
```
The quiz answers were submitted to the online validation data set quiz and the final Gradient Boost Model was 100% accurate for the validation data set.

