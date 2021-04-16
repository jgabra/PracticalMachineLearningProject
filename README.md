# PracticalMachineLearningProject
Project for Practical Machine Learning Class offered by JHU on Coursera

# Executive Summary
Machine learning models are used to predict the classe of weight lifting data from Velloso et al. 2013. Specifically, he data is from acceleromters on the participants belt, forearm, arm, and dumbell as the participants do curl exercies. In this particular data set, thee weight was lifted correctly (Classe = A) or incorreclty in 4 different manners (Classes B, C, D, E). There are 2 data sets, training and validation. The training data set is further divided into a training and a testing data set. The data is pre-processed to handle missing values and zero variance predictors. Various models are built using the training data set and tested for accuracy against the testing data set. The best model was the Gradient Boost Machine model that was then used to predict the classe of the validation data set. The final model accuratley predicted validation classes 100% of the time.

# Project Information

Project data comes from:   
Velloso, E.; Bulling, A.; Gellersen, H.; Ugulino, W.; Fuks, H. Qualitative Activity Recognition of Weight Lifting Exercises. Proceedings of 4th International Conference in Cooperation with SIGCHI (Augmented Human '13) . Stuttgart, Germany: ACM SIGCHI, 2013 

http://web.archive.org/web/20161224072740/http:/groupware.les.inf.puc-rio.br/har 

The data is from acceleromters on the participants belt, forearm, arm, and dumbell as the participants do curl exercies. In this particular data set, thee weight was lifted correctly (Classe = A) or incorreclty in 4 different manners (Classes B, C, D, E)