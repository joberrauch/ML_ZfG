# Versuch RFOREST wenig_daten

package_packages <- c('tidyverse', 'mlr3', 'glmnet', 'caret', 'data.table', 'ggplot2', 'randomForest')
install.packages(package_packages)
library(tidyverse)
library(glmnet)
library(mlr3)
library(caret)
library(data.table)
library(ggplot2)
library(randomForest)

wenig_daten <- fread('C:/Users/Johanner/R projects/Resilienz/wenig_daten.csv')

head(wenig_daten)
str(wenig_daten)
set.seed(2)

# aufsetzen von trainings und testsplit 70/30
index_wenig_rf <- sample(2, nrow(wenig_daten), replace = TRUE, prob = c(0.7, 0.3))
wenig_train_rf <- wenig_daten[index_wenig_rf == 1, ]
wenig_test_rf <- wenig_daten[index_wenig_rf == 2, ]

# control parameter
wenig_control_rf <- trainControl(method = 'repeatedcv', number = 10, repeats = 5, verboseIter = TRUE)

# model fitting
wenig_rf <- train(chandler_score~., wenig_train_rf,  method = 'rf', ntreeTry = 500, mtryStart = 4, stepFactor = 1.5, improve = 0.01, trControl = wenig_control_rf)

# variablen gewichtung plotten
wenig_rf_plot <- plot(varImp(wenig_rf,scale=TRUE))

# predictions fuer wenig_test_rf
wenig_predictions_rf <- predict(wenig_rf, newdata = wenig_test_rf)

# mean squared error zwischen predicted chandler_score und realem chandler_score
wenig_mse_rf <- mean((wenig_test_rf$chandler_score - wenig_predictions_rf)^2)
wenig_mse_rf 
