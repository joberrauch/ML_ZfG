# Versuch RFOREST alles_daten_test

package_packages <- c('tidyverse', 'mlr3', 'glmnet', 'caret', 'data.table', 'ggplot2', 'randomForest')
install.packages(package_packages)
library(tidyverse)
library(glmnet)
library(mlr3)
library(caret)
library(data.table)
library(ggplot2)
library(randomForest)

alles_daten_test <- fread('C:/Users/Johanner/R projects/Resilienz/alles_daten_test.csv')

head(alles_daten_test)
str(alles_daten_test)
set.seed(2)

# aufsetzen von trainings und testsplit 70/30
index_alles_rf <- sample(2, nrow(alles_daten_test), replace = TRUE, prob = c(0.7, 0.3))
alles_train_rf <- alles_daten_test[index_alles_rf == 1, ]
alles_test_rf <- alles_daten_test[index_alles_rf == 2, ]

# control parameter
alles_control_rf <- trainControl(method = 'repeatedcv', number = 10, repeats = 5, verboseIter = TRUE)

# model fitting
alles_rf <- train(chandler_score~., alles_train_rf,  method = 'rf', ntreeTry = 500, mtryStart = 4, stepFactor = 1.5, improve = 0.01, trControl = alles_control_rf)

# variablen gewichtung plotten
alles_rf_plot <- plot(varImp(alles_rf,scale=TRUE))

# predictions fuer alles_test
alles_predictions_rf <- predict(alles_rf, newdata = alles_test_rf)

# mean squared error zwischen predicted chandler_score und realem chandler_score
alles_mse_rf <- mean((alles_test_rf$chandler_score - alles_predictions_rf)^2)
alles_mse_rf 
