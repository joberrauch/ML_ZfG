# Versuch ELNET alles_daten_test

package_packages <- c('tidyverse', 'mlr3', 'glmnet', 'caret', 'data.table', 'ggplot2', 'rsample','recipes')
install.packages(package_packages)
library(tidyverse)
library(glmnet)
library(mlr3)
library(caret)
library(data.table)
library(ggplot2)
library(rsample)
library(recipes)

alles_daten_test <- fread('C:/Users/Johanner/R projects/Resilienz/alles_daten_test.csv')

head(alles_daten_test)
str(alles_daten_test)
set.seed(2)

# aufsetzen von trainings und testsplit 70/30
index_alles_en <- sample(2, nrow(alles_daten_test), replace = TRUE, prob = c(0.7, 0.3))
alles_train_en <- alles_daten_test[index_alles_en == 1, ]
alles_test_en <- alles_daten_test[index_alles_en == 2, ]

# control parameter
alles_control_en <- trainControl(method = 'repeatedcv', number = 10, repeats = 5, verboseIter = TRUE)

# model fitting
alles_elnet <- train(chandler_score~., alles_train_en,  method = 'glmnet', tuneGrid = expand.grid(alpha = seq(0,1,length = 10), lambda = seq(0.0001, 0.2, length = 5)), 
trControl = alles_control_en)

# variablen gewichtung plotten
alles_elnet_plot <- plot(varImp(alles_elnet,scale=TRUE))

# predictions fuer alles_test_en
alles_predictions_en <- predict(alles_elnet, newdata = alles_test_en)

# mean squared error zwischen predicted chandler_score und realem chandler_score
alles_mse_en <- mean((alles_test_en$chandler_score - alles_predictions_en)^2)
alles_mse_en
