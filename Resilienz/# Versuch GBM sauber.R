# Versuch GBM sauber

package_packages <- c('tidyverse', 'mlr3', 'glmnet', 'caret', 'data.table', 'ggplot2', 'gbm','h2o', 'xgboost')
#install.packages(package_packages)
library(tidyverse)
library(glmnet)
library(mlr3)
library(caret)
library(data.table)
library(ggplot2)
library(rsample)
library(recipes)
library(gbm)
library(h2o)
library(xgboost)
library(data.table)

#x <- sample(1:400,1)
#292s
set.seed(292)

sauber <- fread('C:/Users/Johanner/R projects/Resilienz/alles_daten_test.csv')

# sex von weiblich zu 0/maennlich zu 1; fct
sauber$sex <- sauber$sex %>% str_replace_all('Weiblich', '0')
sauber$sex <- sauber$sex %>% str_replace_all('Männlich', '1')
sauber$sex <- as_factor(sauber$sex)

# apoe_stat von int zu fct
sauber$apoe_stat <- as_factor(sauber$apoe_stat)

# alcohol_consumption zu fct; Niemals = 0; 0-7 = 1; 8-14 = 2; >14 = 3
sauber$alcohol_consumption <- sauber$alcohol_consumption %>% str_replace_all('Niemals', '0')
sauber$alcohol_consumption <- sauber$alcohol_consumption %>% str_replace_all('0-7 pro Woche', '1')
sauber$alcohol_consumption <- sauber$alcohol_consumption %>% str_replace_all('8-14 pro Woche', '2')
sauber$alcohol_consumption <- sauber$alcohol_consumption %>% str_replace_all('> 14 pro Woche', '3')
sauber$alcohol_consumption <- as_factor(sauber$alcohol_consumption)

# risk_sports zu fct; Niemals = 0; In der Vergangenheit = 1; Aktuell = 2
sauber$risk_sports <- sauber$risk_sports %>% str_replace_all('Niemals', '0')
sauber$risk_sports <- sauber$risk_sports %>% str_replace_all('In der Vergangenheit', '1')
sauber$risk_sports <- sauber$risk_sports %>% str_replace_all('Aktuell', '2')
sauber$risk_sports <- as_factor(sauber$risk_sports)

# WVT_tot_left, WVT_tot_right & SCDImem von int zu fct
sauber$WVT_tot_left <- as_factor(sauber$WVT_tot_left)
sauber$WVT_tot_right <- as_factor(sauber$WVT_tot_right)
sauber$SCDImem <- as_factor(sauber$SCDImem)

# tobacco_frequency zu fct; Nichtraucher = 0; Gelegentlich = 1; Täglich = 2
sauber$tobacco_frequency <- sauber$tobacco_frequency %>% str_replace_all('Nichtraucher', '0')
sauber$tobacco_frequency <- sauber$tobacco_frequency %>% str_replace_all('Gelegentlich', '1')
sauber$tobacco_frequency <- sauber$tobacco_frequency %>% str_replace_all('Täglich', '2')
sauber$tobacco_frequency <- as_factor(sauber$tobacco_frequency)

#sauber <- sauber %>% select('age', 'BRS_tot', 'LSNS_tot', 'GDS_tot', 'BMI', 'chandler_score')

# aufsetzen von trainings und testsplit 70/30
index_sauber_gbm <- sample(2, nrow(sauber), replace = TRUE, prob = c(0.7, 0.3))
sauber_train_gbm <- sauber[index_sauber_gbm == 1, ]
sauber_test_gbm <- sauber[index_sauber_gbm == 2, ]

# model
sauber_gbm <- gbm(
  formula = chandler_score ~ .,
  data = sauber_train_gbm,
  distribution = "gaussian",
  n.trees = 10000,
  shrinkage = 0.01,
  interaction.depth = 10,
  n.minobsinnode = 15,
  cv.folds = 20
)

# find index for number trees with minimum CV error
best <- which.min(sauber_gbm$cv.error)

# get MSE and compute RMSE
sqrt(sauber_gbm$cv.error[best])

# plot error curve
gbm.perf(sauber_gbm, method = "cv")

# predictions fuer sauber_test_gbm
sauber_predictions_gbm <- predict(sauber_gbm, newdata = sauber_test_gbm)

# mean squared error zwischen predicted chandler_score und realem chandler_score
sauber_mse_gbm <- sqrt(mean((sauber_test_gbm$chandler_score - sauber_predictions_gbm)^2))

# variablen gewichtung plotten
plot1 <- plot(varImp(sauber_gbm,scale=TRUE), 
  main = as.character(x), 
  sub = as.character(sauber_mse_gbm)
)
print(plot1)
