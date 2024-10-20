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

# target variable ist leicht rechtsverschoben -> transformation um normalverteilung darzustellen
sauber$chandler_score <- log(sauber$chandler_score)

# filtern von features welche sehr niedrige varianz haben
nearZeroVar(sauber, saveMetrics = TRUE) %>% rownames_to_column() %>% filter(nzv)

# standardisierung, normalisierung & pca
#blueprint <- recipe(chandler_score ~ ., data = sauber_train_gbm) %>%
  #step_YeoJohnson(all_numeric()) %>%
  #step_center(all_numeric(), -all_outcomes()) %>%
  #step_scale(all_numeric(), -all_outcomes()) %>% 
  #step_pca(all_numeric(), -all_outcomes(),  threshold = .95)

#sauber <- sauber %>% select('age', 'apoe_stat', 'SCDImem', '3IL_tot', 'BRS_tot', 'LSNS_tot', 'GDS_tot', 'BMI', 'weight', 'RR_sys', 'MAT_tot', 'tobacco_frequency', 'education','sex', 'chandler_score')

# aufsetzen von trainings und testsplit 70/30 + stratify(balanced distribution von target variable)
index_sauber_gbm <- initial_split(sauber, prop = 0.7, strata = 'chandler_score')
sauber_train_gbm <- training(index_sauber_gbm)
sauber_test_gbm <- testing(index_sauber_gbm)

# anwenden von blueprint
#prepare <- prep(blueprint, training = sauber_train_gbm)
#trainfin <- bake(prepare, new_data = sauber_train_gbm)
#testfin <- bake(prepare, new_data = sauber_test_gbm)

# model
sauber_gbm <- gbm(
  formula = chandler_score ~ .,
  data = sauber_train_gbm,
  distribution = "gaussian",
  n.trees = 5000,
  shrinkage = 0.01,
  interaction.depth = 15,
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
sauber_predictions_gbm <- exp(sauber_predictions_gbm)

# mean squared error zwischen predicted chandler_score und realem chandler_score
sauber_mse_gbm <- sqrt(mean((exp(sauber_test_gbm$chandler_score) - sauber_predictions_gbm)^2))

# mean squared error zwischen mean und real
mean1 <- mean(exp(sauber_train_gbm$chandler_score))
pcr_mse_stupid <- sqrt(mean((exp(sauber_test_gbm$chandler_score) - mean1)^2))

# variablen gewichtung plotten
plot1 <- plot(varImp(sauber_gbm,scale=TRUE))
print(plot1)
