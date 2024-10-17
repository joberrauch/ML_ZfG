# Versuch ELNET sauber

package_packages <- c('tidyverse', 'mlr3', 'glmnet', 'caret', 'data.table', 'ggplot2', 'rsample','recipes')
#install.packages(package_packages)
library(tidyverse)
library(glmnet)
library(mlr3)
library(caret)
library(data.table)
library(ggplot2)
library(rsample)
library(recipes)

set.seed(2)

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

# aufsetzen von trainings und testsplit 70/30
index_sauber_en <- sample(2, nrow(sauber), replace = TRUE, prob = c(0.7, 0.3))
sauber_train_en <- sauber[index_sauber_en == 1, ]
sauber_test_en <- sauber[index_sauber_en == 2, ]

# control parameter
sauber_control_en <- trainControl(method = 'repeatedcv', number = 10, repeats = 5, verboseIter = TRUE)

# model fitting
sauber_elnet <- train(chandler_score~., sauber_train_en,  method = 'glmnet', tuneGrid = expand.grid(alpha = seq(0,1,length = 10), lambda = seq(0.0001, 0.2, length = 5)), 
trControl = sauber_control_en)

# plotten
plot(varImp(sauber_elnet,scale=TRUE))
ggplot(sauber_elnet)

# predictions fuer wenig_test
sauber_predictions_en <- predict(sauber_elnet, newdata = sauber_test_en)

# mean squared error zwischen predicted chandler_score und realem chandler_score
sauber_mse_en <- mean((sauber_test_en$chandler_score - sauber_predictions_en)^2)