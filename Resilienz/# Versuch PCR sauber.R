# Versuch PCR sauber.R

package_packages <- c('tidyverse', 'mlr3', 'glmnet', 'caret', 'data.table', 'ggplot2', 'gbm','h2o', 'pls', 'xgboost')
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
library(pls)

#x <- sample(1:400,1)
#292s
set.seed(2)

sauber <- fread('C:/Users/Johanner/R projects/Resilienz/alles_daten_test.csv')

# sex von weiblich zu 0/maennlich zu 1; fct
sauber$sex <- sauber$sex %>% str_replace_all('Weiblich', '0')
sauber$sex <- sauber$sex %>% str_replace_all('Männlich', '1')
sauber$sex <- as.integer(sauber$sex)

# apoe_stat von int zu fct
sauber$apoe_stat <- as.integer(sauber$apoe_stat)

# alcohol_consumption zu fct; Niemals = 0; 0-7 = 1; 8-14 = 2; >14 = 3
sauber$alcohol_consumption <- sauber$alcohol_consumption %>% str_replace_all('Niemals', '0')
sauber$alcohol_consumption <- sauber$alcohol_consumption %>% str_replace_all('0-7 pro Woche', '1')
sauber$alcohol_consumption <- sauber$alcohol_consumption %>% str_replace_all('8-14 pro Woche', '2')
sauber$alcohol_consumption <- sauber$alcohol_consumption %>% str_replace_all('> 14 pro Woche', '3')
sauber$alcohol_consumption <- as.integer(sauber$alcohol_consumption)

# risk_sports zu fct; Niemals = 0; In der Vergangenheit = 1; Aktuell = 2
sauber$risk_sports <- sauber$risk_sports %>% str_replace_all('Niemals', '0')
sauber$risk_sports <- sauber$risk_sports %>% str_replace_all('In der Vergangenheit', '1')
sauber$risk_sports <- sauber$risk_sports %>% str_replace_all('Aktuell', '2')
sauber$risk_sports <- as.integer(sauber$risk_sports)

# WVT_tot_left, WVT_tot_right & SCDImem von int zu fct
sauber$WVT_tot_left <- as.integer(sauber$WVT_tot_left)
sauber$WVT_tot_right <- as.integer(sauber$WVT_tot_right)
sauber$SCDImem <- as.integer(sauber$SCDImem)

# tobacco_frequency zu fct; Nichtraucher = 0; Gelegentlich = 1; Täglich = 2
sauber$tobacco_frequency <- sauber$tobacco_frequency %>% str_replace_all('Nichtraucher', '0')
sauber$tobacco_frequency <- sauber$tobacco_frequency %>% str_replace_all('Gelegentlich', '1')
sauber$tobacco_frequency <- sauber$tobacco_frequency %>% str_replace_all('Täglich', '2')
sauber$tobacco_frequency <- as.integer(sauber$tobacco_frequency)

# aufsetzen von trainings und testsplit 70/30
index_sauber_pcr <- sample(2, nrow(sauber), replace = TRUE, prob = c(0.7, 0.3))
sauber_train_pcr <- sauber[index_sauber_pcr == 1, ]
sauber_test_pcr <- sauber[index_sauber_pcr == 2, ]

pca <- prcomp(sauber, scale = TRUE)

#reverse the signs
pca$rotation <- -1*pca$rotation

#reverse the signs of the scores
pca$x <- -1*pca$x

#display principal components
pca$rotation

#plot
biplot(pca, scale = 0)

#calculate total variance explained by each principal component
tot_var <- pca$sdev^2 / sum(pca$sdev^2)

#fit PCR model
model <- pcr(chandler_score ~., data = sauber, scale = TRUE, validation = "CV")

#visualize cross-validation plots
validationplot(model)

#train model with training data
model <- pcr(chandler_score ~., data = sauber_train_pcr, scale = TRUE, validation = "CV")
summary(model)

#make predictions on test data with 15 comp
pcr_pred <- predict(model, sauber_test_pcr, ncomp=8)

#calculate RMSE
pcr_rmse <- sqrt(mean((pcr_pred - sauber_test_pcr$chandler_score)^2))

# mean squared error zwischen mean und real
mean1 <- mean(sauber_train_pcr$chandler_score)
pcr_mse_stupid <- sqrt(mean((sauber_test_pcr$chandler_score - mean1)^2))