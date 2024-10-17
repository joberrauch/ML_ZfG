# Versuch ELNET volumina_alles

package_packages <- c('tidyverse', 'mlr3', 'glmnet', 'caret', 'data.table', 'ggplot2', 'randomForest', 'keras', 'tfruns', 'tfestimators')
#install.packages(package_packages)
library(tidyverse)
library(glmnet)
library(mlr3)
library(caret)
library(data.table)
library(ggplot2)
library(randomForest)
library(keras)
library(tfruns)
library(tfestimators)

vol_alles <- readxl::read_xlsx('C:/Users/Johanner/R projects/MRIvolumina/ZfG_APOE4_Volumina.xlsx')

# gebdatum zu alter
geb_als_datum <- as.Date(vol_alles$Gebdatum)
heute <- Sys.Date()
age <- as.numeric(difftime(heute, geb_als_datum, units = "weeks")) %/% 52.25
vol_alles$Gebdatum <- age
vol_alles$Gebdatum <- vol_alles$Gebdatum %>% as.integer(vol_alles$Gebdatum)
names(vol_alles)[names(vol_alles)=="Gebdatum"] <- "Alter"

# APOE von chr zu fct
vol_alles$APOE <- as.factor(vol_alles$APOE)

head(vol_alles)
str(vol_alles)
set.seed(2)

# aufsetzen von trainings und testsplit 70/30
index_vol_alles_en <- sample(2, nrow(vol_alles), replace = TRUE, prob = c(0.7, 0.3))
vol_alles_train_en <- vol_alles[index_vol_alles_en == 1, ]
vol_alles_test_en <- vol_alles[index_vol_alles_en == 2, ]

# control parameter
vol_alles_control_en <- trainControl(method = 'repeatedcv', number = 10, repeats = 5, verboseIter = TRUE)

# model fitting
vol_alles_elnet <- fit(APOE~., vol_alles_train_en,  method = 'glmnet', tuneGrid = expand.grid(alpha = seq(0,1,length = 10), lambda = seq(0.0001, 0.2, length = 5)), trControl = vol_alles_control_en)

# variablen gewichtung plotten
vol_alles_elnet_plot <- plot(varImp(vol_alles_elnet,scale=TRUE))

# predictions fuer wenig_test
voll_alles_predictions_en <- predict(vol_alles_elnet, newdata = vol_alles_test_en)

# nicht mean squared error zwischen predicted chandler_score und realem chandler_score
vol_alles_mse_en <- (vol_alles_test_en$APOE == voll_alles_predictions_en)
vol_alles_TRUE <- as.character(vol_alles_mse_en)
vol_alles_TRUE <- as.data.frame(vol_alles_TRUE)
dumdum <- vol_alles_TRUE %>% filter(vol_alles_TRUE == 'TRUE')
