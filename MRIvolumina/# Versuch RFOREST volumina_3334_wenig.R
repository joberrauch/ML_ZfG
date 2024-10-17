# Versuch ELNET volumina_3334_wenig.R

package_packages <- c('tidyverse', 'mlr3', 'glmnet', 'caret', 'data.table', 'ggplot2', 'randomForest')
install.packages(package_packages)
library(tidyverse)
library(glmnet)
library(mlr3)
library(caret)
library(data.table)
library(ggplot2)
library(randomForest)

vol_alles <- readxl::read_xlsx('C:/Users/Johanner/R projects/MRIvolumina/ZfG_APOE4_Volumina.xlsx')

# aus vol_alles vol_weniger und dann vol_3334 machen
vol_weniger <- vol_alles %>% select(1:39)
vol_weniger <- vol_weniger %>% select(!Gebdatum)
vol_3334 <- filter(vol_weniger, vol_weniger$APOE == 34 | vol_weniger$APOE == 33)
vol_3334_wenig <- select(vol_3334, c('APOE','Norm_R Pallidum_Volumen', 'Norm_R Gyrus Parahippocampalis_Volumen', 'Norm_R Putamen_Volumen', 'Norm_L Putamen_Volumen', 'Norm_Vierter Ventrikel_Volumen', 'Norm_L Regio Entorhinalis_Volumen', 'Norm_L Gyrus Parahippocampalis_Volumen', 'Norm_L Pallidum_Volumen', 'Norm_Mesenzephalon_Volumen', 'Norm_Dritter Ventrikel_Volumen', 'Norm_R Hippocampus_Volumen', 'Norm_L Hippocampus_Volumen','Norm_L Thalamus_Volumen', 'Norm_R Thalamus_Volumen', 'Norm_Hirnstamm_Volumen', 'Norm_R Nucleus Caudatus_Volumen', 'Norm_L Nucleus Caudatus_Volumen','Norm_R Regio Entorhinalis_Volumen', 'Norm_Pons_Volumen'))

# APOE von chr zu fct
vol_3334_wenig$APOE <- as.factor(vol_3334_wenig$APOE)

head(vol_3334_wenig)
str(vol_3334_wenig)
set.seed(15)

# aufsetzen von trainings und testsplit 70/30
index_vol_3334_wenig_rf <- sample(2, nrow(vol_3334_wenig), replace = TRUE, prob = c(0.665, 0.335))
vol_3334_wenig_train_rf <- vol_3334_wenig[index_vol_3334_wenig_rf == 1, ]
vol_3334_wenig_test_rf <- vol_3334_wenig[index_vol_3334_wenig_rf == 2, ]

# control parameter
vol_3334_wenig_control_rf <- trainControl(method = 'repeatedcv', number = 5, repeats = 5, verboseIter = TRUE)

# model fitting
vol_3334_wenig_rf <- train(APOE~., vol_3334_wenig_train_rf,  method = 'rf', ntreeTry = 500, mtryStart = 4, stepFactor = 1.5, improve = 0.01, trControl = vol_3334_wenig_control_rf)

# variablen gewichtung plotten
plot(varImp(vol_3334_wenig_rf,scale=TRUE))

# predictions fuer alles_test
vol_3334_wenig_predictions_rf <- predict(vol_3334_wenig_rf, newdata = vol_3334_wenig_test_rf)

# nicht mean squared error zwischen predicted chandler_score und realem chandler_score
vol_3334_wenig_eval_rf <- (vol_3334_wenig_test_rf$APOE == vol_3334_wenig_predictions_rf)
vol_3334_wenig_TRUE_rf <- as.character(vol_3334_wenig_eval_rf)
vol_3334_wenig_TRUE_rf <- as.data.frame(vol_3334_wenig_TRUE_rf)
dumdumrf4 <- vol_3334_wenig_TRUE_rf %>% filter(vol_3334_wenig_TRUE_rf == 'TRUE')