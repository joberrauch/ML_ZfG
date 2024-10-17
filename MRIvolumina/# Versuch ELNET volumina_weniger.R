# Versuch ELNET volumina_weniger

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

# aus vol_alles vol_weniger machen
vol_weniger <- vol_alles %>% select(1:39)
vol_weniger <- vol_weniger %>% select(!Gebdatum)

# gebdatum zu alter
#geb_als_datum2 <- as.Date(vol_weniger$Gebdatum)
#heute2 <- Sys.Date()
#age2 <- as.numeric(difftime(heute2, geb_als_datum2, units = "weeks")) %/% 52.25
#vol_weniger$Gebdatum <- age2
#vol_weniger$Gebdatum <- vol_weniger$Gebdatum %>% as.integer(vol_weniger$Gebdatum)
#names(vol_weniger)[names(vol_weniger)=="Gebdatum"] <- "Alter"

# APOE von chr zu fct
vol_weniger$APOE <- as.factor(vol_weniger$APOE)

head(vol_weniger)
str(vol_weniger)
set.seed(6)

# aufsetzen von trainings und testsplit 70/30
index_vol_weniger_en <- sample(2, nrow(vol_weniger), replace = TRUE, prob = c(0.665, 0.335))
vol_weniger_train_en <- vol_weniger[index_vol_weniger_en == 1, ]
vol_weniger_test_en <- vol_weniger[index_vol_weniger_en == 2, ]

# control parameter
vol_weniger_control_en <- trainControl(method = 'repeatedcv', number = 5, repeats = 5, verboseIter = TRUE)

# model fitting
vol_weniger_elnet <- train(APOE~., vol_weniger_train_en,  method = 'glmnet', tuneGrid = expand.grid(alpha = seq(0,1,length = 10), lambda = seq(0.0001, 0.2, length = 5)), trControl = vol_weniger_control_en)

# variablen gewichtung plotten
plot(varImp(vol_weniger_elnet,scale=TRUE))

# predictions fuer wenig_test
vol_weniger_predictions_en <- predict(vol_weniger_elnet, newdata = vol_weniger_test_en)

# nicht mean squared error zwischen predicted chandler_score und realem chandler_score
vol_weniger_eval_en <- (vol_weniger_test_en$APOE == vol_weniger_predictions_en)
vol_weniger_TRUE <- as.character(vol_weniger_eval_en)
vol_weniger_TRUE <- as.data.frame(vol_weniger_TRUE)
dumdum2 <- vol_weniger_TRUE %>% filter(vol_weniger_TRUE == 'TRUE')