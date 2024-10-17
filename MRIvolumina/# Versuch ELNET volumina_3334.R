# Versuch ELNET volumina_3334

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

# APOE von chr zu fct
vol_3334$APOE <- as.factor(vol_3334$APOE)

head(vol_3334)
str(vol_3334)
set.seed(2)

# aufsetzen von trainings und testsplit 70/30
index_vol_3334_en <- sample(2, nrow(vol_3334), replace = TRUE, prob = c(0.665, 0.335))
vol_3334_train_en <- vol_3334[index_vol_3334_en == 1, ]
vol_3334_test_en <- vol_3334[index_vol_3334_en == 2, ]

# control parameter
vol_3334_control_en <- trainControl(method = 'repeatedcv', number = 5, repeats = 5, verboseIter = TRUE)

# model fitting
vol_3334_elnet <- train(APOE~., vol_3334_train_en,  method = 'glmnet', tuneGrid = expand.grid(alpha = seq(0,1,length = 10), lambda = seq(0.0001, 0.2, length = 5)), trControl = vol_3334_control_en)

# variablen gewichtung plotten
plot(varImp(vol_3334_elnet,scale=TRUE))

# predictions fuer wenig_test
vol_3334_predictions_en <- predict(vol_3334_elnet, newdata = vol_3334_test_en)

# nicht mean squared error zwischen predicted chandler_score und realem chandler_score
vol_3334_eval_en <- (vol_3334_test_en$APOE == vol_3334_predictions_en)
vol_3334_TRUE <- as.character(vol_3334_eval_en)
vol_3334_TRUE <- as.data.frame(vol_3334_TRUE)
dumdum3 <- vol_3334_TRUE %>% filter(vol_3334_TRUE == 'TRUE')