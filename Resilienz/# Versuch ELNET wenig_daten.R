# Versuch ELNET wenig_daten

package_packages <- c('tidyverse', 'mlr3', 'glmnet', 'caret', 'data.table', 'ggplot2')
#install.packages(package_packages)
library(tidyverse)
library(glmnet)
library(mlr3)
library(caret)
library(data.table)
library(ggplot2)

wenig_daten <- fread('C:/Users/Johanner/R projects/Resilienz/wenig_daten.csv')
#wenig_daten <- select(wenig_daten1, 'age', 'BRS_tot', 'LSNS_tot', 'GDS_tot', 'BMI', 'chandler_score')

#head(wenig_daten)
#str(wenig_daten)
x <- sample(1:400,1)
set.seed(x)

# aufsetzen von trainings und testsplit 70/30
index_wenig_en <- sample(
  2, 
  nrow(wenig_daten), 
  replace = TRUE, 
  prob = c(0.7, 0.3)
)
wenig_train_en <- wenig_daten[index_wenig_en == 1, ]
wenig_test_en <- wenig_daten[index_wenig_en == 2, ]

# control parameter
wenig_control_en <- trainControl(
  method = 'repeatedcv', 
  number = 10, repeats = 5, 
  verboseIter = FALSE
)

# model fitting
wenig_elnet <- train(
  chandler_score~., 
  wenig_train_en,  
  method = 'glmnet', 
  tuneGrid = expand.grid(alpha = seq(0,1,length = 10), lambda = seq(0.0001, 0.2, length = 5)), 
  trControl = wenig_control_en
)

# predictions fuer wenig_test
wenig_predictions_en <- predict(
  wenig_elnet, 
  newdata = wenig_test_en
)

# mean squared error zwischen predicted chandler_score und realem chandler_score
wenig_mse_en <- mean((wenig_test_en$chandler_score - wenig_predictions_en)^2)
wenig_mse_en

# variablen gewichtung plotten
plot1 <- plot(varImp(wenig_elnet,scale=TRUE), 
  main = as.character(x), 
  sub = as.character(wenig_mse_en)
)
print(plot1)