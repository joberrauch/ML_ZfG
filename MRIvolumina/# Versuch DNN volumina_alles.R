# Versuch DNN volumina_alles

package_packages <- c('tidyverse', 'mlr3', 'glmnet', 'caret', 'data.table', 'ggplot2', 'randomForest', 'keras', 'tfruns', 'tfestimators','tensorflow', 'devtools')
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
library(tensorflow)
library(devtools)
install_keras(method = "conda", tensorflow = "cpu")


vol_alles <- readxl::read_xlsx('C:/Users/Johanner/R projects/MRIvolumina/ZfG_APOE4_Volumina.xlsx')

# gebdatum zu alter
geb_als_datum <- as.Date(vol_alles$Gebdatum)
heute <- Sys.Date()
age <- as.numeric(difftime(heute, geb_als_datum, units = "weeks")) %/% 52.25
vol_alles$Gebdatum <- age
vol_alles$Gebdatum <- vol_alles$Gebdatum %>% as.integer(vol_alles$Gebdatum)
names(vol_alles)[names(vol_alles)=="Gebdatum"] <- "Alter"

# APOE von str zu int
vol_alles$APOE <- as.factor(vol_alles$APOE)

head(vol_alles)
str(vol_alles)
set.seed(2)

# definieren von predictor und features
vol_alles_dnn_x <- vol_alles %>% select(!APOE)
vol_alles_dnn_y <- vol_alles %>% select(APOE)

# model
model <- keras_model_sequential() %>%
  
  # Network architecture
  layer_dense(units = 128, activation = "relu", input_shape = ncol(vol_alles_dnn_x), layer_dropout(rate = 0.2)) %>%
  layer_batch_normalization() %>%
  layer_dense(units = 64, activation = "relu", layer_dropout(rate = 0.2)) %>%
  layer_batch_normalization() %>%
  layer_dense(units = 10, activation = "softmax")  %>%
  
  # Backpropagation
  compile(
    loss = 'categorical_crossentropy',
    optimizer = optimizer_rmsprop(),
    metrics = c('accuracy')
  )

# train model with data from vol_alles_train_en
fit1 <- model %>%
  fit(
    x = vol_alles_dnn_x,
    y = vol_alles_dnn_y,
    epochs = 25,
    batch_size = 128,
    validation_split = 0.2,
    verbose = TRUE
  )

# Display output
plot(fit1)