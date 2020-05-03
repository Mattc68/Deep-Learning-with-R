setwd("C:/Users/clear/Workspace/Deep Learning/dogs-vs-cats/workspace")

base <- getwd()
dir_train <- file.path(base, "train")
dir_val <- file.path(base, "val")
dir_test <- file.path(base, "test")

library(keras)
library(tidyverse)

conv_base <- application_vgg16(
  weights = "imagenet",
  include_top = FALSE,
  input_shape = c(150, 150, 3)
)

datagen <- image_data_generator(rescale = 1/255)
batch_size <- 20

extract_features <- function(directory, sample_count){
  features <- array(0, c(sample_count, 4, 4, 512))
  labels <- array(0, c(sample_count))
  
  generator <- flow_images_from_directory(
    directory = directory,
    generator = datagen,
    class_mode = "binary",
    batch_size = batch_size,
    target_size = c(150, 150)
  )
  
  i <- 0
  
  while(TRUE){
    batch <- generator_next(generator)
    input_batch <- batch[[1]]
    label_batch <- batch[[2]]
    feature_batch <- conv_base %>% predict(input_batch)
    
    index_range <- ((i * batch_size)+ 1):((i + 1) * batch_size)
    features[index_range,,,] <- feature_batch
    labels[index_range] <- label_batch
    
    i <- i + 1
    if(i * batch_size >= sample_count)
      break
    
  }
  
  return(list(features = features, labels = labels))
}

train <- extract_features(dir_train, 2000)
val <- extract_features(dir_val, 1000)
test <- extract_features(dir_test, 1000)

reshape_features <- function(features){
  array_reshape(features, c(nrow(features), 4*4*512))
}

train$features <- reshape_features(train$features)
val$features <- reshape_features(val$features)
test$features <- reshape_features(test$features)

model <- keras_model_sequential() %>% 
  layer_dense(units = 256, activation = "relu", input_shape = 4*4*512) %>% 
  layer_dropout(rate = 0.5) %>% 
  layer_dense(units = 1, activation = "sigmoid")

model %>% compile(
  optimizer = optimizer_rmsprop(lr = 2e-5),
  loss = "binary_crossentropy",
  metrics = c("accuracy")
)

history <- model %>% fit(train$features, train$labels, epochs = 30, batch_size = 20, 
                         validation_data = list(val$features, val$labels))


