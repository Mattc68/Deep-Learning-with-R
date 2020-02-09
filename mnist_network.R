library(keras)
library(dplyr)

setwd("c:/Users/clear/Workspace/Deep Learning/")

mnist <- dataset_mnist()

train_images <- mnist$train$x
train_labels <- mnist$train$y

test_images <- mnist$test$x
test_labels <- mnist$test$y

plot(as.raster(train_images[1,,], max = 255))

train_images <- train_images/255
test_images <- test_images/255

train_labels <- to_categorical(train_labels)
test_labels <- to_categorical(test_labels)

train_images <- train_images %>% array_reshape(dim = c(nrow(train_images), 28*28))
test_images <- test_images %>% array_reshape(dim = c(nrow(test_images), 28*28))

network <- keras_model_sequential() %>% 
  layer_dense(units = 512, activation = "relu", input_shape = c(28*28)) %>% 
  layer_dense(units = 10, activation = "softmax")

network %>% compile(
  optimizer = "rmsprop",
  loss = "categorical_crossentropy",
  metrics = c("accuracy")
)

network %>% fit(train_images, train_labels, epochs = 5, batch_size = 128)

metrics <- network %>% evaluate(test_images, test_labels)
metrics$loss
metrics$acc

network %>% predict_classes(test_images[1:10, ])
