library(keras)
library(dplyr)

imdb <- keras::dataset_imdb(num_words = 10000)

train_x <- imdb$train$x
train_y <- imdb$train$y

test_x <- imdb$test$x
test_y <- imdb$test$y

train_x[[1]]

vectorize_sequence <- function(sequence, dimension = 10000){
  m <- matrix(0, nrow = length(sequence), ncol = dimension)
  
  for(i in 1:length(sequence)){
    m[i, sequence[[i]]] <- 1
  }
  m
}

train_x <- vectorize_sequence(train_x)
test_x <- vectorize_sequence(test_x)

train_y <- as.numeric(train_y)
test_y <- as.numeric(test_y)

model <- keras_model_sequential() %>% 
  layer_dense(units = 16, activation = "relu", input_shape = c(10000)) %>% 
  layer_dense(units = 16, activation = "relu") %>% 
  layer_dense(units = 1, activation = "sigmoid")

model %>% compile(
  optimizer = "rmsprop",
  loss = "binary_crossentropy",
  metrics = c("accuracy")
)

val_indices <- 1:1000

validation_x <- train_x[val_indices,]
train_x <- train_x[-val_indices,]

validation_y <- train_y[val_indices]
train_y <- train_y[-val_indices]

history <- model %>% fit(train_x, train_y, epochs = 4, batchsize = 512, validation_data = list(validation_x, validation_y))

