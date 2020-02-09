library(keras)
library(dplyr)

reuters <- keras::dataset_reuters(num_words = 10000)
word_index <- keras::dataset_reuters_word_index()
t_word_index <- names(word_index)
names(t_word_index) <- word_index
head(word_index)
head(t_word_index)

head(t_word_index)

train_x <- reuters$train$x
train_y <- reuters$train$y

test_x <- reuters$test$x
test_y <- reuters$test$y

decode <- function(article){
  translated <- vector(mode = 'character')
  for(i in 1:length(article)){
    translated <- paste0(translated, " ", t_word_index[[article[i]]])
  }
  return(translated)
}

decode(train_x[[1]])

vectorize_text <- function(text, dimensions = 10000){
  m <- matrix(0, nrow = length(text), ncol = dimensions)
  for(i in 1:nrow(m)){
    m[i, text[[i]]] <- 1
  }
  return(m)
}

train_x <- vectorize_text(train_x)        
test_x <- vectorize_text(test_x)

# to_hot_encode <- function(labels, dimensions = 46){
#   m <- matrix(0, nrow = length(labels), ncol = dimensions)
#   for(i in 1:length(labels)){
#     m[i, labels[i] + 1] <- 1
#   }
#   return(m)
# }
# 
# train_y <- to_hot_encode(train_y)
# test_y <- to_hot_encode(test_y)

train_y <- to_categorical(train_y)
test_y <- to_categorical(test_y)

model <- keras_model_sequential() %>% 
  layer_dense(units = 64, activation = "relu", input_shape = c(10000)) %>% 
  layer_dense(units = 64, activation = "relu") %>% 
  layer_dense(units = 46, activation = "softmax")

model %>% compile(
  optimizer = "rmsprop",
  loss = "categorical_crossentropy",
  metrics = c("accuracy")
)

val_indices <- 1:1000
x_val <- train_x[val_indices, ]
y_val <- train_y[val_indices, ]

train_x <- train_x[-val_indices, ]
train_y <- train_y[-val_indices, ]

history <- model %>% fit(
  train_x,
  train_y,
  epochs = 20,
  batch_size = 512,
  validation_data = list(x_val, y_val)
)

