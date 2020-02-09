library(keras)

housing <- keras::dataset_boston_housing()

train_x <- housing$train$x
train_y <- housing$train$y

test_x <- housing$test$x
test_y <- housing$test$y

mean <- apply(train_x, 2, mean)
mean

std <- apply(train_x, 2, sd)
std

train_x <- scale(train_x, center = mean, scale = std)
test_x <- scale(test_x, center = mean, scale = std)

dim(train_x)[2]

build_model <- function(){
  model <- keras_model_sequential() %>% 
    layer_dense(units = 64, activation = "relu", input_shape = dim(train_x)[2]) %>% 
    layer_dense(units = 64, activation = "relu") %>% 
    layer_dense(units = 1)
  
  model %>% compile(
    optimizer = "rmsprop",
    loss = "mse",
    metrics = c("mae")
  )
}

k <- 4

indices <- sample(1:nrow(train_x))
length(indices)

folds <- cut(1:length(indices), breaks = k, labels = FALSE)
folds

num_epochs <- 100
all_scores <- c()

for(i in 1:k){
  
  print(paste0("processing fold #", i))
  
  val_indices <- which(folds == k, arr.ind = TRUE)
  val_data <- train_x[val_indices, ]
  val_targets <- train_y[val_indices]
  
  partial_train_x <- train_x[-val_indices, ]
  partial_train_y <- train_y[-val_indices]
  
  model <- build_model()
  
  model %>% fit(partial_train_x, partial_train_y, epochs = num_epochs, batch_size = 1, verbose = 0)
  results <- model %>% evaluate(val_data, val_targets, verbose = 0)
  all_scores <- c(all_scores, results$mean_absolute_error)
}
