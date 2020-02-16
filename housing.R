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

#num_epochs <- 100
# Average MAE is around 2.4 ($2400) which is fairly large
num_epochs <- 500

#all_scores <- c()
all_mae_history <- NULL

for(i in 1:k){
  
  print(paste0("processing fold #", i))
  
  val_indices <- which(folds == k, arr.ind = TRUE)
  val_data <- train_x[val_indices, ]
  val_targets <- train_y[val_indices]
  
  partial_train_x <- train_x[-val_indices, ]
  partial_train_y <- train_y[-val_indices]
  
  model <- build_model()
  
  #history <- model %>% fit(partial_train_x, partial_train_y, epochs = num_epochs, batch_size = 1, verbose = 0)
  history <- model %>% fit(partial_train_x, partial_train_y, epochs = num_epochs, batch_size = 1, validation_data = list(val_data, val_targets), verbose = 0)
  
  # history <- model %>% evaluate(val_data, val_targets, verbose = 0)
  # all_scores <- c(all_scores, results$mean_absolute_error)
  
  mae_history <- history$metrics$val_mean_absolute_error
  all_mae_history <- rbind(all_mae_history, mae_history)
}

average_mae_history <- data.frame(
  epoch = seq(1:ncol(all_mae_history)),
  validation_mae = apply(all_mae_history, 2, mean)
)

library(ggplot2)
ggplot(average_mae_history) + geom_smooth(aes(epoch, validation_mae))

## Final Model
model <- build_model()
model %>% fit(train_x, train_y, epochs = 130, batch_size = 16, verbose = 0)
result <- model %>% evaluate(test_x, test_y)
