setwd("C:/Users/clear/Workspace/Deep Learning/")

original_dataset_dir <- "C:/Users/clear/Workspace/Deep Learning/dogs-vs-cats/"

base_dir <- "C:/Users/clear/Workspace/Deep Learning/cats_and_dogs_small"
# dir.create(base_dir)

train_dir <- file.path(base_dir, "train")
# dir.create(train_dir)

validation_dir <- file.path(base_dir, "validation")
# dir.create(validation_dir)

test_dir <- file.path(base_dir, "test")
# dir.create(test_dir)

train_cats_dir <- file.path(train_dir, "cats")
train_dogs_dir <- file.path(train_dir, "dogs")
# dir.create(train_cats_dir)
# dir.create(train_dogs_dir)

validation_cats_dir <- file.path(validation_dir, "cats")
validation_dogs_dir <- file.path(validation_dir, "dogs")
# dir.create(validation_cats_dir)
# (validation_dogs_dir)

test_cats_dir <- file.path(test_dir, "cats")
test_dogs_dir <- file.path(test_dir, "dogs")
# dir.create(test_cats_dir)
# dir.create(test_dogs_dir)


# fnames <- paste0("cat.", 1:1000, ".jpg")
# fnames
# 
# file.copy(file.path('C:/Users/clear/Workspace/Deep Learning/dogs-vs-cats/train/train/', fnames), file.path(train_cats_dir))
# 
# fnames <- paste0("cat.", 1001:1500, ".jpg")
# file.copy(file.path('C:/Users/clear/Workspace/Deep Learning/dogs-vs-cats/train/train/', fnames), file.path(validation_cats_dir))
# 
# fnames <- paste0("cat.", 1501:2000, ".jpg")
# file.copy(file.path('C:/Users/clear/Workspace/Deep Learning/dogs-vs-cats/train/train/', fnames), file.path(test_cats_dir))
# 
# fnames <- paste0("dog.", 1:1000, ".jpg")
# fnames
# 
# file.copy(file.path('C:/Users/clear/Workspace/Deep Learning/dogs-vs-cats/train/train/', fnames), file.path(train_dogs_dir))
# 
# fnames <- paste0("dog.", 1001:1500, ".jpg")
# file.copy(file.path('C:/Users/clear/Workspace/Deep Learning/dogs-vs-cats/train/train/', fnames), file.path(validation_dogs_dir))
# 
# fnames <- paste0("dog.", 1501:2000, ".jpg")
# file.copy(file.path('C:/Users/clear/Workspace/Deep Learning/dogs-vs-cats/train/train/', fnames), file.path(test_dogs_dir))
# 
# cat("total training cat images: ", length(list.files(train_cats_dir)), "\n")
# cat("total validation cat images: ", length(list.files(validation_cats_dir)), "\n")
# cat("total test cat images: ", length(list.files(test_cats_dir)), "\n")
# 
# cat("total training dog images: ", length(list.files(train_dogs_dir)), "\n")
# cat("total validation dog images: ", length(list.files(validation_dogs_dir)), "\n")
# cat("total test dog images: ", length(list.files(test_dogs_dir)), "\n")

library(keras)

model <- keras_model_sequential() %>% 
  layer_conv_2d(filters = 32, kernel_size = c(3, 3), activation = "relu", input_shape = c(150, 150, 3)) %>% 
  layer_max_pooling_2d(pool_size = c(2, 2)) %>% 
  layer_conv_2d(filters = 64, kernel_size = c(3, 3), activation = "relu") %>% 
  layer_max_pooling_2d(pool_size = c(2, 2)) %>% 
  layer_conv_2d(filters = 128, kernel_size = c(3, 3), activation = "relu") %>% 
  layer_max_pooling_2d(pool_size = c(2, 2)) %>% 
  layer_conv_2d(filters = 128, kernel_size = c(3, 3), activation = "relu") %>% 
  layer_max_pooling_2d(pool_size = c(2, 2)) %>%
  layer_flatten() %>% 
  layer_dropout(rate = 0.5) %>% 
  layer_dense(units = 512, activation = "relu") %>% 
  layer_dense(units = 1, activation = "sigmoid")

model %>% compile(
  loss = "binary_crossentropy",
  optimizer = optimizer_rmsprop(lr = 1e-4),
  metrics = c("acc")
)  

##Data Augmentation
datagen <- image_data_generator(
  rescale = 1/255,
  rotation_range = 40,
  width_shift_range = 0.2,
  height_shift_range = 0.2,
  shear_range = 0.2,
  zoom_range = 0.2,
  horizontal_flip = TRUE,
  fill_mode = "nearest"
)

train_datagen <- image_data_generator(rescale = 1/255)
validation_datagen <- image_data_generator(rescale = 1/255)
test_datagen <- image_data_generator(rescale = 1/255)

train_generator <- flow_images_from_directory(
  train_dir,
  datagen,
  target_size = c(150, 150),
  batch_size = 20,
  class_mode = "binary"
)

validation_generator <- flow_images_from_directory(
  validation_dir,
  test_datagen,
  target_size = c(150, 150),
  batch_size = 20,
  class_mode = "binary"
)

batch <- generator_next(train_generator)
str(batch)

history <- model %>% fit_generator(
  train_generator,
  steps_per_epoch = 100,
  epochs = 30,
  validation_data = validation_generator,
  validation_steps = 50
)

model %>% save_model_hdf5("cats_and_dogs_small_2.h5")

#displaying randomly augmented training images
fnames <- list.files(train_cats_dir, full.names = TRUE)
img_path <- fnames[[3]]

img <- image_load(img_path, target_size = c(150, 150))
img_array <- image_to_array(img)
img_array <- array_reshape(img_array, c(1, 150, 150, 3))

augmentation_generator <- flow_images_from_data(
  img_array,
  generator = datagen,
  batch_size = 1
)

op <- par(mfrow = c(2, 2), pty = "s", mar = c(1, 0, 1, 0))

for(i in 1:4){
  batch <- generator_next(augmentation_generator)
  plot(as.raster(batch[1,,,]))
}

par(op)

# Building a model based on the VGG16

library(keras)
library(tidyverse)

conv_base <- application_vgg16(
  weights = "imagenet",
  include_top = FALSE,
  input_shape = c(150, 150, 3)
)

## Extracting Features using the pretrained Convolutional Base

base_dir <- "C:\\Users\\clear\\Workspace\\Deep Learning\\cats_and_dogs_small"
train_dir <- file.path(base_dir, "train")
validation_dir <- file.path(base_dir, "validation")
test_dir <- file.path(base_dir, "test")

datagen <- image_data_generator(1/255)
batch_size = 20

extract_features <- function(directory, sample_count){
  
  features <- array(0, dim = c(sample_count, 4, 4, 512))
  labels <- array(0, dim = c(sample_count))
  
  generator <- flow_images_from_directory(
    directory = directory,
    generator = datagen,
    target_size = c(150, 150),
    batch_size = batch_size,
    class_mode = "binary"
  )
  
  
  i <- 0
  
  while(TRUE){
    batch <- generator_next(generator)
    inputs_batch <- batch[[1]]
    labels_batch <- batch [[2]]
    features_batch <- conv_base %>% predict(inputs_batch)
    
    index_range <- ((i * batch_size)+1):((i +1) * batch_size)
    features[index_range,,,] <- features_batch
    labels[index_range] <- labels_batch
    
    i <- i + 1
    if(i * batch_size >= sample_count)
      break
  }
  
  list(features = features,
       labels = labels
  )
}

train <- extract_features(train_dir, 2000)
validation <- extract_features(validation_dir, 1000)
test <- extract_features(test_dir, 1000)

reshape_features <- function(features){
  array_reshape(features, dim = c(nrow(features), 4*4*512))
}

train$features <- reshape_features(train$features)
validation$features <- reshape_features(validation$features)
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
                         validation_data = list(validation$features, validation$labels)
) 


## Visualizig intermediate activations
library(keras)
model <- load_model_hdf5("cats_and_dogs_small_2.h5")
model

img_path <- "C:/Users/clear/Workspace/Deep Learning/cats_and_dogs_small/test/cats/cat.1700.jpg"
img <- image_load(img_path, target_size = c(150, 150))
