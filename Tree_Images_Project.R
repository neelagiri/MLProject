library(keras)


model <- keras_model_sequential() %>% 
  layer_conv_2d(filters = 32, kernel_size = c(3,3), activation = "relu", 
                input_shape = c(32,32,3)) %>% 
  layer_max_pooling_2d(pool_size = c(2,2)) %>% 
  layer_conv_2d(filters = 64, kernel_size = c(3,3), activation = "relu") %>% 
  layer_max_pooling_2d(pool_size = c(2,2)) %>% 
  layer_conv_2d(filters = 64, kernel_size = c(3,3), activation = "relu")



# adding a dense layer on top of the model
model %>% 
  layer_flatten() %>% 
  layer_dense(units = 512, activation = "relu") %>% 
  layer_dense(units = 1, activation = "sigmoid")

summary(model)


# compiling and training the model

model %>% compile(
  loss = "binary_crossentropy",
  optimizer = optimizer_rmsprop(lr = 1e-4),
  metrics = c("acc")
)

# All images will be rescaled by 1/255
train_datagen <- image_data_generator(rescale = 1/255)
epochs = 5
nb_train_samples = 74
# training and testing tree dataframe of images that were converted from python

dataframe_tree_train <- "/Users/chayan/Desktop/Project/Final Data sheets/Trees/Images/exportImage__Train_DF.csv"
dataframe_tree_train_read <- read.csv(dataframe_tree_train)

# creating a training generator

ImageDataGenerator.flow(
  x,
  y=None,
  batch_size=32,
  shuffle=True,
  sample_weight=None,
  seed=None,
  save_to_dir=None,
  save_prefix="",
  save_format="png",
  subset=None,
)

#creating an image data generator
train_datagen <- image_data_generator(
  rescale = 1/255,
  rotation_range = 40,
  width_shift_range = 0.2,
  height_shift_range = 0.2,
  shear_range = 0.2,
  zoom_range = 0.2,
  horizontal_flip = TRUE
)


train_generator <- flow_images_from_dataframe(
  # This is the datafame file
  dataframe_tree_train_read,
  #adding the above train generator
  generator = train_datagen,
  # This is the data generator
  directory=None,
  x_col="label",
  y_col="class",
  color_mode="rgb",
  classes=None,
  # All images will be resized to 30x30
  target_size = c(30, 30),
  batch_size = 316,
  # Since we use binary_crossentropy loss, we need binary labels
  class_mode = "binary"
)

#fitting our training generator on our model
history <- model %>% fit_generator(
  train_generator,
  steps_per_epoch = nb_train_samples/epochs,
  epochs = 5,
)

# our model is ready based on the training data set above , now we will create a test generator to test our final results

dataframe_tree_test <- "/Users/chayan/Desktop/Project/Final Data sheets/Trees/Images/exportImage_Tree_Test_DF.csv"
dataframe_tree_test_read <- read.csv(datafram_tree_test)

test_datagen <- image_data_generator(rescale = 1/255)



test_generator <- flow_images_from_dataframe(
  # This is the datafame file
  datafram_tree_test_read,
  #adding the above train generator
  generator = test_datagen,
  # This is the data generator
  directory=None,
  x_col="label",
  y_col="class",
  color_mode="rgb",
  classes=None,
  # All images will be resized to 30x30
  target_size = c(30, 30),
  batch_size = 316,
  # Since we use binary_crossentropy loss, we need binary labels
  class_mode = "binary"
)


# running our final prediction
predicted_data = model %>% predict(test_generator)
write.csv(predicted_data, "labels")




