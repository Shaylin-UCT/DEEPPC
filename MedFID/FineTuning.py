# ------------------------- Set up training -------------------------                  
steps_per_epoch = 50
epochs = 50
validation_steps = 10
# ------------------------- Set up training -------------------------

from keras.applications.inception_v3 import InceptionV3,preprocess_input
from keras.preprocessing import image
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D, Dropout

#Datasets
from keras.preprocessing.image import ImageDataGenerator
train_datagen = ImageDataGenerator(rescale = 1./255.,
                                   width_shift_range = 0.2,
                                   height_shift_range = 0.2,
                                   horizontal_flip = True,
                                   preprocessing_function=preprocess_input
                                   )

image_size = 299

test_datagen = ImageDataGenerator(rescale = 1.0/255.) 

import os
base_dir = "/mnt/lustre/users/schetty1/ImagesforResearch/MuraExtra2"
train_dir = os.path.join(base_dir, "train")
test_dir = os.path.join(base_dir, "valid")
train_generator = train_datagen.flow_from_directory(train_dir,
                                                    batch_size = 20, 
                                                    class_mode = "categorical",
                                                    target_size = (image_size, image_size))

test_generator = test_datagen.flow_from_directory(test_dir,
                                                    batch_size = 20, 
                                                    class_mode = "categorical",
                                                    target_size = (image_size, image_size))



# create the base pre-trained model
import keras
base_model = keras.models.load_model("/home/schetty1/MedFIDAttempt2/baseInception.h5")

for layer in base_model.layers:
    layer.trainable = False

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(2048, activation='relu')(x)
x = Dense(2048, activation='relu')(x)
x = Dense(2048, activation='relu')(x)
x = Dense(2048, activation='relu')(x)
x = Dropout(0.2)(x)
predictions = Dense(7, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=predictions)
def myprint2(s):
    with open('/home/schetty1/MedFIDAttempt2/sequential_updatedFineTuning.txt','a') as f:
        print(s, file=f)

model.summary(print_fn=myprint2)


weights_file_path = "/home/schetty1/MedFIDAttempt2/FineTunedWeights.hdf5" #Needs a hdf5 format
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, EarlyStopping, ReduceLROnPlateau, TensorBoard
checkpoint = ModelCheckpoint(weights_file_path, monitor="acc", verbose=1, save_best_only=True, mode = "max")
early = EarlyStopping(monitor="acc", mode="max", patience=15) #See what these do
callbacks_list = [checkpoint, early]
from keras.optimizers import RMSprop
model.compile(optimizer = RMSprop(learning_rate=0.0001), 
              loss = "categorical_crossentropy",  
              metrics = ["acc"])

# train the model on the new data for a few epochs
history = model.fit_generator(
    train_generator, 
    validation_data = test_generator,
    steps_per_epoch = steps_per_epoch, 
    epochs = epochs, 
    validation_steps = validation_steps,  
    verbose = 2, 
    callbacks=callbacks_list
)

# we chose to train the top 2 inception blocks, i.e. we will freeze
# the first 249 layers and unfreeze the rest:
for layer in model.layers[:249]:
   layer.trainable = False
for layer in model.layers[249:]:
   layer.trainable = True

# we need to recompile the model for these modifications to take effect
# we use SGD with a low learning rate
from keras.optimizers import SGD
model.compile(optimizer=SGD(lr=0.0001, momentum=0.9), loss='categorical_crossentropy',metrics = ["acc"])

# we train our model again (this time fine-tuning the top 2 inception blocks
# alongside the top Dense layers
history = model.fit_generator(
    train_generator, 
    validation_data = test_generator,
    steps_per_epoch = 50, #was 50 
    epochs = 20, #was 20
    validation_steps = 10, #was 10 
    verbose = 2, #Could it be True
    callbacks=callbacks_list
)

results = model.evaluate(test_generator)
model.save("/home/schetty1/MedFIDAttempt2/FineTunedModel.h5")