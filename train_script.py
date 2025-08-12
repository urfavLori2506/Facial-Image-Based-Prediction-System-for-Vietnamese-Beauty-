import os
import numpy as np

import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras import Sequential, Model, Input
from tensorflow.keras.layers import Dense
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import load_model
import tensorflow.keras.backend as K

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error

from alt_model_checkpoint.tensorflow import AltModelCheckpoint

import warnings
warnings.filterwarnings('ignore')

import helper

# manually reload single modules if needed
import importlib
importlib.reload(helper)

if tf.version.VERSION < '2.0':
    # TF1.X and older
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.log_device_placement = True
    sess = tf.Session(config=config)
    K.set_session(sess)
else:
    # TF2.X and newer
    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = True
    config.log_device_placement = True
    sess = tf.compat.v1.Session(config=config)
    tf.compat.v1.keras.backend.set_session(sess)

helper.download_data()

target_size = (350,350)

data_generator, num_samples = helper.create_dataset(target_size)
batch_size = 32

# Calculate steps per epoch
steps_per_epoch = num_samples // batch_size

# Split the data into train/val/test
train_steps = int(steps_per_epoch * 0.7)
val_steps = int(steps_per_epoch * 0.15)
test_steps = steps_per_epoch - train_steps - val_steps

print(f'Total samples: {num_samples}')
print(f'Steps per epoch: {steps_per_epoch}')
print(f'Train steps: {train_steps}, Val steps: {val_steps}, Test steps: {test_steps}')

model_name = 'attractiveNet_mnv2'
model_dir = 'models'

model_path= model_dir + '/' + model_name + '.h5'
if not os.path.isdir(model_dir): os.mkdir(model_dir)

basemodel = MobileNetV2(include_top=False, pooling='avg', weights='imagenet')

model = Sequential(name=model_name)
model.add(basemodel)
model.add(Dense(1))

epochs = 30
lr=0.001

model.layers[0].trainable = False
model.compile(loss='mean_squared_error', optimizer=Adam(learning_rate=lr))
print(model.summary())

callbacks = [
    EarlyStopping(
        monitor='val_loss',
        min_delta=1e-3,
        patience=7,
        verbose=1,
        ),
    AltModelCheckpoint(
        model_path,
        model,
        monitor='val_loss', 
        verbose=1, 
        save_best_only=True, 
        save_weights_only=False, 
        )
]

# Create train and validation generators
train_gen = data_generator(batch_size)
val_gen = data_generator(batch_size)

history1 = model.fit(
    train_gen,
    epochs=epochs,
    steps_per_epoch=train_steps,
    validation_data=val_gen,
    validation_steps=val_steps,
    verbose=1,
    callbacks=callbacks,
)

helper.plot_metrics(history1, model_name, 1)

epochs = 30
lr=0.0001

model = load_model(model_path)
model.trainable = True
model.compile(loss='mean_squared_error', optimizer=Adam(learning_rate=lr))
print(model.summary())

callbacks = [
    EarlyStopping(
        monitor='val_loss',
        min_delta=1e-3,
        patience=7,
        verbose=1,
        ),
    AltModelCheckpoint(
        model_path,
        model,
        monitor='val_loss', 
        verbose=1, 
        save_best_only=True, 
        save_weights_only=False, 
        )
]

history2 = model.fit(
    train_gen,
    epochs=epochs,
    steps_per_epoch=train_steps,
    validation_data=val_gen,
    validation_steps=val_steps,
    verbose=1,
    callbacks=callbacks,
)

helper.plot_metrics(history2, model_name, 2)

model = load_model(model_path)

# Create test generator
test_gen = data_generator(batch_size)

# Evaluate on test set
test_loss = model.evaluate(test_gen, steps=test_steps)
print(f'Test loss: {test_loss}')

