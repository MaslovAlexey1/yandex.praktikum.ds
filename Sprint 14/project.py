import pandas as pd
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
from pathlib import Path
path = Path.cwd()
from tensorflow.keras.applications.resnet import ResNet50
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Flatten
from tensorflow.keras.optimizers import Adam
import PIL as PIL
from PIL import Image

labels = pd.read_csv('{}/datasets/faces/labels.csv'.format(path))

def load_train(path):
    datagen = ImageDataGenerator(rescale=1/255., validation_split=0.25)
    train_gen_flow = datagen.flow_from_dataframe(
        dataframe=labels,
        directory='{}/datasets/faces/final_files'.format(path),
        x_col='file_name',
        y_col='real_age',
        target_size=(224, 224),
        batch_size=32,
        class_mode='raw',
        subset='training',
        seed=12345
    )
    return train_gen_flow

def load_test(path):
    datagen = ImageDataGenerator(rescale=1/255., validation_split=0.25)
    test_gen_flow = datagen.flow_from_dataframe(
        dataframe=labels,
        directory='{}/datasets/faces/final_files'.format(path),
        x_col='file_name',
        y_col='real_age',
        target_size=(224, 224),
        batch_size=32,
        class_mode='raw',
        subset='validation',
        seed=12345
    )
    return test_gen_flow

def create_model(input_shape):
    backbone = ResNet50(input_shape=(150, 150, 3),
                    weights=None, 
                    include_top=False)
    model = Sequential()
    model.add(backbone)
    model.add(GlobalAveragePooling2D())
    model.add(Flatten())
    model.add(Dense(1, activation='relu'))

    optimizer = Adam(lr=0.1)
    model.compile(optimizer=optimizer, loss='mse', 
                  metrics=['acc'])
    return model
    
def train_model(model, train_data, test_data, batch_size=None, epochs=1, steps_per_epoch=None, validation_steps=None):
    if steps_per_epoch is None:
        steps_per_epoch = len(train_data)
    if validation_steps is None:
        validation_steps = len(test_data)
    model.fit(train_data, 
              validation_data=test_data,
              batch_size=batch_size, epochs=epochs,
              steps_per_epoch=steps_per_epoch,
              validation_steps=validation_steps,
              verbose=2)
    return model

train_data = load_train('/Users/alexeymaslov/Git/github/praktikum/Sprint 14')
test_data = load_test('/Users/alexeymaslov/Git/github/praktikum/Sprint 14')

model = create_model((150, 150, 3))

train_model(model, train_data, test_data, batch_size=None, epochs=1,
                steps_per_epoch=None, validation_steps=None)