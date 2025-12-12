# train_mobilenet.py
import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint

DATA_DIR = "./Original Data"
TRAIN_DIR = os.path.join(DATA_DIR, "train")
VAL_DIR   = os.path.join(DATA_DIR, "test")
IMG_SIZE = (224, 224)
BATCH_SIZE = 16
EPOCHS = 3   # quick run for testing — raise later
MODEL_OUT = "./models/model.h5"

os.makedirs("models", exist_ok=True)

train_datagen = ImageDataGenerator(rescale=1./255,
                                   rotation_range=20,
                                   width_shift_range=0.1,
                                   height_shift_range=0.1,
                                   horizontal_flip=True)

val_datagen = ImageDataGenerator(rescale=1./255)

train_gen = train_datagen.flow_from_directory(
    TRAIN_DIR, target_size=IMG_SIZE, batch_size=BATCH_SIZE, class_mode="categorical"
)

val_gen = val_datagen.flow_from_directory(
    VAL_DIR, target_size=IMG_SIZE, batch_size=BATCH_SIZE, class_mode="categorical"
)

base = MobileNetV2(weights="imagenet", include_top=False, input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3))
x = base.output
x = GlobalAveragePooling2D()(x)
x = Dense(256, activation="relu")(x)
preds = Dense(train_gen.num_classes, activation="softmax")(x)

model = Model(inputs=base.input, outputs=preds)
for layer in base.layers:
    layer.trainable = False

model.compile(optimizer=Adam(1e-3), loss="categorical_crossentropy", metrics=["accuracy"])

checkpoint = ModelCheckpoint(MODEL_OUT, monitor="val_accuracy", save_best_only=True, verbose=1)

model.fit(train_gen, validation_data=val_gen, epochs=EPOCHS, callbacks=[checkpoint])

print("SAVED MODEL:", MODEL_OUT)
