# # Galaxy Classification using Supervised Learning with Deep Convolutional Neural Networks for Multi-Class Image Classification
# ## Group 7 AI Class Final Projects
# Members:
# - Abi
# - Gavin
# - Rasyid
# - Hikmal

# ---

# ### Import Libraries

#Change/add any additional imports here
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow import keras
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.regularizers import l2
from tensorflow.keras.layers import BatchNormalization

# ### Data Wrangling & Preprocessing

#Process the labels first
def major_class(t):
    t = str(t).strip()
    if t.startswith("E"):
        return "E"
    elif t.startswith("S0"):
        return "E" #S0 is closer to E (Lenticular)
    elif t.startswith("SAB"):
        return "SB" #SAB is closer to SB (Barred Spiral)
    elif t.startswith("SB"):
        return "SB"
    elif t.startswith("S"):
        return "S"
    else:
        return "Other"  #Catchall for irregular/unknown types

df = pd.read_csv('efigiuse/label.csv')
print(df.head())
print("\nunique classes:")
print(df["type"].unique()) #Checking the unique classes
df["major_class"] = df["type"].apply(major_class)
print(df["major_class"].value_counts()) #Checking the number of major classes

#Create new dataframe with only PGC_name and major_class columns
df_major = df[["PGC_name", "major_class"]]
print(df_major.head(15)) #df_major at a glance

#Create a filtered version of df_major ("Other" class removed)
df_use = df_major[df_major["major_class"] != "Other"].reset_index(drop=True)
print(df_use.head(15)) #df_use at a glance

#Label processing complete

#Image processing
image_dir = "efigiuse/png/"
image_size = (255, 255)
X, y = [], [] #Initialize empty lists for images and labels
for idx, row in df_use.iterrows():
    image_path = os.path.join(image_dir, f"{row['PGC_name']}.png")
    if os.path.exists(image_path):
        img = load_img(image_path, target_size=image_size)
        img_array = img_to_array(img) / 255.0  #Normalization to [0, 1]
        X.append(img_array) #X is for the images
        y.append(row["major_class"]) #y is for the labels
    else:
        print(f"Missing image: {image_path}")
#No images should be missing. If any are, check efigiuse/png/ and redownload the images from: 
#https://www.astromatic.net/download/efigi/efigi_png_gri-1.6.tgz
encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y) #Label encoding
print("Classes: " + str(encoder.classes_)) #Checking the classes, should be ["E", "S", "SB"]

#Train, validation, and test split
X = np.array(X) #Convert X & y to numpy arrays
y_encoded = np.array(y_encoded)
#Train (70%) - Rest (30%) split
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y_encoded, test_size=0.30, random_state=42, stratify=y_encoded)
#Validation (15%) - Test (15%) split
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.50, random_state=42, stratify=y_temp)

#Training data augmentation to prevent overfitting
datagen = ImageDataGenerator(
    rotation_range=10,
    zoom_range=0.1,
    width_shift_range=0.05,
    height_shift_range=0.05,
    horizontal_flip=True,
    fill_mode='nearest'
)
datagen.fit(X_train)

num_classes = len(df_use["major_class"].unique()); print("Number of classes: " + str(num_classes)) #Checking the number of classes
#Current arrays are integer coded (0,1,2). Convert to one-hot encoding
y_train = to_categorical(y_train, num_classes=num_classes)
y_val = to_categorical(y_val, num_classes=num_classes)
y_test = to_categorical(y_test, num_classes=num_classes)

weight_decay = 1e-4 #L2 regularization parameter
#Class weights to handle class imbalance (especially for SB class (2))
class_weights = {0: 1.2486979166666667, 1: 0.7365591397849462, 2: 1.1883519206939281}
#Source:
# class_weights_array = compute_class_weight(class_weight='balanced', classes=np.unique(y_train_int), y=y_train_int)
# class_weights = dict(enumerate(class_weights_array))
# print("Class weights: ", class_weights)

model = Sequential([
    Input(shape=(255, 255, 3)),
    Conv2D(32, (3, 3), activation='relu', kernel_regularizer=l2(weight_decay)),
    Conv2D(32, (3, 3), activation='relu', kernel_regularizer=l2(weight_decay)),
    BatchNormalization(),
    MaxPooling2D(pool_size=(2, 2)),

    Conv2D(64, (3, 3), activation='relu', kernel_regularizer=l2(weight_decay)),
    Conv2D(64, (3, 3), activation='relu', kernel_regularizer=l2(weight_decay)),
    BatchNormalization(),
    MaxPooling2D(pool_size=(2, 2)),

    Conv2D(128, (3, 3), activation='relu', kernel_regularizer=l2(weight_decay)),
    Conv2D(128, (3, 3), activation='relu', kernel_regularizer=l2(weight_decay)),
    Conv2D(128, (3, 3), activation='relu', kernel_regularizer=l2(weight_decay)),
    BatchNormalization(),
    MaxPooling2D(pool_size=(2, 2)),

    Flatten(),
    Dense(128, activation='relu', kernel_regularizer=l2(weight_decay)),
    BatchNormalization(),
    Dropout(0.5),  #Prevents overfitting
    Dense(num_classes, activation='softmax')
])

model.compile(optimizer=Adam(0.001), loss='categorical_crossentropy', metrics=['accuracy'])
early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
history = model.fit(datagen.flow(X_train, y_train, batch_size=32),
                    validation_data=(X_val, y_val),
                    epochs=30,
                    callbacks=[early_stop],
                    class_weight=class_weights)

#WARNING: MAKE SURE TO HAVE A GENUINELY GOOD COMPUTING SETUP WITH MAXIMUM COOLING AND POWER BEFORE EXECUTING
#DO NOT LET YOUR MACHINE GET OVERHEATED

#Model evaluation
loss, acc = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {acc:.4f}")

from sklearn.metrics import classification_report, confusion_matrix

y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true = np.argmax(y_test, axis=1)

print(classification_report(y_true, y_pred_classes))
print(confusion_matrix(y_true, y_pred_classes))

#Training history plots to visualize performance (underfitting/overfitting)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.legend()
plt.show()

plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.legend()
plt.show()