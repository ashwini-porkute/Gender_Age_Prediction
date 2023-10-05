# Step 1: Import Libraries
import os
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
from keras.models import Model
from keras.layers import *
from keras.preprocessing.image import load_img
# from keras.utils import plot_model
from keras.utils.vis_utils import plot_model
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')
import pandas as pd

Base_img_dir = "./UTKFacedataset/UTKFace"

gender_dict = {0: 'Male', 1: "Female"}

image_paths = []
gender_labels = []
age_labels = []

# Step 2: Converting Dataset to Pandas Dataframe.
for img in tqdm(os.listdir(Base_img_dir)):
    image_path = os.path.join(Base_img_dir, img)
    Age = int(img.split("_")[0])
    Gender = int(img.split("_")[1])
    gender_labels.append(Gender)
    age_labels.append(Age)
    image_paths.append(image_path)

df = pd.DataFrame()
df['ImagePath'], df['Age'], df['Gender'] = image_paths, age_labels, gender_labels
print(df.head())

# Step 3: EDA (Exploratory Data Analysis, to visualize the data patterns of datasets)
# Skipped in the script as it is not mandatory for the script to run, just for visualization purpose.

# Step 4: Converting Dataframes to numpy array as keras model supports numpy arrays as input.
def feature_extract(image_paths):
    features = []
    for img_path in tqdm(image_paths):
        img = load_img(img_path, grayscale = True)
        img = img.resize((128,128), Image.ANTIALIAS)
        img = np.array(img)
        features.append(img)

    features = np.array(features)
    # before Reshaping:  (23708, 128, 128)
    # After Reshaping:  (23708, 128, 128, 1)

    # print("before Reshaping: ", features.shape)

    features = features.reshape((len(features), 128, 128, 1))
    return features

# Step 5: Normalizing data and categorizing in dependant and independant category.
X = feature_extract(df['ImagePath'])
X = X/255.0 ### min max normalization (x-0/255-0)
y_gender = np.array(df['Gender'])
y_age = np.array(df['Age'])


# print("After Reshaping: ", X.shape)

# Step 6: Model Creation.
input_shape = (128, 128, 1)

ip = Input(shape=input_shape)
conv1 = Conv2D(32, (3,3), strides=(2, 2), activation='relu')(ip)
maxp1 = MaxPooling2D()(conv1)

conv2 = Conv2D(64, (3,3), strides=(2, 2), activation='relu')(maxp1)
maxp2 = MaxPooling2D()(conv2)

conv3 = Conv2D(128, (3,3), strides=(2, 2), activation='relu')(maxp2)
maxp3 = MaxPooling2D()(conv3)

flatten = Flatten() (maxp3)
op1 = Dense(1, activation='sigmoid', name = 'Gender_out')(flatten)
op2 = Dense(1, activation='relu', name = 'Age_out')(flatten)

model = Model(inputs=[ip], outputs=[op1, op2])


# Step 7: Model Visualization.
plot_model(model)

# Step 8: Compiling model with loss, optimizer and metrics to be used.
model.compile(loss=['binary_crossentropy', 'mae'], optimizer='adam', metrics=['accuracy'])

# Step 9: Training the model with batch size, epochs etc.
training = model.fit(x=X, y=[y_gender, y_age], batch_size=32, epochs=1, validation_split=0.2)

# Step 10: Plot the loss and accuracy graphs to check the performance of the model.
# this step can be done to finetune the model after visualizing it's not mandatory to plot, so skipped.

# Step 11: Predicting the output.
pred = model.predict(X[100].reshape(1, 128, 128, 1))
print("pred: {}\npred_shape:{}".format(pred, pred.shape))