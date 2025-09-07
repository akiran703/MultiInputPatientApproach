from PIL import Image
import glob
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from sklearn.model_selection import train_test_split
import numpy as np
from keras import backend as K
import tensorflow.keras as tf
from tensorflow.keras import Sequential
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten,Concatenate, Dropout
from tensorflow.keras.layers import Input, Add, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D, AveragePooling2D, MaxPooling2D, GlobalMaxPooling2D,MaxPool2D
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.initializers import glorot_uniform
import os
import pandas as pd
import csv
import cv2
import random
from google.colab import files
import glob
from IPython.core.debugger import Tracer

from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive
from google.colab import auth
from oauth2client.client import GoogleCredentials
auth.authenticate_user()
gauth = GoogleAuth()
gauth.credentials = GoogleCredentials.get_application_default()
drive = GoogleDrive(gauth)


download_CT_COVID = drive.CreateFile({'id':'1bDI0Lokr1Do1adDq6kY2I6NIedaPGDbS'})
download_CT_COVID.GetContentFile('clustercovid.zip')


#-----------------------------------------------setting up the first data set by normalizing it and breaking into test, train, val-------------------

file = files.upload()

dfcovidtest = pd.read_csv('update_covid_test.csv')
patientidcovidtest = dfcovidtest["Patient ID"].value_counts().keys().tolist()
countcovidtest = dfcovidtest["Patient ID"].value_counts().tolist()


dfcovidtrain = pd.read_csv('update_covid_train.csv')
patientidcovidtrain = dfcovidtrain["Patient ID"].value_counts().keys().tolist()
countcovidtrain = dfcovidtrain["Patient ID"].value_counts().tolist()


dfcovidval = pd.read_csv('update_covid_val.csv')
patientidcovidval = dfcovidval["Patient ID"].value_counts().keys().tolist()
countcovidval = dfcovidval["Patient ID"].value_counts().tolist()



dfnoncovidtest = pd.read_csv('update_noncovid_test.csv')
patientidnoncovidtest = dfnoncovidtest["patient id"].value_counts().keys().tolist()
countnoncovidtest = dfnoncovidtest["patient id"].value_counts().tolist()



dfnoncovidtrain = pd.read_csv('update_noncovid_train.csv')
patientidnoncovidtrain = dfnoncovidtrain["patient id"].value_counts().keys().tolist()
countnoncovidtrain = dfnoncovidtrain["patient id"].value_counts().tolist()



dfnoncovidval = pd.read_csv('update_noncovid_val.csv')
patientidnoncovidval = dfnoncovidval["patient id"].value_counts().keys().tolist()
countnoncovidval = dfnoncovidval["patient id"].value_counts().tolist()

desired_size=224
inputparameter = 7

covidtrainfinalpic = {}
for i in range(inputparameter):
  covidtrainfinalpic['pic'+str(i)] = []
covidtrainfinallabel = []


covidtestfinal = {}
for i in range(inputparameter):
  covidtestfinal['pic'+str(i)] = []
covidtestfinallabel = []

covidvalfinal = {}
for i in range(inputparameter):
  covidvalfinal['pic'+str(i)] = []
covidvalfinallabel = []

noncovidtrainfinal = {}
for i in range(inputparameter):
  noncovidtrainfinal['nonpic'+str(i)] = []
noncovidtrainfinallabel = []


noncovidtestfinal = {}
for i in range(inputparameter):
  noncovidtestfinal['nonpic'+str(i)] = []
noncovidtestfinallabel = []

noncovidvalfinal = {}
for i in range(inputparameter):
  noncovidvalfinal['nonpic'+str(i)] = []
noncovidvalfinallabel = []

def createlist(p):
    if (len(p) > inputparameter):
        while (len(p) != inputparameter):
            popvalue = random.randrange(len(p))
            p.pop(popvalue)
    elif (len(p) < inputparameter):
        while (len(p) != inputparameter):
            p.append(np.zeros((desired_size, desired_size, 3)))
    for i in range(len(p)):
        covidtestfinal['pic' + str(i)].append(p[i])


def traincreatelist(p):
    if(len(p) > inputparameter):
        while(len(p) != inputparameter):
            popvalue = random.randrange(len(p))
            p.pop(popvalue)
    elif(len(p) < inputparameter):
        while(len(p) != inputparameter):
            p.append(np.zeros((desired_size, desired_size, 3)))
    for i in range(len(p)):
        covidtrainfinalpic['pic' + str(i)].append(p[i])


def valcreatelist(p):
    if(len(p) > inputparameter):
        while(len(p) != inputparameter):
            popvalue = random.randrange(len(p))
            p.pop(popvalue)
    elif(len(p) < inputparameter):
        while(len(p) != inputparameter):
            p.append(np.zeros((desired_size, desired_size, 3)))
    for i in range(len(p)):
        covidvalfinal['pic' + str(i)].append(p[i])

def myFunc(name):
    temptwo = []
    for i in range(len(name)):
        img = cv2.imread(name[i])
        # print(pId + " iam here" + f1)
        # imgreszie = cv2.resize(img, (480, 480))
        old_size = img.shape[:2]
        ratio = float(desired_size) / max(old_size)
        new_size = tuple([int(x * ratio) for x in old_size])
        img = cv2.resize(img, (new_size[1], new_size[0]))
        delta_w = desired_size - new_size[1]
        delta_h = desired_size - new_size[0]
        top, bottom = delta_h // 2, delta_h - (delta_h // 2)
        left, right = delta_w // 2, delta_w - (delta_w // 2)
        color = [0, 0, 0]
        imgreszie = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
        normresize = cv2.normalize(imgreszie, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        #print(normresize)
        temptwo.append(normresize)
    covidtestfinallabel.append(1)
    createlist(temptwo)


def trainmyFunc(name):
    temptwo = []
    for i in range(len(name)):
        img = cv2.imread(name[i])
        # print(pId + " iam here" + f1)
        # imgreszie = cv2.resize(img, (480, 480))
        old_size = img.shape[:2]
        ratio = float(desired_size) / max(old_size)
        new_size = tuple([int(x * ratio) for x in old_size])
        img = cv2.resize(img, (new_size[1], new_size[0]))
        delta_w = desired_size - new_size[1]
        delta_h = desired_size - new_size[0]
        top, bottom = delta_h // 2, delta_h - (delta_h // 2)
        left, right = delta_w // 2, delta_w - (delta_w // 2)
        color = [0, 0, 0]
        imgreszie = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
        normresize = cv2.normalize(imgreszie, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        #print(normresize)
        temptwo.append(normresize)
    covidtrainfinallabel.append(1)
    traincreatelist(temptwo)


def valmyFunc(name):
    temptwo = []
    for i in range(len(name)):
        img = cv2.imread(name[i])
        # print(pId + " iam here" + f1)
        # imgreszie = cv2.resize(img, (480, 480))
        old_size = img.shape[:2]
        ratio = float(desired_size) / max(old_size)
        new_size = tuple([int(x * ratio) for x in old_size])
        img = cv2.resize(img, (new_size[1], new_size[0]))
        delta_w = desired_size - new_size[1]
        delta_h = desired_size - new_size[0]
        top, bottom = delta_h // 2, delta_h - (delta_h // 2)
        left, right = delta_w // 2, delta_w - (delta_w // 2)
        color = [0, 0, 0]
        imgreszie = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
        normresize = cv2.normalize(imgreszie, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        #print(normresize)
        temptwo.append(normresize)
    covidvalfinallabel.append(1)
    valcreatelist(temptwo)


def seekvalue(y):
    findValue = dfcovidtest.loc[dfcovidtest['Patient ID'] == y]
    templist = []
    for i, row in findValue.iterrows():
        #print(f"Index: {i}")
        image_value = f"{row['File name']}"
        f1 = f'/content/Data/clustercovid/coviddata/CT_COVID/{image_value}'
        #print(f'{findValue} +  " patient id " + {f1}')
        templist.append(f1)
    myFunc(templist)


def trainseekvalue(y):
    findValue = dfcovidtrain.loc[dfcovidtrain['Patient ID'] == y]
    templist = []
    for i, row in findValue.iterrows():
        # print(f"Index: {i}")
        image_value = f"{row['File name']}"
        f1 = f'/content/Data/clustercovid/coviddata/CT_COVID/{image_value}'
        # print(f'{findValue} +  " patient id " + {f1}')
        templist.append(f1)
    trainmyFunc(templist)

def valseekvalue(y):
    findValue = dfcovidval.loc[dfcovidval['Patient ID'] == y]
    templist = []
    for i, row in findValue.iterrows():
        # print(f"Index: {i}")
        image_value = f"{row['File name']}"
        f1 = f'/content/Data/clustercovid/coviddata/CT_COVID/{image_value}'
        # print(f'{findValue} +  " patient id " + {f1}')
        templist.append(f1)
    valmyFunc(templist)



for i in range(len(patientidcovidtest)):
    x = seekvalue(patientidcovidtest[i])


for i in range(len(patientidcovidtrain)):
    x = trainseekvalue(patientidcovidtrain[i])

for i in range(len(patientidcovidval)):
    x = valseekvalue(patientidcovidval[i])

def noncreatelist(p):
    if (len(p) > inputparameter):
        while (len(p) != inputparameter):
            popvalue = random.randrange(len(p))
            p.pop(popvalue)
    elif (len(p) < inputparameter):
        while (len(p) != inputparameter):
            p.append(np.zeros((desired_size, desired_size, 3)))
    for i in range(len(p)):
        noncovidtestfinal['nonpic' + str(i)].append(p[i])

def nontraincreatelist(p):
    if(len(p) > inputparameter):
        while(len(p) != inputparameter):
            popvalue = random.randrange(len(p))
            p.pop(popvalue)
    elif(len(p) < inputparameter):
        while(len(p) != inputparameter):
            p.append(np.zeros((desired_size, desired_size, 3)))
    for i in range(len(p)):
        noncovidtrainfinal['nonpic' + str(i)].append(p[i])

def nonvalcreatelist(p):
    if(len(p) > inputparameter):
        while(len(p) != inputparameter):
            popvalue = random.randrange(len(p))
            p.pop(popvalue)
    elif(len(p) < inputparameter):
        while(len(p) != inputparameter):
            p.append(np.zeros((desired_size, desired_size, 3)))
    for i in range(len(p)):
        noncovidvalfinal['nonpic' + str(i)].append(p[i])




def nonmyFunc(name):
    temptwo = []
    for i in range(len(name)):
        img = cv2.imread(name[i])
        # print(pId + " iam here" + f1)
        # imgreszie = cv2.resize(img, (480, 480))
        old_size = img.shape[:2]
        ratio = float(desired_size) / max(old_size)
        new_size = tuple([int(x * ratio) for x in old_size])
        img = cv2.resize(img, (new_size[1], new_size[0]))
        delta_w = desired_size - new_size[1]
        delta_h = desired_size - new_size[0]
        top, bottom = delta_h // 2, delta_h - (delta_h // 2)
        left, right = delta_w // 2, delta_w - (delta_w // 2)
        color = [0, 0, 0]
        imgreszie = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
        normresize = cv2.normalize(imgreszie, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        #print(normresize)
        temptwo.append(normresize)
    noncovidtestfinallabel.append(0)
    noncreatelist(temptwo)


def nontrainmyFunc(name):
    temptwo = []
    for i in range(len(name)):
        img = cv2.imread(name[i])
        # print(pId + " iam here" + f1)
        # imgreszie = cv2.resize(img, (480, 480))
        old_size = img.shape[:2]
        ratio = float(desired_size) / max(old_size)
        new_size = tuple([int(x * ratio) for x in old_size])
        img = cv2.resize(img, (new_size[1], new_size[0]))
        delta_w = desired_size - new_size[1]
        delta_h = desired_size - new_size[0]
        top, bottom = delta_h // 2, delta_h - (delta_h // 2)
        left, right = delta_w // 2, delta_w - (delta_w // 2)
        color = [0, 0, 0]
        imgreszie = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
        normresize = cv2.normalize(imgreszie, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        #print(normresize)
        temptwo.append(normresize)
    noncovidtrainfinallabel.append(0)
    nontraincreatelist(temptwo)


def nonvalmyFunc(name):
    temptwo = []
    for i in range(len(name)):
        img = cv2.imread(name[i])
        # print(pId + " iam here" + f1)
        # imgreszie = cv2.resize(img, (480, 480))
        old_size = img.shape[:2]
        ratio = float(desired_size) / max(old_size)
        new_size = tuple([int(x * ratio) for x in old_size])
        img = cv2.resize(img, (new_size[1], new_size[0]))
        delta_w = desired_size - new_size[1]
        delta_h = desired_size - new_size[0]
        top, bottom = delta_h // 2, delta_h - (delta_h // 2)
        left, right = delta_w // 2, delta_w - (delta_w // 2)
        color = [0, 0, 0]
        imgreszie = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
        normresize = cv2.normalize(imgreszie, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        #print(normresize)
        temptwo.append(normresize)
    noncovidvalfinallabel.append(0)
    nonvalcreatelist(temptwo)





def nonseekvalue(y):
    findValue = dfnoncovidtest.loc[dfnoncovidtest['patient id'] == y]
    templist = []
    for i, row in findValue.iterrows():
        #print(f"Index: {i}")
        image_value = f"{row['image name']}"
        f1 = f'/content/Data/clustercovid/coviddata/CT_NonCOVID/{image_value}'
        #print(f'{findValue} +  " patient id " + {f1}')
        templist.append(f1)
    nonmyFunc(templist)


def nontrainseekvalue(y):
    findValue = dfnoncovidtrain.loc[dfnoncovidtrain['patient id'] == y]
    templist = []
    for i, row in findValue.iterrows():
        # print(f"Index: {i}")
        image_value = f"{row['image name']}"
        f1 = f'/content/Data/clustercovid/coviddata/CT_NonCOVID/{image_value}'
        # print(f'{findValue} +  " patient id " + {f1}')
        templist.append(f1)
    nontrainmyFunc(templist)

def nonvalseekvalue(y):
    findValue = dfnoncovidval.loc[dfnoncovidval['patient id'] == y]
    templist = []
    for i, row in findValue.iterrows():
        # print(f"Index: {i}")
        image_value = f"{row['image name']}"
        f1 = f'/content/Data/clustercovid/coviddata/CT_NonCOVID/{image_value}'
        # print(f'{findValue} +  " patient id " + {f1}')
        templist.append(f1)
    nonvalmyFunc(templist)


for i in range(len(patientidnoncovidtest)):
    x = nonseekvalue(patientidnoncovidtest[i])

for i in range(len(patientidnoncovidtrain)):
    x = nontrainseekvalue(patientidnoncovidtrain[i])

for i in range(len(patientidnoncovidval)):
    x = nonvalseekvalue(patientidnoncovidval[i])

endtrainlabel = covidtrainfinallabel + noncovidtrainfinallabel
endtrainlabel = np.array(endtrainlabel)
endtrainlabel = to_categorical(endtrainlabel)
#endtrainlabel = np.reshape(endtrainlabel,(1,233,2))


endtestlabel = covidtestfinallabel + noncovidtestfinallabel
endtestlabel = np.array(endtestlabel)
endtestlabel = to_categorical(endtestlabel)
#endtestlabel = np.reshape(endtestlabel,(1,95,2))

endvalabel = covidvalfinallabel + noncovidvalfinallabel
endvalabel = np.array(endvalabel)
endvalabel = to_categorical(endvalabel)
#endvalabel = np.reshape(endvalabel,(1,56,2))


pic1 = (list(covidtrainfinalpic.values())[0] + list(noncovidtrainfinal.values())[0])
pic2 = (list(covidtrainfinalpic.values())[1] + list(noncovidtrainfinal.values())[1])
pic3 = (list(covidtrainfinalpic.values())[2] + list(noncovidtrainfinal.values())[2])
pic4 = (list(covidtrainfinalpic.values())[3] + list(noncovidtrainfinal.values())[3])
pic5 = (list(covidtrainfinalpic.values())[4] + list(noncovidtrainfinal.values())[4])
pic6 = (list(covidtrainfinalpic.values())[5] + list(noncovidtrainfinal.values())[5])
pic7 = (list(covidtrainfinalpic.values())[6] + list(noncovidtrainfinal.values())[6])



pic11 = (list(covidtestfinal.values())[0] + list(noncovidtestfinal.values())[0])
pic22 = (list(covidtestfinal.values())[1] + list(noncovidtestfinal.values())[1])
pic33 = (list(covidtestfinal.values())[2] + list(noncovidtestfinal.values())[2])
pic44 = (list(covidtestfinal.values())[3] + list(noncovidtestfinal.values())[3])
pic55 = (list(covidtestfinal.values())[4] + list(noncovidtestfinal.values())[4])
pic66 = (list(covidtestfinal.values())[5] + list(noncovidtestfinal.values())[5])
pic77 = (list(covidtestfinal.values())[6] + list(noncovidtestfinal.values())[6])



pic111 = (list(covidvalfinal.values())[0] + list(noncovidvalfinal.values())[0])
pic222 = (list(covidvalfinal.values())[1] + list(noncovidvalfinal.values())[1])
pic333 = (list(covidvalfinal.values())[2] + list(noncovidvalfinal.values())[2])
pic444 = (list(covidvalfinal.values())[3] + list(noncovidvalfinal.values())[3])
pic555 = (list(covidvalfinal.values())[4] + list(noncovidvalfinal.values())[4])
pic666 = (list(covidvalfinal.values())[5] + list(noncovidvalfinal.values())[5])
pic777 = (list(covidvalfinal.values())[6] + list(noncovidvalfinal.values())[6])

#----------------------------------------set up model and train based on first data set----------------------------------------

# alexnet
#model = Sequential()
def create_alexnet_branch(input_tensor, branch_name):
    """
    Create a single AlexNet branch
    """
    # First Convolutional Layer
    x = Conv2D(96, (11, 11), strides=(4, 4), activation='relu', 
               name=f'{branch_name}_conv1', padding='valid')(input_tensor)
    x = MaxPooling2D((3, 3), strides=(2, 2), name=f'{branch_name}_pool1')(x)
    x = BatchNormalization(name=f'{branch_name}_bn1')(x)
    
    # Second Convolutional Layer
    x = Conv2D(256, (5, 5), activation='relu', 
               name=f'{branch_name}_conv2', padding='same')(x)
    x = MaxPooling2D((3, 3), strides=(2, 2), name=f'{branch_name}_pool2')(x)
    x = BatchNormalization(name=f'{branch_name}_bn2')(x)
    
    # Third Convolutional Layer
    x = Conv2D(384, (3, 3), activation='relu', 
               name=f'{branch_name}_conv3', padding='same')(x)
    
    # Fourth Convolutional Layer
    x = Conv2D(384, (3, 3), activation='relu', 
               name=f'{branch_name}_conv4', padding='same')(x)
    
    # Fifth Convolutional Layer
    x = Conv2D(256, (3, 3), activation='relu', 
               name=f'{branch_name}_conv5', padding='same')(x)
    x = MaxPooling2D((3, 3), strides=(2, 2), name=f'{branch_name}_pool5')(x)
    
    return x

# Create 7 input branches
input1 = Input(shape=(desired_size, desired_size, 3), name='input1')
input2 = Input(shape=(desired_size, desired_size, 3), name='input2')
input3 = Input(shape=(desired_size, desired_size, 3), name='input3')
input4 = Input(shape=(desired_size, desired_size, 3), name='input4')
input5 = Input(shape=(desired_size, desired_size, 3), name='input5')
input6 = Input(shape=(desired_size, desired_size, 3), name='input6')
input7 = Input(shape=(desired_size, desired_size, 3), name='input7')


# Create AlexNet branches
branch1 = create_alexnet_branch(input1, 'branch1')
branch2 = create_alexnet_branch(input2, 'branch2')
branch3 = create_alexnet_branch(input3, 'branch3')
branch4 = create_alexnet_branch(input4, 'branch4')
branch5 = create_alexnet_branch(input4, 'branch5')
branch6 = create_alexnet_branch(input4, 'branch6')
branch7 = create_alexnet_branch(input4, 'branch7')

# Concatenate all branches
concatenated = Concatenate(axis=-1, name='concatenate')([branch1, branch2, branch3, branch4, branch5, branch6, branch7])

# Flatten for fully connected layers
flattened = Flatten(name='flatten')(concatenated)

# Fully Connected Layers (similar to original AlexNet)
x = Dense(4096, activation='relu', name='fc1')(flattened)
x = Dropout(0.5, name='dropout1')(x)

x = Dense(4096, activation='relu', name='fc2')(x)
x = Dropout(0.5, name='dropout2')(x)

x = Dense(1000, activation='relu', name='fc3')(x)
x = Dropout(0.5, name='dropout3')(x)

# Output layer
output = Dense(2, activation='softmax', name='output')(x)

# Create model
model = Model(inputs=[input1, input2, input3, input4, input5, input6, input7], outputs=output, name='AlexNet_7Input')
model.summary()

#--------------------------------------------------------train the neural network and save the weights--------------------
#save weights
checkpoint_filepath = '/content/checkpoint_alexnet.hdf5'
model_checkpoint_callback = tf.callbacks.ModelCheckpoint(
    filepath=checkpoint_filepath,
    save_weights_only=True,
    monitor='val_loss',
    mode='min',
    save_best_only=True
)

# compile
optimizer = tf.optimizers.Adam(learning_rate=0.001)  # Using Adam for AlexNet
model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

history = model.fit(
    x=[np.array(pic1), np.array(pic2), np.array(pic3), np.array(pic4),np.array(pic5),np.array(pic6), np.array(pic7)], 
    y=endtrainlabel, 
    batch_size=32, 
    epochs=50,
    callbacks=[model_checkpoint_callback],
    verbose=True,
    validation_data=([np.array(pic111), np.array(pic222), np.array(pic333), np.array(pic444), np.array(pic555),np.array(pic666), np.array(pic777)], endvalabel)
)


#-----------------------------------start of second phase of workflow-----------------------------------------------------

download_transfer = drive.CreateFile({'id':'1NTcd09yqjCh5a3kkWqiCGcclv8Nwo63q'})
download_transfer.GetContentFile('archive.zip')



covidtransfer = {}
for i in range(inputparameter):
  covidtransfer['pic'+str(i)] = []

healthytransfer = {}
for i in range(inputparameter):
  healthytransfer['pic'+str(i)] = []

def transfernormalize(name):
    temptwo = []
    for i in range(len(name)):
        img = cv2.imread(name[i])
        # print(pId + " iam here" + f1)
        # imgreszie = cv2.resize(img, (480, 480))
        old_size = img.shape[:2]
        ratio = float(desired_size) / max(old_size)
        new_size = tuple([int(x * ratio) for x in old_size])
        img = cv2.resize(img, (new_size[1], new_size[0]))
        delta_w = desired_size - new_size[1]
        delta_h = desired_size - new_size[0]
        top, bottom = delta_h // 2, delta_h - (delta_h // 2)
        left, right = delta_w // 2, delta_w - (delta_w // 2)
        color = [0, 0, 0]
        imgreszie = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
        normresize = (imgreszie* 1.0)/np.max(imgreszie)
        temptwo.append(normresize)
    transfersize(temptwo)

def nontransfernormalize(name):
    temptwo = []
    for i in range(len(name)):
        img = cv2.imread(name[i])
        # print(pId + " iam here" + f1)
        # imgreszie = cv2.resize(img, (480, 480))
        old_size = img.shape[:2]
        ratio = float(desired_size) / max(old_size)
        new_size = tuple([int(x * ratio) for x in old_size])
        img = cv2.resize(img, (new_size[1], new_size[0]))
        delta_w = desired_size - new_size[1]
        delta_h = desired_size - new_size[0]
        top, bottom = delta_h // 2, delta_h - (delta_h // 2)
        left, right = delta_w // 2, delta_w - (delta_w // 2)
        color = [0, 0, 0]
        imgreszie = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
        normresize = (imgreszie* 1.0)/np.max(imgreszie)
        temptwo.append(normresize)
    healthytransfersize(temptwo)

def transfersize(p):
    if(len(p) > inputparameter):
        while(len(p) != inputparameter):
            popvalue = random.randrange(len(p))
            p.pop(popvalue)
    elif(len(p) < inputparameter):
        while(len(p) != inputparameter):
            p.append(np.zeros((desired_size, desired_size, 3)))
    for i in range(len(p)):
        covidtransfer['pic' + str(i)].append(p[i])

def healthytransfersize(p):
    if(len(p) > inputparameter):
        while(len(p) != inputparameter):
            popvalue = random.randrange(len(p))
            p.pop(popvalue)
    elif(len(p) < inputparameter):
        while(len(p) != inputparameter):
            p.append(np.zeros((desired_size, desired_size, 3)))
    for i in range(len(p)):
        healthytransfer['pic' + str(i)].append(p[i])
        
#--------------------------------set up second data set, normalize the data, break into train, test, val-----------------

img_dir_transfer = "/content/transfer/New_Data_CoV2/Covid"
img_dirnon_transfer = "/content/transfer/New_Data_CoV2/Healthy"

healthy_patient_dirs = glob.glob("/content/transfer/New_Data_CoV2/Healthy/*/")
covid_patient_dirs = glob.glob("/content/transfer/New_Data_CoV2/Covid/*/")

# read in on an image-by-image basis
healthy_patient_imgs = glob.glob("/content/transfer/New_Data_CoV2/Healthy/*/*.png")
covid_patient_imgs = glob.glob("/content/transfer/New_Data_CoV2/Covid/*/*.png")

data=[]

labels=[]
healthylabels = []

for curr_dir in healthy_patient_dirs:
    list_of_imgs =[]
    imgs = glob.glob(curr_dir+'/*.png')
    for img in imgs:
        list_of_imgs.append(img)
    transfernormalize(list_of_imgs)
    labels.append([0])

for curr_dir in covid_patient_dirs:
    list_of_imgs =[]
    imgs = glob.glob(curr_dir+'/*.png')
    for img in imgs:
        list_of_imgs.append(img)
    nontransfernormalize(list_of_imgs)
    healthylabels.append([1])

pic_1 = (list(covidtransfer.values())[0] + list(healthytransfer.values())[0])
pic_2 = (list(covidtransfer.values())[1] + list(healthytransfer.values())[1])
pic_3 = (list(covidtransfer.values())[2] + list(healthytransfer.values())[2])
pic_4 = (list(covidtransfer.values())[3] + list(healthytransfer.values())[3])
pic_5 = (list(covidtransfer.values())[4] + list(healthytransfer.values())[4])
pic_6 = (list(covidtransfer.values())[5] + list(healthytransfer.values())[5])
pic_7 = (list(covidtransfer.values())[6] + list(healthytransfer.values())[6])



rand_order = np.random.permutation(len(pic_1))
pic_1 =np.array(pic_1)
pic_1 = pic_1[rand_order, :, :, :]

rand_order = np.random.permutation(len(pic_2))
pic_2 =np.array(pic_2)
pic_2 = pic_2[rand_order, :, :, :]

rand_order = np.random.permutation(len(pic_3))
pic_3 =np.array(pic_3)
pic_3 = pic_3[rand_order, :, :, :]

rand_order = np.random.permutation(len(pic_4))
pic_4 =np.array(pic_4)
pic_4 = pic_4[rand_order, :, :, :]

rand_order = np.random.permutation(len(pic_5))
pic_5 =np.array(pic_5)
pic_5 = pic_5[rand_order, :, :, :]

rand_order = np.random.permutation(len(pic_6))
pic_6 =np.array(pic_6)
pic_6 = pic_6[rand_order, :, :, :]

rand_order = np.random.permutation(len(pic_7))
pic_7 =np.array(pic_7)
pic_7 = pic_7[rand_order, :, :, :]

transferlabel = labels + healthylabels
transferlabel = np.array(transferlabel)
transferlabel = to_categorical(transferlabel)
print(np.shape(transferlabel))
transferlabel = transferlabel[rand_order]

pic_1_train = []
pic_1_test = []
pic_1_val = []

pic_2_train = []
pic_2_test = []
pic_2_val = []

pic_3_train = []
pic_3_test = []
pic_3_val = []

pic_4_train = []
pic_4_test = []
pic_4_val = []


pic_5_train = []
pic_5_test = []
pic_5_val = []

pic_6_train = []
pic_6_test = []
pic_6_val = []

pic_7_train = []
pic_7_test = []
pic_7_val = []





trainlabel =[]
testlabel =[]
vallabel = []

#---------------------------------make sure all patients get their associated image and it is not spilling into other data groups

for i in range(0,78):
  pic_1_train.append(pic_1[i])
  pic_2_train.append(pic_2[i])
  pic_3_train.append(pic_3[i])
  pic_4_train.append(pic_4[i])
  pic_5_train.append(pic_5[i])
  pic_6_train.append(pic_6[i])
  pic_7_train.append(pic_7[i])


  trainlabel.append(transferlabel[i])



for p in range(78,104):
  pic_1_val.append(pic_1[p])
  pic_2_val.append(pic_2[p])
  pic_3_val.append(pic_3[p])
  pic_4_val.append(pic_4[p])
  pic_5_val.append(pic_5[p])
  pic_6_val.append(pic_6[p])
  pic_7_val.append(pic_7[p])


  vallabel.append(transferlabel[p])


for x in range(104,130):
  pic_1_test.append(pic_1[x])
  pic_2_test.append(pic_2[x])
  pic_3_test.append(pic_3[x])
  pic_4_test.append(pic_4[x])
  pic_5_test.append(pic_5[x])
  pic_6_test.append(pic_6[x])
  pic_7_test.append(pic_7[x])


  testlabel.append(transferlabel[x])


trainlabel = np.array(trainlabel)
testlabel = np.array(testlabel)
vallabel = np.array(vallabel)

checkpoint_filepath_first = '/content/checkpoint_4input_good.hdf5'


#save weights
checkpoint_filepath_lol = '/content/checkpoint_v1.hdf5'
model_checkpoint_callback = tf.callbacks.ModelCheckpoint(filepath=checkpoint_filepath_lol,save_weights_only=True,
    monitor='val_loss',
    mode='min',
    save_best_only=True)


#--------------------------------------metric functions--------------------

def recall_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = (true_positives * 1.0) / (possible_positives + K.epsilon())
    return recall

def precision_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = (true_positives * 1.0) / (predicted_positives + K.epsilon())
    return precision

def f1_m(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))

#------------------------------transfer learning and using the model to predict on test dataset---------------------------------
# complie
optimizer=tf.optimizers.SGD(learning_rate=0.05)
model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy',f1_m,precision_m, recall_m])

model.load_weights(checkpoint_filepath)
history = model.fit(x=[np.array(pic_1_train),np.array(pic_2_train),np.array(pic_3_train),np.array(pic_4_train),np.array(pic_5_train),np.array(pic_6_train),np.array(pic_7_train)], y=trainlabel, batch_size=32, epochs=50,callbacks=[model_checkpoint_callback],verbose=True,validation_data=([np.array(pic_1_val),np.array(pic_2_val),np.array(pic_3_val),np.array(pic_4_val),np.array(pic_5_val),np.array(pic_6_val),np.array(pic_7_val)],vallabel))

model.load_weights(checkpoint_filepath_lol)

loss, accuracy, f1_score, precision, recall =model.evaluate([np.array(pic_1_test),np.array(pic_2_test),np.array(pic_3_test),np.array(pic_4_test),np.array(pic_5_test),np.array(pic_6_test),np.array(pic_7_test)],testlabel,verbose=1)
print(f'Test loss: {loss:.2}')
print(f'Test accuracy: {accuracy:.2}')
print(f'Test f1_score: {f1_score:.2}')
print(f'Test precision: {precision:.2}')
print(f'Test recall: {recall:.2}')

model.load_weights(checkpoint_filepath_lol)
prediction = model.predict([np.array(pic_1_test),np.array(pic_2_test),np.array(pic_3_test),np.array(pic_4_test),np.array(pic_5_test),np.array(pic_6_test),np.array(pic_7_test)])


#------------------------------display metrics-------------------------


# Commented out IPython magic to ensure Python compatibility.
# %matplotlib inline
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
import itertools
import matplotlib.pyplot as plt

rounded_predictions = np.round(prediction)

print(classification_report(y_true=testlabel.argmax(axis=1), y_pred=rounded_predictions.argmax(axis=1)))

confusion_matrix(y_true=testlabel.argmax(axis=1), y_pred=rounded_predictions.argmax(axis=1))