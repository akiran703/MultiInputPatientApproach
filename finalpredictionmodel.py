from PIL import Image
import glob
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from sklearn.model_selection import train_test_split
import numpy as np
import tensorflow.keras as tf
from tensorflow.keras import Sequential
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.callbacks import CSVLogger
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten,Concatenate, Dropout, BatchNormalization
from tensorflow.keras.utils import to_categorical
from keras import backend as K
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
import itertools
import matplotlib.pyplot as plt


import os
import pandas as pd
import csv
import cv2
import random
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



def main():
            auth.authenticate_user()
            gauth = GoogleAuth()
            gauth.credentials = GoogleCredentials.get_application_default()
            drive = GoogleDrive(gauth)

            download_CT_COVID = drive.CreateFile({'id':'1bDI0Lokr1Do1adDq6kY2I6NIedaPGDbS'})
            download_CT_COVID.GetContentFile('clustercovid.zip')

            #-------------------------------setting up the data for training, validation and test--------------------------------------------------------------------------------

            file = files.upload()

            dfcovidtest = pd.read_csv('update_covid_test.csv')
            patientidcovidtest = dfcovidtest["Patient ID"].value_counts().keys().tolist()



            dfcovidtrain = pd.read_csv('update_covid_train.csv')
            patientidcovidtrain = dfcovidtrain["Patient ID"].value_counts().keys().tolist()
     


            dfcovidval = pd.read_csv('update_covid_val.csv')
            patientidcovidval = dfcovidval["Patient ID"].value_counts().keys().tolist()
       



            dfnoncovidtest = pd.read_csv('update_noncovid_test.csv')
            patientidnoncovidtest = dfnoncovidtest["patient id"].value_counts().keys().tolist()
     



            dfnoncovidtrain = pd.read_csv('update_noncovid_train.csv')
            patientidnoncovidtrain = dfnoncovidtrain["patient id"].value_counts().keys().tolist()
        



            dfnoncovidval = pd.read_csv('update_noncovid_val.csv')
            patientidnoncovidval = dfnoncovidval["patient id"].value_counts().keys().tolist()
           

            covidtest = []
            covidtrain = []
            covidval = []

            covidtestlabel = []
            covidtrainlabel = []
            covidvallabel = []

            covidtestid = []
            covidtrainid = []
            covidvalid = []

            noncovidtest = []
            noncovidtrain = []
            noncovidval = []

            noncovidtestlabel = []
            noncovidtrainlabel = []
            noncovidvallabel = []


            noncovidtestid = []
            noncovidtrainid = []
            noncovidvalid = []

            desired_size=200            
            
            #---------------------------------------normalize covid images------------------------
            def myFunc(name):
                    img = cv2.imread(name)
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
                    covidtest.append(normresize)
                    covidtestlabel.append(1)



            def trainmyFunc(name):
                    img = cv2.imread(name)
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
                    covidtrain.append(normresize)
                    covidtrainlabel.append(1)



            def valmyFunc(name):
                    img = cv2.imread(name)
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
                    covidval.append(normresize)
                    covidvallabel.append(1)



            def seekvalue(y):
                findValue = dfcovidtest.loc[dfcovidtest['Patient ID'] == y]
                for i, row in findValue.iterrows():
                    image_value = f"{row['File name']}"
                    f1 = f'/content/Data/clustercovid/coviddata/CT_COVID/{image_value}'
                    covidtestid.append(y)
                    myFunc(f1)


            def trainseekvalue(y):
                findValue = dfcovidtrain.loc[dfcovidtrain['Patient ID'] == y]
                for i, row in findValue.iterrows():
                    image_value = f"{row['File name']}"
                    f1 = f'/content/Data/clustercovid/coviddata/CT_COVID/{image_value}'
                    covidtrainid.append(y)
                    trainmyFunc(f1)

            def valseekvalue(y):
                findValue = dfcovidval.loc[dfcovidval['Patient ID'] == y]
                for i, row in findValue.iterrows():
                    image_value = f"{row['File name']}"
                    f1 = f'/content/Data/clustercovid/coviddata/CT_COVID/{image_value}'
                    covidvalid.append(y)
                    valmyFunc(f1)



            for i in range(len(patientidcovidtest)):
                seekvalue(patientidcovidtest[i])


            for i in range(len(patientidcovidtrain)):
                trainseekvalue(patientidcovidtrain[i])

            for i in range(len(patientidcovidval)):
                valseekvalue(patientidcovidval[i])


            #---------------------------------normalize noncovid images-----------------

            def nonmyFunc(name):
                    img = cv2.imread(name)
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
                    noncovidtest.append(normresize)
                    noncovidtestlabel.append(0)


            def nontrainmyFunc(name):
                    img = cv2.imread(name)
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
                    noncovidtrain.append(normresize)
                    noncovidtrainlabel.append(0)


            def nonvalmyFunc(name):
                    img = cv2.imread(name)
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
                    noncovidval.append(normresize)
                    noncovidvallabel.append(0)



            def nonseekvalue(y):
                findValue = dfnoncovidtest.loc[dfnoncovidtest['patient id'] == y]
                for i, row in findValue.iterrows():
                    image_value = f"{row['image name']}"
                    f1 = f'/content/Data/clustercovid/coviddata/CT_NonCOVID/{image_value}'
                    noncovidtestid.append(y)
                    nonmyFunc(f1)


            def nontrainseekvalue(y):
                findValue = dfnoncovidtrain.loc[dfnoncovidtrain['patient id'] == y]
                for i, row in findValue.iterrows():
                    image_value = f"{row['image name']}"
                    f1 = f'/content/Data/clustercovid/coviddata/CT_NonCOVID/{image_value}'
                    noncovidtrainid.append(y)
                    nontrainmyFunc(f1)

            def nonvalseekvalue(y):
                findValue = dfnoncovidval.loc[dfnoncovidval['patient id'] == y]
                for i, row in findValue.iterrows():
                    image_value = f"{row['image name']}"
                    f1 = f'/content/Data/clustercovid/coviddata/CT_NonCOVID/{image_value}'
                    noncovidvalid.append(y)
                    nonvalmyFunc(f1)


            for i in range(len(patientidnoncovidtest)):
                nonseekvalue(patientidnoncovidtest[i])

            for i in range(len(patientidnoncovidtrain)):
                nontrainseekvalue(patientidnoncovidtrain[i])

            for i in range(len(patientidnoncovidval)):
                nonvalseekvalue(patientidnoncovidval[i])
                
            #----------------------------------------------------combine covid + noncovid iamges and randomize it-----------

            test = noncovidtest + covidtest
            train = covidtrain + noncovidtrain
            val = noncovidval + covidval

            testlabel = covidtestlabel + noncovidtestlabel
            trainlabel = noncovidtrainlabel + covidtrainlabel
            vallabel = covidvallabel + noncovidvallabel

            testid = noncovidtestid + covidtestid
            trainid = noncovidtrainid + covidtrainid
            valid = covidvalid + noncovidvalid

       

            rand_order = np.random.permutation(len(test))
            test = np.array(test)
            test = test[rand_order, :, :, :]
            testlabel = to_categorical(testlabel)
            testlabel = testlabel[rand_order]
            testid = np.array(testid)
            testid = testid[rand_order]


            rand_order = np.random.permutation(len(train))
            train = np.array(train)
            train = train[rand_order, :, :, :]
            trainlabel = to_categorical(trainlabel)
            trainlabel = trainlabel[rand_order]
            trainid = np.array(trainid)
            trainid = trainid[rand_order]

            rand_order = np.random.permutation(len(val))
            val = np.array(val)
            val = val[rand_order, :, :, :]
            vallabel = to_categorical(vallabel)
            vallabel = vallabel[rand_order]
            valid = np.array(valid)
            valid = valid[rand_order]
            
            
            #---------------------------------build neural network--------------------------------

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

            # Create 1 input branches
            input1 = Input(shape=(desired_size, desired_size, 3), name='input1')
           


            # Create AlexNet branches
            branch1 = create_alexnet_branch(input1, 'branch1')


            # Concatenate all branches
            concatenated = Concatenate(axis=-1, name='concatenate')([branch1])

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
            model = Model(inputs=[input1], outputs=output, name='AlexNet')
            model.summary()


            #---------------------------------------train model and save weights for transfer learning---------------------------
            #save weights
            checkpoint_filepath = '/content/checkpoint_predictionmodel.hdf5'
            model_checkpoint_callback = tf.callbacks.ModelCheckpoint(filepath=checkpoint_filepath,save_weights_only=True,
                monitor='val_loss',
                mode='min',
                save_best_only=True)

            csv_log_paper = CSVLogger("prediction_paper.csv")

            # complie
            optimizer=tf.optimizers.SGD(learning_rate=0.05)
            model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

            #history = model.fit(x=[pic1,pic2,pic3,pic4], y=endtrainlabel, batch_size=32, epochs=20,verbose=True,validation_data=([pic111,pic222,pic333,pic444],endvalabel))
            history = model.fit(train, trainlabel, batch_size=32, epochs=50, callbacks=[model_checkpoint_callback],verbose=True, validation_data=(val,vallabel))

            model.load_weights(checkpoint_filepath)

            download_transfer = drive.CreateFile({'id':'1NTcd09yqjCh5a3kkWqiCGcclv8Nwo63q'})
            download_transfer.GetContentFile('archive.zip')


            
            #-------------------------load in second data (this data has images associated with patient)-----------------------

            img_dir_transfer = "/content/transfer/New_Data_CoV2/Covid"
            img_dirnon_transfer = "/content/transfer/New_Data_CoV2/Healthy"

            healthy_patient_dirs = glob.glob("/content/transfer/New_Data_CoV2/Healthy/*/")
            covid_patient_dirs = glob.glob("/content/transfer/New_Data_CoV2/Covid/*/")

            # read in on an image-by-image basis
            healthy_patient_imgs = glob.glob("/content/transfer/New_Data_CoV2/Healthy/*/*.png")
            covid_patient_imgs = glob.glob("/content/transfer/New_Data_CoV2/Covid/*/*.png")

            img_covid =[]
            img_covid_label = []

            img_healthy =[]
            img_healthy_label = []

            covidtransferid = []
            healthytransferid = []

            counter = 0
            healthycounter = 1
            
            #---------------------------normalize covid and noncovid images-------------

            for curr_dir in healthy_patient_dirs:
                imgs = glob.glob(curr_dir+'/*.png')
                for img in imgs:
                    img = cv2.imread(img)
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
                    img_healthy.append(normresize)
                    healthytransferid.append(healthycounter)
                    img_healthy_label.append([1])
                healthycounter = healthycounter + 2

            for curr_dir in covid_patient_dirs:
                imgs = glob.glob(curr_dir+'/*.png')
                for img in imgs:
                    img = cv2.imread(img)
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
                    img_covid.append(normresize)
                    covidtransferid.append(counter)
                    img_covid_label.append([0])
                counter = counter + 2

            #-----------------------------------break data into training, test, and validation + randomized----------------

            finaltrain = []
            finaltrainlabel = []
            finaltrainid = []


            finalval = []
            finalvallabel = []
            finalvalid = []


            finaltest = []
            finaltestlabel = []
            finaltestid = []

            for i in range(0,606):
                finaltrain.append(img_healthy[i])
                finaltrainlabel.append(img_healthy_label[i])
                finaltrainid.append(healthytransferid[i])


            for i in range(0,1755):
                finaltrain.append(img_covid[i])
                finaltrainlabel.append(img_covid_label[i])
                finaltrainid.append(covidtransferid[i])

            for i in range(606,757):
                finaltest.append(img_healthy[i])
                finaltestlabel.append(img_healthy_label[i])
                finaltestid.append(healthytransferid[i])

            for i in range(1755, 2167):
                finaltest.append(img_covid[i])
                finaltestlabel.append(img_covid_label[i])
                finaltestid.append(covidtransferid[i])



            for i in range(506,823):
                finalval.append(finaltrain[i])
                finalvallabel.append(finaltrainlabel[i])
                finalvalid.append(finaltrainid[i])

            del finaltrain[506:823]
            del finaltrainid[506:823]
            del finaltrainlabel[506:823]



            rand_order_train = np.random.permutation(len(finaltrain))
            rand_order_test = np.random.permutation(len(finaltest))
            rand_order_val = np.random.permutation(len(finalval))

            finaltrain = np.array(finaltrain)
            finaltrain = finaltrain[rand_order_train, :, :, :]

            finaltest = np.array(finaltest)
            finaltest = finaltest[rand_order_test, :, :, :]

            finalval = np.array(finalval)
            finalval = finalval[rand_order_val, :, :, :]

            finaltrainlabel = to_categorical(finaltrainlabel)
            finaltrainlabel = finaltrainlabel[rand_order_train]

            finaltestlabel = to_categorical(finaltestlabel)
            finaltestlabel = finaltestlabel[rand_order_test]

            finalvallabel = to_categorical(finalvallabel)
            finalvallabel = finalvallabel[rand_order_val]



            finaltrainid = np.array(finaltrainid)
            finaltrainid = finaltrainid[rand_order_train]

            finaltestid = np.array(finaltestid)
            finaltestid = finaltestid[rand_order_test]

            finalvalid = np.array(finalvalid)
            finalvalid = finalvalid[rand_order_val]

            print(finaltrainlabel)

           

            #------------------------------------------------functions to calculate metrics---------------------------------------------------------
         

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

            #-----------------------------------------------------------------------model makes the predictions with transfer learning-----------------
            checkpoint_filepath = "/content/checkpoint_predictionmodel.hdf5"

            
            checkpoint_filepath_1 = '/content/please_work.hdf5'
            model_checkpoint_callback = tf.callbacks.ModelCheckpoint(filepath=checkpoint_filepath_1,save_weights_only=True,
                monitor='val_loss',
                mode='min',
                save_best_only=True)
            
            
            optimizer = tf.optimizers.SGD(learning_rate=0.05)
            model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

            model.load_weights(checkpoint_filepath)
            history = model.fit(finaltrain, finaltrainlabel, batch_size=32, epochs=50,callbacks=[model_checkpoint_callback],verbose=True,validation_data=(finalval,finalvallabel))

            model.load_weights(checkpoint_filepath_1)

            prediction = model.predict(finaltest,verbose=1)


            #-------------------------------------we store the patient id in a dictionary and count how many images they have in the testing data set


            def getDuplicatesWithInfo(listOfElems):
                ''' Get duplicate element in a list along with their indices in list and frequency count'''
                dictOfElems = dict()
                index = 0
                # Iterate over each element in list and keep track of index
                for elem in listOfElems:
                    # If element exists in dict then keep its index in lisr & increment its frequency
                    if elem in dictOfElems:
                        dictOfElems[elem][0] += 1
                        dictOfElems[elem][1].append(index)

                    else:
                        # Add a new entry in dictionary
                        dictOfElems[elem] = [1, [index]]
                    index += 1

                dictOfElems = { key:value for key, value in dictOfElems.items() if value[0] > 1}
                return dictOfElems

            dictOfElems = getDuplicatesWithInfo(finaltestid)
            for key, value in dictOfElems.items():
                    print('Element = ', key , ' :: Repeated Count = ', value[0] , ' :: Index Positions =  ', value[1])



            #---------------------------------------------------------average approach-----------------------------


            finalpred = []
            actuallabel = []

            for key, value in dictOfElems.items():
                sum = 0
                evaluation =0
                for i in range(dictOfElems[key][0]):
                    print(prediction[dictOfElems[key][1][i]][1])
                    sum = sum + prediction[dictOfElems[key][1][i]][1]
                print(dictOfElems[key][0])
                sum = sum/dictOfElems[key][0]
                print(sum)
                if(sum < .50):
                    evaluation = 0
                    finalpred.append(tf.cast(evaluation,tf.float32))
                elif(sum > .50):
                    evaluation = 1
                    finalpred.append(tf.cast(evaluation,tf.float32))
                actualvalue = finaltestlabel[dictOfElems[key][1][i]]
                actuallabel.append(tf.cast(actualvalue[1],tf.float32))
                print('Element = ', key ,' :: prediction evaluation = ',evaluation,' :: label = ',actualvalue[1] )
            finalpred = np.array(finalpred)
            actuallabel = np.array(actuallabel)


            #-------------------------------------majority vote counting-----------------------------------
            print("MAJORITY VOTE EVALUATION")


            finalpred_majority = []
            actuallabel_majority = []

            for key, value in dictOfElems.items():
                covid_votes = 0
                non_covid_votes = 0
                
                print(f"\nPatient ID: {key}")
                print("Individual slice predictions:")
                
                # Count votes for each slice
                for i in range(dictOfElems[key][0]):
                    slice_prediction = prediction[dictOfElems[key][1][i]][1]  # COVID probability
                    print(f"  Slice {i+1}: COVID prob = {slice_prediction:.4f}")
                    
                    # Vote based on 0.5 threshold
                    if slice_prediction > 0.5:
                        covid_votes += 1
                        print(f"    → Vote: COVID")
                    else:
                        non_covid_votes += 1
                        print(f"    → Vote: Non-COVID")
                
                # Final diagnosis based on majority vote
                total_slices = covid_votes + non_covid_votes
                print(f"\nVoting Summary:")
                print(f"  COVID votes: {covid_votes}/{total_slices}")
                print(f"  Non-COVID votes: {non_covid_votes}/{total_slices}")
                
                if covid_votes > non_covid_votes:
                    final_diagnosis = 1
                    print(f"  → MAJORITY VOTE DIAGNOSIS: COVID")
                elif non_covid_votes > covid_votes:
                    final_diagnosis = 0
                    print(f"  → MAJORITY VOTE DIAGNOSIS: Non-COVID")
                else:
                    # Tie-breaking: default to non-COVID (conservative approach)
                    final_diagnosis = 0
                    print(f"  → TIE: Defaulting to Non-COVID (conservative)")
                
                finalpred_majority.append(tf.cast(final_diagnosis, tf.float32))
                
                # Get actual label
                actualvalue = finaltestlabel[dictOfElems[key][1][0]]  # Same for all slices of this patient
                actuallabel_majority.append(tf.cast(actualvalue[1], tf.float32))
                
                print(f"  Actual label: {'COVID' if actualvalue[1] == 1 else 'Non-COVID'}")
                print(f"  Correct: {'✓' if final_diagnosis == actualvalue[1] else '✗'}")

            # Convert to numpy arrays
            finalpred_majority = np.array(finalpred_majority)
            actuallabel_majority = np.array(actuallabel_majority)
            
            
            
             #-------------------------------------Entropy-Based Weighting -----------------------------------
             
            #Calculates the Shannon entropy for a probability distribution
            def calculate_entropy(p, epsilon=1e-12):
                p = np.clip(p, epsilon, 1. - epsilon)
                return -np.sum(p * np.log2(p))

            print("ENTROPY-BASED WEIGHTING EVALUATION")

            finalpred_entropy = []
            actuallabel_entropy = []

            for key, value in dictOfElems.items():
                weighted_sum = 0.0
                total_weight = 0.0
                
                print(f"\nPatient ID: {key}")
                print("Individual slice predictions and weights:")

                # Loop through each slice for the patient
                for i in range(dictOfElems[key][0]):
                    # Get the prediction for the current slice
                    slice_pred_vector = prediction[dictOfElems[key][1][i]]
                    prob_covid = slice_pred_vector[1]

                    # Calculate entropy and weight
                    # Low entropy (high confidence) gets a high weight (close to 1)
                    # High entropy (low confidence) gets a low weight (close to 0)
                    entropy = calculate_entropy(slice_pred_vector)
                    weight = 1.0 - entropy

                    print(f"  Slice {i+1}: COVID prob = {prob_covid:.4f}, Entropy = {entropy:.4f}, Weight = {weight:.4f}")

                    # Accumulate weighted sum of COVID probabilities and total weight
                    weighted_sum += prob_covid * weight
                    total_weight += weight

                # Calculate final weighted average diagnosis
                # Handle edge case where total_weight is zero (all predictions were max entropy)
                if total_weight > 0:
                    final_score = weighted_sum / total_weight
                else:
                    final_score = 0.5  # Default to an uncertain score

                # Make final decision based on the weighted score
                final_diagnosis = 1 if final_score > 0.5 else 0
                finalpred_entropy.append(tf.cast(final_diagnosis, tf.float32))

                # Get the actual label (it's the same for all slices of this patient)
                actualvalue = finaltestlabel[dictOfElems[key][1][0]]
                actuallabel_entropy.append(tf.cast(actualvalue[1], tf.float32))

                print(f"\nFinal Weighted Score: {final_score:.4f}")
                print(f"  → ENTROPY-WEIGHTED DIAGNOSIS: {'COVID' if final_diagnosis == 1 else 'Non-COVID'}")
                print(f"  Actual label: {'COVID' if actualvalue[1] == 1 else 'Non-COVID'}")
                print(f"  Correct: {'✓' if final_diagnosis == actualvalue[1] else '✗'}")

            # Convert lists to numpy arrays for metric calculation
            finalpred_entropy = np.array(finalpred_entropy)
            actuallabel_entropy = np.array(actuallabel_entropy)
            
       

            #-------------------------------------Z-score normalization approach-----------------------------------------
            print("Z-SCORE NORMALIZATION EVALUATION")

            finalpred_zscore = []
            actuallabel_zscore = []

            for key, value in dictOfElems.items():
                # Step 1: Collect all COVID probabilities for the current patient's slices
                patient_slice_probs = np.array([prediction[i][1] for i in value[1]])
                
                # Step 2: Calculate the mean and standard deviation of probabilities for this patient
                mean_prob = np.mean(patient_slice_probs)
                std_dev = np.std(patient_slice_probs)
                
                print(f"\nPatient ID: {key}")
                print(f"  Slice Probs: {[f'{p:.2f}' for p in patient_slice_probs]}")
                print(f"  Mean={mean_prob:.4f}, Std Dev={std_dev:.4f}")

                # Step 3: Calculate Z-scores for each slice probability
                # Handle the edge case where standard deviation is zero to avoid division by zero
                if std_dev > 1e-6:  # Use a small epsilon for floating-point stability
                    z_scores = (patient_slice_probs - mean_prob) / std_dev
                else:
                    # If all predictions are identical, their deviation from the mean is zero
                    z_scores = np.zeros_like(patient_slice_probs)
                
                print(f"  Z-Scores: {[f'{z:.2f}' for z in z_scores]}")

                # Step 4: Average the Z-scores to get the final patient-level score
                avg_z_score = np.mean(z_scores)

                # Step 5: Make a final decision. A positive average Z-score suggests the
                # distribution of predictions is skewed towards values higher than the patient's own average.
                final_diagnosis = 1 if avg_z_score > 0 else 0
                
                finalpred_zscore.append(tf.cast(final_diagnosis, tf.float32))

                # Get the actual label for the patient
                actualvalue = finaltestlabel[value[1][0]]
                actuallabel_zscore.append(tf.cast(actualvalue[1], tf.float32))

                print(f"\nFinal Average Z-Score: {avg_z_score:.4f}")
                print(f"  → Z-SCORE DIAGNOSIS: {'COVID' if final_diagnosis == 1 else 'Non-COVID'}")
                print(f"  Actual label: {'COVID' if actualvalue[1] == 1 else 'Non-COVID'}")
                print(f"  Correct: {'✓' if final_diagnosis == final_diagnosis == actualvalue[1] else '✗'}")

            # Convert lists to numpy arrays for metric calculation
            finalpred_zscore = np.array(finalpred_zscore)
            actuallabel_zscore = np.array(actuallabel_zscore)
            
            
            
            #-----------------------------Bayesian Model Averaging approach---------------
            
            print("\nBAYESIAN MODEL AVERAGING (BMA) EVALUATION")

            finalpred_bma = []
            actuallabel_bma = []

            for key, value in dictOfElems.items():
                weighted_sum_probs = 0.0
                total_weight = 0.0
                
                slice_details_for_logging = []

                # Loop through each slice for the patient to calculate its contribution
                for i in value[1]:
                    prob_covid = prediction[i][1]
                    
                    # The weight is a proxy for the posterior probability of this slice's "model".
                    # We define it by the model's confidence: how far the prediction is from 0.5.
                    # We scale by 2 to map the weight to a more intuitive [0, 1] range.
                    weight = 2 * abs(prob_covid - 0.5)
                    
                    # Accumulate the weighted sum of probabilities and the sum of weights
                    weighted_sum_probs += weight * prob_covid
                    total_weight += weight
                    slice_details_for_logging.append((prob_covid, weight))

                # Calculate the final BMA score
                # Handle the edge case where all predictions are exactly 0.5 (total_weight is zero)
                if total_weight > 1e-6:
                    bma_score = weighted_sum_probs / total_weight
                else:
                    bma_score = 0.5 # The result is maximally uncertain

                # Make final decision based on the BMA score
                final_diagnosis = 1 if bma_score > 0.5 else 0
                
                finalpred_bma.append(tf.cast(final_diagnosis, tf.float32))

                # Get the actual label for the patient
                actualvalue = finaltestlabel[value[1][0]]
                actuallabel_bma.append(tf.cast(actualvalue[1], tf.float32))

                # Print details for this patient for clarity
                print(f"\nPatient ID: {key}")
                for prob, w in slice_details_for_logging:
                    print(f"  Slice Prob: {prob:.4f}, Weight (Confidence): {w:.4f}")
                print(f"\nFinal BMA Score: {bma_score:.4f}")
                print(f"  → BMA DIAGNOSIS: {'COVID' if final_diagnosis == 1 else 'Non-COVID'}")
                print(f"  Actual label: {'COVID' if actualvalue[1] == 1 else 'Non-COVID'}")
                print(f"  Correct: {'✓' if final_diagnosis == actualvalue[1] else '✗'}")


            # Convert lists to numpy arrays for metric calculation
            finalpred_bma = np.array(finalpred_bma)
            actuallabel_bma = np.array(actuallabel_bma)
            
            
            
            
            


            #---------------------------------------------Display data----------------------------------------------------------------



            # output metrics

            print("Averaging methond RESULTS")

            print("AVERAGING METHOD:")
            print(f"  Recall: {recall_m(actuallabel, finalpred):.2f}")
            print(f"  Precision: {precision_m(actuallabel, finalpred):.2f}")
            print(f"  F1-Score: {f1_m(actuallabel, finalpred):.2f}")


            print(classification_report(y_true=actuallabel, y_pred=finalpred))

            tn, fp, fn, tp = confusion_matrix(y_true=actuallabel, y_pred=finalpred).ravel()
            (tn, fp, fn, tp)


            print("MAJORITY VOTE RESULTS")


            print("MAJORITY VOTE METHOD:")
            print(f"  Recall: {recall_m(actuallabel_majority, finalpred_majority):.2f}")
            print(f"  Precision: {precision_m(actuallabel_majority, finalpred_majority):.2f}")
            print(f"  F1-Score: {f1_m(actuallabel_majority, finalpred_majority):.2f}")



            print(classification_report(y_true=actuallabel_majority, y_pred=finalpred_majority))

            tn, fp, fn, tp = confusion_matrix(y_true=actuallabel_majority, y_pred=finalpred_majority).ravel()
            (tn, fp, fn, tp)
            
            
           
            print("ENTROPY-BASED WEIGHTING RESULTS")
            
            
            print(f" Recall:    {recall_m(actuallabel_entropy, finalpred_entropy):.2f}")
            print(f" Precision: {precision_m(actuallabel_entropy, finalpred_entropy):.2f}")
            print(f" F1-Score:  {f1_m(actuallabel_entropy, finalpred_entropy):.2f}")
            
            
            
            print(classification_report(y_true=actuallabel_entropy, y_pred=finalpred_entropy))
            
            tn, fp, fn, tp = confusion_matrix(y_true=actuallabel_entropy, y_pred=finalpred_entropy).ravel()
            (tn, fp, fn, tp)
            
            
            
            
            print("Z-SCORE NORMALIZATION RESULT")
            
            print(f"Recall:    {recall_m(actuallabel_zscore, finalpred_zscore):.2f}")
            print(f"Precision: {precision_m(actuallabel_zscore, finalpred_zscore):.2f}")
            print(f"F1-Score:  {f1_m(actuallabel_zscore, finalpred_zscore):.2f}\n")
            
            print(classification_report(y_true=actuallabel_zscore, y_pred=finalpred_zscore))
            tn, fp, fn, tp = confusion_matrix(y_true=actuallabel_zscore, y_pred=finalpred_zscore).ravel()
            (tn, fp, fn, tp)
            
            
            
            print("BAYESIAN MODEL AVERAGING RESULT")
            print(f"Recall:    {recall_m(actuallabel_bma, finalpred_bma):.2f}")
            print(f"Precision: {precision_m(actuallabel_bma, finalpred_bma):.2f}")
            print(f"F1-Score:  {f1_m(actuallabel_bma, finalpred_bma):.2f}\n")
            
            print(classification_report(y_true=actuallabel_bma, y_pred=finalpred_bma))
            tn, fp, fn, tp = confusion_matrix(y_true=actuallabel_bma, y_pred=finalpred_bma).ravel()
            (tn, fp, fn, tp)



if __name__ == '__main__':
    main()






