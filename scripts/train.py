import argparse
import json
import urllib

import os
import numpy as np
import pandas as pd

import keras
from keras import models 
from keras import layers
from keras import optimizers
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Embedding, Flatten, Dense

import azureml.core
from azureml.core import Run
from azureml.core.dataset import Dataset
from azureml.core.datastore import Datastore
from azureml.core.model import Model
import azure.storage.blob
from azure.storage.blob import BlockBlobService
from io import StringIO

import os
import argparse
import pandas as pd
import numpy as np



print("Executing train.py")
print("As a data scientist, this is where I write my training code.")
print("Azure Machine Learning SDK version: {}".format(azureml.core.VERSION))

#-------------------------------------------------------------------
#
# Processing input arguments
#
#-------------------------------------------------------------------

parser = argparse.ArgumentParser("train")

parser.add_argument("--model_name", type=str, help="model name", dest="model_name", required=True)
parser.add_argument("--build_number", type=str, help="build number", dest="build_number", required=True)
# parser.add_argument('--data_folder', type=str, dest='data_folder', help='data folder mounting point')

args = parser.parse_args()

run = Run.get_context()
ws = run.experiment.workspace
ds = Datastore.get_default(ws)

print("Argument 1: %s" % args.model_name)
print("Argument 2: %s" % args.build_number)
# print("Argument 3: %s" % args.data_folder)

#-------------------------------------------------------------------
#
# Define internal variables
#
#-------------------------------------------------------------------

#-------------------------------------------------------------------
#
# Model Code 
#
#-------------------------------------------------------------------


def soloinsulin(df):
    lst=[]
    for i in range(len(df)):
        if df.iloc[i,47] >=2:
            lst.append(0)
        else:
            if df.iloc[i,38] ==1:
                lst.append(1)
            else:
                lst.append(0)
    return lst

def diagnosis(df):
    lst=[]
    for i in range(len(df)):
        if(( df.iloc[i,0].startswith('250') == True) or ( df.iloc[i,1].startswith('250') == True) or ( df.iloc[i,2].startswith('250') == True)):
            lst.append(1)
        else:
            lst.append(0)
    return lst

block_blob_service = BlockBlobService(account_name='snpws7226031546',
                              account_key='ufRwhm+pG/54g6XuKBC8lb2/LOUX/qoLCVZigZRxdxWp43gV8jr2op3BYjXadIenizbb2DeC81pzJi0P9m4tLg==' )
# get data from blob storage in the form of bytes
blob_byte_data = block_blob_service.get_blob_to_bytes('azureml-blobstore-c824db1b-c93b-4d21-b9fe-800dab866169','final_data.csv')
# convert to bytes data into pandas df to fit scaler transform
s=str(blob_byte_data.content,'utf-8')
bytedata = StringIO(s)
final=pd.read_csv(bytedata)
    
final_ds = Dataset.Tabular.from_delimited_files(path=[(ds, '/final_data.csv')])
 
# For each run, register a new version of the dataset and tag it with the build number.
# This provides full traceability using a specific Azure DevOps build number.
 
final_ds.register(workspace=ws, name="Diabetes Dataset", description="Diabetes Dataset",
    tags={"build_number": args.build_number}, create_new_version=True)
print('Diabetes dataset successfully registered.')

#pre-processing
## Replacing ? with mode - Caucasian
final['race'].replace({'?':'Caucasian'},inplace=True)
# replacing Unknown/Invalid with mode - Female
final['gender'].replace({'Unknown/Invalid':'Female'},inplace=True)
final['diag_1'].replace({'?':'missing'},inplace=True)
final['diag_2'].replace({'?':'missing'},inplace=True)
final['diag_3'].replace({'?':'missing'},inplace=True)
final['medical_specialty'].replace({'?':'missing'},inplace=True) # as close to 50% data is missing, we cannot replace with mode
# replacing ? with missing because many vals are missing values
final['payer_code'].replace({'?':'missing'},inplace=True)
final=final.drop(['payer_code','medical_specialty'],axis=1)
medyes = final[final['diabetesMed']=='Yes']
copymedyes=medyes.copy()
copymedyes.iloc[:,21:43]=copymedyes.iloc[:,21:43].replace(['No','Up','Down','Steady'],[0,1,1,1])
# counting the total no. of drugs given to each diabetic patient
copymedyes['Total_drugs'] =copymedyes.iloc[:,21:43].sum(axis=1)
copymedyes['Solo_Insulin'] =soloinsulin(copymedyes)
cmy=copymedyes.copy()
cmy['diagnosis']= diagnosis(cmy)
cmy=cmy.drop(cmy.columns[19:43],axis=1)
cols=[0,1,2,19,21]
cmy=cmy.drop(cmy.columns[cols],axis=1)
temp1=cmy.copy()
temp1.drop(['num_procedures','patient_nbr','encounter_id'],axis=1,inplace=True)

print("Creating Model")
# naives bayes
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

x=temp1.drop('Solo_Insulin',axis=1)
y=temp1['Solo_Insulin']

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=1,stratify=y)


num_train=x_train[['number_diagnoses','time_in_hospital','num_lab_procedures', 'num_medications', 'number_outpatient',
       'number_emergency', 'number_inpatient','Total_drugs']]
num_test=num=x_test[['number_diagnoses','time_in_hospital','num_lab_procedures', 'num_medications', 'number_outpatient',
       'number_emergency', 'number_inpatient','Total_drugs']]


sc=StandardScaler()
num_train=pd.DataFrame(sc.fit_transform(num_train),columns=num_train.columns)
num_test=pd.DataFrame(sc.transform(num_test),columns=num_test.columns)


x_train.index=np.arange(len(x_train))
x_test.index=np.arange(len(x_test))


x_train=pd.concat([x_train,num_train],axis=1)
x_test=pd.concat([x_test,num_test],axis=1)

x_train = pd.get_dummies(x_train)
x_test = pd.get_dummies(x_test)

y_train.index=np.arange(len(y_train))
y_test.index=np.arange(len(y_test))

import sklearn.metrics as metrics

model = GaussianNB()
model.fit(x_train, y_train)
y_predict = model.predict(x_test)


fpr,tpr,_ =metrics.roc_curve(y_test, y_predict)
aucmetric=metrics.auc(fpr,tpr)
acc=metrics.accuracy_score(y_test, y_predict)
kappa=metrics.cohen_kappa_score(y_test,y_predict)
f1=metrics.f1_score(y_test,y_predict)

print("Model build completed")

print(metrics.classification_report(y_test, y_predict))
print('Accuracy is {}'.format(metrics.accuracy_score(y_test, y_predict)))
print('AUC is {}'.format(aucmetric))
print('Kappa score is {}'.format(metrics.cohen_kappa_score(y_test,y_predict)))
print('F1-Score is {}'.format(metrics.f1_score(y_test,y_predict)))



#run.log(model.metrics_names[0], evaluation_metrics[0], 'Model test data loss')
#run.log(model.metrics_names[1], evaluation_metrics[1], 'Model test data accuracy')

print("Running model completed")


run.log('Accuracy', acc)
run.log('AUC', aucmetric)
run.log('Cohens Kappa score', kappa)
run.log('F1-score', f1)

print("Saving model files...")
# create a ./outputs/model folder in the compute target
# files saved in the "./outputs" folder are automatically uploaded into run history
import _pickle as cPickle
from sklearn.externals import joblib
# save the classifier

os.makedirs('outputs', exist_ok=True)
# note file saved in the outputs folder is automatically uploaded into experiment record
joblib.dump(value=model, filename='outputs/model.pkl')

print('Done')    
print("Model Saved")

#-------------------------------------------------------------------
#
# Evaluate the model
#
#-------------------------------------------------------------------

# print('Model evaluation will print the following metrics: ', model.metrics_names)
# evaluation_metrics = model.evaluate(x_test, y_test)
# print(evaluation_metrics)

# run = Run.get_context()
# run.log(model.metrics_names[0], evaluation_metrics[0], 'Model test data loss')
# run.log(model.metrics_names[1], evaluation_metrics[1], 'Model test data accuracy')

#-------------------------------------------------------------------
#
# Register the model the model
#
#-------------------------------------------------------------------

os.chdir("./outputs")

# The registered model references the data set used to provide its training data

model_description = 'Recommend solo insulin or a conjunction of other drugs/ treatment? '
model = Model.register(
    model_path='model.pkl',  # this points to a local file
    model_name=args.model_name,  # this is the name the model is registered as
    tags={"type": "classification", "run_id": run.id, "build_number": args.build_number},
    description=model_description,
    workspace=run.experiment.workspace,
    datasets=[('training data', final_ds)])

print("Model registered: {} \nModel Description: {} \nModel Version: {}".format(model.name, 
                                                                                model.description, model.version))
