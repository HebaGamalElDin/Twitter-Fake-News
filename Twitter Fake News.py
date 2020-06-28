# -*- coding: utf-8 -*-
"""
Created on Wed May  6 22:07:50 2020

@author: Heba Gamal El-Din
"""
#######################################
""" Importing Necessary Libraries """
######################################
import json
import os
import pandas as pd
import glob 
import random
from nltk.corpus import stopwords
stop = stopwords.words('english')
from textblob import Word
from sklearn import metrics, svm
from sklearn.metrics import classification_report,confusion_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV 
from keras.models import Sequential
from keras import layers
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import numpy as np
from keras.models import load_model
from keras.optimizers import Adam
from keras.layers import Dense
from keras.layers import LSTM, Dropout, Bidirectional
from keras.layers.embeddings import Embedding
from sklearn.ensemble import StackingClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
import pickle
import seaborn as sns
import matplotlib.pyplot as plt
plt.style.use('ggplot')
###################################
""" Reading JSON Files From 
    Directories and Load Them """
##################################
# Tweets_Non = []
# Reactions_Non = []
# Tweets = []
# Reactions = []
# Non_Rumours = []
# Rumours = []
# os.chdir("D:\Work\Martians Tech\Twitter Fake News\Twitter Fake News\pheme-rnr-dataset")
# Main_Folders = os.listdir(os.curdir)

# for I in Main_Folders:
#     PATH = r'D:\Work\Martians Tech\Twitter Fake News\Twitter Fake News\pheme-rnr-dataset\%s'%I +"\\non-rumours"
#     os.chdir(PATH)
#     Non_Rumours = os.listdir(os.curdir)
#     for X in Non_Rumours:
#         Path = X + "\\reactions"
#         Reactions_Non.append([json.loads(open(os.path.join(PATH, Path, pos_json), 'r').read()) for pos_json in os.listdir(Path) if pos_json.endswith('.json')])
#         Path = X + "\\source-tweet"
#         Tweets_Non.append(json.loads(open(glob.glob(os.path.join(PATH, Path,'*.json'))[0], 'r').read()))
    
#     PATH = r'D:\Work\Martians Tech\Twitter Fake News\Twitter Fake News\pheme-rnr-dataset\%s'%I +'\\rumours'
#     os.chdir(PATH)
#     Rumours = os.listdir(os.curdir)
#     for X in Rumours:
#         Path = X + "\\reactions"
#         Reactions.append([json.loads(open(os.path.join(PATH, Path, pos_json), 'r').read()) for pos_json in os.listdir(Path) if pos_json.endswith('.json')])
#         Path = X + "\source-tweet"
#         Tweets.append(json.loads(open(glob.glob(os.path.join(PATH, Path,'*.json'))[0], 'r').read()))

# Reactions_Non = Reactions_Non[:len(Reactions_Non)//2]
# Tweets_Non = Tweets_Non[:len(Tweets_Non)//2]

# #########################################
# """ Preparing Data Of The Fake News """
# ########################################
# Data = []
# for i in range(len(Tweets)):
#     Data.append([Tweets[i]['text'], [Reaction['text'] for Reaction in Reactions[i]], 1])

# #########################################
# """ Preparing Data Of Non Fake News """
# ########################################
# for j in range(len(Tweets_Non)):
#     Data.append([Tweets_Non[j]['text'], [Reaction['text'] for Reaction in Reactions_Non[j]], 0])
    
# random.shuffle(Data)  
# ########################
# """ Final DataFrame """ 
# ########################
# DF = pd.DataFrame(Data, columns = ['Tweets', 'Retweets', 'Label'])
# DF.to_csv('Fake-Non Fake News.csv')

###########################
""" Data Preprocessing """
##########################
DF = pd.read_csv('Fake-Non Fake News.csv', index_col=0)
DF.Tweets = DF.Tweets.str.lower().replace('[^\w\s]','').str.replace(r'http\S+', '').str.strip().str.replace('\s+', ' ')
DF.Tweets = DF.Tweets.apply(lambda x: ' '.join([Word(item).lemmatize() for item in x.split() if item not in stop]))
X = DF.Tweets
Y = DF.Label
Category = ["Fake" if y==1 else "Real" for y in Y]
Category = pd.Series(Category)
plt.title("Category Rates")
plt.xlabel("Category")
plt.ylabel("Count")
Category.value_counts().plot(kind='bar')

""" Feature Extraction """
###########################
Vectorizer = TfidfVectorizer(min_df=1, stop_words="english",smooth_idf=True,use_idf=True)
X_Vectors = (Vectorizer.fit_transform(X))

#######################
""" Data Splitting """
#######################
X_train1, X_test1, Y_train1, Y_test1 = train_test_split(X_Vectors, Y, test_size=0.25,random_state=109)

############
""" SVM """
############
# """ Parameter Tuning """
# Param = {'C': [0.1, 1, 10, 100, 1000],  
#               'gamma': ['auto', 1, 0.1, 0.01, 0.001, 0.0001], 
#               'kernel': ['rbf', 'linear']}  
# Grid = GridSearchCV(svm.SVC(), Param, refit = True, verbose = 3) 
# Grid.fit(X_train1.toarray(), Y_train1)
# print(Grid.best_params_) 
# print(Grid.best_estimator_) 
# Preds = Grid.predict(X_test1) 
# accuracy = metrics.accuracy_score(Preds, Y_test1)
# print ("SVM ACCURACY: ", accuracy * 100)
# print("SVM REPORT :\n ", classification_report(Y_test1, Preds)) 

""" Fitting With Best Parameters """
Classifier = svm.SVC(C=10, kernel='rbf', degree=3, gamma=1)
Classifier.fit(X_train1.toarray(), Y_train1)
filename = 'SVM.sav'
pickle.dump(Classifier, open(filename, 'wb'))
Predictions = Classifier.predict(X_test1.toarray())
tr_accuracy = metrics.accuracy_score(Y_train1, Y_train1)
ts_accuracy = metrics.accuracy_score(Predictions, Y_test1)
print ("SVM's Training ACCURACY : ", tr_accuracy * 100)
print ("SVM's Testing ACCURACY : ", ts_accuracy * 100)
print("SVM's REPORT :\n ", classification_report(Y_test1, Predictions))
print("SVM's Confusion Matrix :\n ", confusion_matrix(Y_test1,Predictions))

def Input_pipeline(Tweet, filename):
    Tweet = str(Tweet).lower().replace('[^\w\s]','').replace(r'http\S+', '').strip().replace('\s+', ' ')
    Tweet = ' '.join([Word(item).lemmatize() for item in Tweet.split() if item not in stop])
    TFIDF = Vectorizer.transform([Tweet]).toarray()
    Classifier_ = pickle.load(open(filename, 'rb'))
    Prediction = Classifier_.predict(TFIDF)
    if Prediction[0] == 0:
        return 'Real'
    elif Prediction[0] == 1:
        return 'Fake'
    return Prediction

Result = Input_pipeline("#ECOWAS has formally endorsed Dr. @NOIweala for the position of Director-General of the World Trade Organisation for the period of 2021-2025. Also called on other African and Non African countries to do same. #COVID19  @NigeriaGov @DigiCommsNG @ecowas_cedeao @_AfricanUnion @wto", filename)
print("This Tweet Is :: {}".format(Result))
#############################
""" Deep Learning Models """
#############################
""" Data Splitting For DNs """
X_train, X_test, Y_train, Y_test = train_test_split(DF.Tweets, Y, test_size=0.25,random_state=109)

""" Training History Plotting """
def plot_history(history):
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']

    loss = history.history['loss']
    val_loss = history.history['val_loss']
    x = range(1, len(acc) + 1)

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(x, acc, 'b', label='Training acc')
    plt.plot(x, val_acc, 'r', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(x, loss, 'b', label='Training loss')
    plt.plot(x, val_loss, 'r', label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()

##############################
""" Prepare Text Seuences """
#############################
tokenizer = Tokenizer(num_words=500)
tokenizer.fit_on_texts(X)
X_train1 = tokenizer.texts_to_sequences(X_train)
X_test1 = tokenizer.texts_to_sequences(X_test)
maxlen = 100
X_train_ = pad_sequences(X_train1, padding='post', maxlen=maxlen)
X_test_ = pad_sequences(X_test1, padding='post', maxlen=maxlen)


def Prediction_pip(Tweet, model):
    Tweet = tokenizer.texts_to_sequences(Tweet)
    Tweet_ = pad_sequences(Tweet, padding='post', maxlen=maxlen)
    Model = load_model(model)
    Prediction = Model.predict_classes(Tweet_)
    if Prediction[0] == 0:
        return 'Real'
    elif Prediction[0] == 1:
        return 'Fake'
    return Prediction
############
""" CNN """
############
np.random.seed(7)
vocab_size = len(tokenizer.word_index) + 1
embedding_dim = 100

CNN = Sequential()
CNN.add(layers.Embedding(vocab_size, embedding_dim, input_length=maxlen))
CNN.add(layers.Conv1D(128, 5, activation='relu'))
CNN.add(layers.GlobalMaxPooling1D())
CNN.add(Dropout(0.2))
CNN.add(layers.Dense(1, activation='sigmoid'))
CNN.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
CNN.summary()
history = CNN.fit(X_train_, Y_train, epochs=10, verbose=True, validation_data=(X_test_, Y_test), batch_size=20)
CNN.save("CNN")
Predictions2 = CNN.predict_classes(X_test_)
print("CNN's Confusion Matrix :\n ", confusion_matrix(Y_test,Predictions2.round()))
print("CNN's REPORT :\n ", classification_report(Y_test, Predictions2.round()))
loss_t, accuracy_t = CNN.evaluate(X_train_, Y_train, verbose=False)
print("Training Accuracy: {:.4f}".format(accuracy_t))
loss_s, accuracy_s = CNN.evaluate(X_test_, Y_test, verbose=False)
print("Testing Accuracy:  {:.4f}".format(accuracy_s))
plot_history(history)
Prediction_pip("#ECOWAS has formally endorsed Dr. @NOIweala for the position of Director-General of the World Trade Organisation for the period of 2021-2025. Also called on other African and Non African countries to do same. #COVID19  @NigeriaGov @DigiCommsNG @ecowas_cedeao @_AfricanUnion @wto", "CNN")

############
""" RNN """
############
np.random.seed(7)
RNN = Sequential()
RNN.add(Embedding(vocab_size, embedding_dim, input_length=maxlen))
RNN.add(Bidirectional(LSTM(100)))
RNN.add(Dropout(0.2))
RNN.add(Dense(1, activation='sigmoid'))
Optmizer = Adam(
    learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-07, amsgrad=True
)
RNN.compile(loss='binary_crossentropy', optimizer=Optmizer, metrics=['accuracy'])
print(RNN.summary())
model_ = RNN.fit(X_train_, Y_train, validation_data=(X_test_, Y_test), epochs=40, batch_size=10)
RNN.save("RNN")
Predictions3 = RNN.predict_classes(X_test_)
print("RNN's Confusion Matrix :\n ", confusion_matrix(Y_test,Predictions3.round()))
print("RNN's REPORT :\n ", classification_report(Y_test, Predictions3.round()))
loss_tr, accuracy_tr = RNN.evaluate(X_train_, Y_train, verbose=False)
print("Training Accuracy: {:.2f}".format(accuracy_tr * 100))
loss_ts, accuracy_ts = RNN.evaluate(X_test_, Y_test, verbose=False)
print("Testing Accuracy:  {:.2f}".format(accuracy_ts * 100))
plot_history(model_)
import time
time.sleep(15)
Prediction_pip("Barak Ubama is the prisident of United States", "RNN")
#########################
""" Model Comparison """
########################
Models = ['SVM', 'CNN', 'RNN']
Accuracies_tr = [tr_accuracy, accuracy_t, accuracy_tr]
plt.bar(Models, Accuracies_tr)
plt.xlabel('Models')
plt.ylabel('Accuracy')
plt.title('Training Accuracy')
plt.show()

Accuracies_ts = [ts_accuracy, accuracy_s, accuracy_ts]
plt.bar(Models, Accuracies_ts)
plt.title('Testing Accuracy')
plt.xlabel('Models')
plt.ylabel('Accuracy')
plt.show()

# ####################
# """ Ensembling """
# ####################
# # get a stacking ensemble of models
# def get_stacking():
#     # define the base models
#     level0 = list()
#     level0.append(('svm', svm.SVC()))
#     level0.append(("RNN", RNN))
#     level0.append(("RNN", CNN))
#     # define meta learner model
#     level1 = svm.SVC()
#     # define the stacking ensemble
#     model = StackingClassifier(estimators=level0, final_estimator=level1, cv=5)
#     return model

# # get a list of models to evaluate
# def get_models():
#     models = dict()
#     models['svm'] = svm.SVC()
#     models['stacking'] = get_stacking()
#     return models
 
# # evaluate a give model using cross-validation
# def evaluate_model(model):
#     cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
#     scores = cross_val_score(model, X, Y, scoring='accuracy', cv=cv, n_jobs=-1, error_score='raise')
#     return scores

# # get the models to evaluate
# models = get_models()
# # evaluate the models and store results
# results, names = list(), list()
# for name, model in models.items():
# 	scores = evaluate_model(model)
# 	results.append(scores)
# 	names.append(name)
# 	print('>%s %.3f (%.3f)' % (name, np.mean(scores), np.std(scores)))
# # plot model performance for comparison
# plt.boxplot(results, labels=names, showmeans=True)
# plt.show()

# ###################
# """ Deployment """
# ###################
# from flask import Flask
# from flask_restful import reqparse, abort, Api, Resource
