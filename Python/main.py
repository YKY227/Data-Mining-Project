# -*- coding: utf-8 -*-
"""
Created on Sat Apr 13 10:51:37 2019

@author: Wei Qin
"""

import nltk #library for NLP
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import string
import csv
import numpy as np
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn import linear_model
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from sklearn import model_selection
from sklearn.model_selection import train_test_split
from statistics import mean
from mpl_toolkits.mplot3d import axes3d,Axes3D

"""
run nltk.download_shell() at the terminal and download stopwords
"""
#nltk.download('stopwords')

comments = pd.read_csv('comments.csv',
                      names=['Comment','Label'])

comments.head() #show the first 5 data

#Exploratory Data Analysis
comments.describe()
comments.groupby('Label').describe()
comments.fillna(value='N/A',inplace=True)
comments['length'] = comments['Comment'].apply(len)
comments.head()

#plotting
comments['length'].plot(bins=50, kind='hist')
comments.length.describe()
comments[comments['length'] == 1161]['Comment'].iloc[0]
comments.hist(column='length',by='Label',bins=50,figsize=(12,4))


my_reader = csv.reader(open('comments.csv'))
positive = 0
negative = 0
neutral = 0
for record in my_reader:
    if record[1] == 'positive':
        positive += 1
    elif record[1] == 'negative':
        negative += 1
    elif record[1] == 'neutral':
        neutral += 1
print("Positve: ",positive)
print("Negative: ",negative)
print("Neutral: ",neutral)

#number_of_comments = [positive,negative,neutral]
#comments_type = ["Positive","Negative","Neutral"]
#plt.plot(comments_type, number_of_comments)
#plt.xlabel('Label Types')
#plt.ylabel('Number of Comments')

#objects = ("Positive","Negative","Neutral")
#y_pos = np.arange(len(objects))
#performance = [positive,negative,neutral]

#plt.barh(y_pos, performance, align='center', alpha=0.5)
#plt.yticks(y_pos, objects)
#plt.xlabel('Number of Comments')
#plt.ylabel('Label Type')

plt.show()

##Text processing
def text_process(comment):
    """
    Takes in a string of text, then performs the following:
    1. Remove all punctuation
    2. Remove all stopwords
    3. Returns a list of the cleaned text
    """
    # Check characters to see if they are in punctuation
    nopunc = [char for char in comment if char not in string.punctuation]

    # Join the characters again to form the string.
    nopunc = ''.join(nopunc)
    
    # Now just remove any stopwords
    return [word for word in nopunc.split() if word.lower() not in stopwords.words('english')]
    #if word.lower() not in stopwords.words('english')]

nb_results = []
mlr_results = []
svm_results = []
rf_results = []

i = 1
for i in range(1,11):
  #Training set and test set
  comment_train, comment_test, label_train, label_test =\
  train_test_split(comments['Comment'],comments['Label'],test_size=0.2)

  #Vectorizing text
  bow_transformer = CountVectorizer(analyzer = text_process).fit(comments["Comment"])
  print(len(bow_transformer.vocabulary_))
  bow_comment = bow_transformer.transform(comments['Comment'])
  bow_comment_train = bow_transformer.transform(comment_train)
  bow_comment_test = bow_transformer.transform(comment_test)

  tfidf_transformer = TfidfTransformer().fit(bow_comment)
  tfidf_comment_train = tfidf_transformer.transform(bow_comment_train)
  tfidf_comment_test = tfidf_transformer.transform(bow_comment_test)

  #Train model
  nb = MultinomialNB().fit(tfidf_comment_train, label_train)
  predictions_nb = nb.predict(tfidf_comment_test)
  nb_results.append(metrics.accuracy_score(label_test,predictions_nb))
  
  mlr = linear_model.LogisticRegression(multi_class='multinomial', solver='newton-cg').fit(tfidf_comment_train, label_train)
  predictions_mlr = mlr.predict(tfidf_comment_test)
  mlr_results.append(metrics.accuracy_score(label_test,predictions_mlr))
  
  svm = LinearSVC().fit(tfidf_comment_train, label_train)
  predictions_svm = svm.predict(tfidf_comment_test)
  svm_results.append(metrics.accuracy_score(label_test,predictions_svm))
 
  rf = RandomForestClassifier(n_estimators = 30)
  rf.fit(tfidf_comment_train, label_train)
  predictions_rf = rf.predict(tfidf_comment_test)
  rf_results.append(metrics.accuracy_score(label_test,predictions_rf))
  

print('Average accuracy for NB for 10 iterations:', mean(nb_results))
print('Average accuracy for MLR for 10 iterations:', mean(mlr_results))
print('Average accuracy for SVM for 10 iterations:', mean(svm_results))
print('Average accuracy for RF for 10 iterations:', mean(rf_results))

x = [
  'should not have voted for PH!',
  'it will be a good source of revenue',
  'more and more tax, please stop!',
  'why are u adding more burden to the people','Its okay, still can afford',
  'Ali, see this'
  ]

x_trans = bow_transformer.transform(x)
x_trans = tfidf_transformer.transform(x_trans)

print(nb.predict(x_trans))
print(mlr.predict(x_trans))
print(svm.predict(x_trans))
print(rf.predict(x_trans))

results =[]
results.append(nb_results)
results.append(mlr_results)
results.append(svm_results)
results.append(rf_results)

fig = plt.figure()
fig.suptitle('Algorithm Comparison')
ax = fig.add_subplot(111)
plt.boxplot(results)
ax.set_xticklabels(['NB','MLR','SVM','RF'])
plt.savefig('results.png')