import os
import pandas as pd
import codecs

def create_email_list(folder_path):
    email_list = []
    folder = os.listdir(folder_path)
    for txt in folder:
        file_name = fr'{folder_path}/{txt}'
        with codecs.open(file_name, 'r', encoding = 'utf-8',errors = 'ignore')as f:
            email = f.read().replace('\r\n', ' ')
            email_list.append(email)
    return email_list
spam_list = create_email_list(r"C:\Users\dhruv\Documents\DS-class\classification\Enron\enron1\spam")
ham_list = create_email_list(r"C:\Users\dhruv\Documents\DS-class\classification\Enron\enron1\ham")
df_spam = pd.DataFrame(spam_list, columns=['mail'])
df_ham = pd.DataFrame(ham_list, columns=['mail'])
df_spam['label']=1
df_ham['label']=0    
base = [df_spam, df_ham]
df = pd.concat(base)
df = df.sample(frac=1)
#aaaaaaaaaaaaaaa the boring work now
import re 
import string
import pandas as pd
def clean_text_round1(text):
    #makes lowercase, removes brackets, removes punct
    text = text.lower()
    text = re.sub('\[.*?\]', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub('\w*\d\w*', '', text)
    text = re.sub( u"(\ud83d[\ude00-\ude4f])|", '', text)
    text = re.sub('"\ufeff"', '', text)
    
    return text
r1 = lambda x: clean_text_round1(x)
data_clean = pd.DataFrame(df.mail.apply(r1))

# lets go againnnn aaaaaaaaaaaaaa
def clean_text_round2(text):
    #other punctuation, and non-sensical text missed
    text = re.sub('[‘’“”…]', '', text)
    text = re.sub('\n', '', text)
    return text
r2 = lambda x: clean_text_round2(x)
data_clean = pd.DataFrame(data_clean.mail.apply(r2))
def remove_non_ascii(text): # removes all non-ascii characters
    return ''.join(i for i in text if ord(i)<128)
data_clean.mail = data_clean.mail.apply(remove_non_ascii)
split = pd.DataFrame(data_clean.mail.str.split(' '))
data = df.assign(mail=split['mail'])
from nltk.stem.snowball import SnowballStemmer
#i am specifying english for optimisation (?) 
stemmer = SnowballStemmer("english")
data['stemmed'] = data['mail'].apply(lambda x: [stemmer.stem(y) for y in x])
from nltk.corpus import stopwords 
stops = set(stopwords.words("english"))

def remove_stopwords(n):
    list = n['stemmed']
    useful = [w for w in list if not w in stops]
    return(useful)
data['prop'] = data.apply(remove_stopwords, axis =1)
data['STR'] = data.prop.apply(lambda x:', '.join([str(i) for i in x]))
del data["mail"]
del data["stemmed"]
import numpy as np

train_set, validate, test_set = \
              np.split(data.sample(frac=1, random_state=42), 
                       [int(.6*len(data)), int(.8*len(data))])
spam = train_set[train_set['label']==1]
ham = train_set[train_set['label']==0]
Z = []
for words in spam.prop:
    for val in words:
        Z.append(val)
BOW_S = []
for val in Z:
    if val not in BOW_S:
        BOW_S.append(val)
L = []
for words in ham.prop:
    for val in words:
        L.append(val)
BOW_H = []
for val in L:
    if val not in BOW_H:
        BOW_H.append(val)
from collections import Counter
negF = Counter(Z)
posF = Counter(L)        
def extract_features(tweet):
    n = 0
    p = 0
    for word in tweet:
        if word in negF.keys():
            n += negF[word]
        if word in posF.keys():
            p +=  posF[word]
    X = [1, p, n]
    return X
X = np.zeros((3103,3))
temp = []
for row in train_set.itertuples(index = True):
    
    lst = extract_features(getattr(row, "prop"))
    temp.append(lst)

X = np.array(temp)
temp = []
y = train_set["label"].to_numpy()
    
    
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
temp = []   
def sigmoid(x):
    return .5 * (1 + np.tanh(.5 * x))
class LR:
     
    def sigmoid(self,z):
        return .5 * (1 + np.tanh(.5 * z))
     
    def add_intercept(self, X):
        intercept = np.ones((X.shape[0], 1))
        return np.concatenate((intercept, X), axis=1)
     
    def loss_func(self,X, y, weights):                 
        m =len(X)                
        yhat = sigmoid(np.dot(X, weights))        
        predict_1 = y * np.log(yhat)
        predict_0 = (1 - y) * np.log(1 - yhat)        
        return -sum(predict_1 + predict_0) / m
     
    def fit(self,X,y,epochs=25,learning_rate=0.05):
        loss = []
         
        X = self.add_intercept(X)
         
        weights = np.random.rand(X.shape[1])
        N = len(X)
 
        for _ in range(epochs):
            z = np.dot(X,weights)
            y_hat = sigmoid(z)
            weights -= learning_rate * (X.T @ (y_hat-y))/N
            loss.append(self.loss_func(X,y,weights))
        self.weights = weights
        self.loss = loss
        print(weights)
         
    def predict(self, X): 
        X = self.add_intercept(X)
        z = np.dot(X, self.weights)
        #Binary result
        return [1 if i > 0.5 else 0 for i in self.sigmoid(z)]
 
 
lr = LR()
weights = lr.fit(X_train,y_train,epochs=1000000,learning_rate=0.0001  
predection = lr.predict(X_test)
len(predection)
print(predection)
pred = np.array(predection)
validation = []
for (e1 , e2) in zip(pred, y_test):
    if e1 == e2:
        validation.append(1)
    if e1 != e2:
        validation.append(0)
print(predection)
acc = sum(validation)/len(validation)
print(acc)
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        









