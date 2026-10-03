import pandas as pd
import numpy as np
from string import punctuation
from nltk.corpus import stopwords
import nltk
from nltk.stem import WordNetLemmatizer
import pickle
from nltk.stem import PorterStemmer
import os
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer #loading tfidf vector
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
#=================flask code starts here
from flask import Flask, render_template, request, redirect, url_for, session,send_from_directory
import pymysql

app = Flask(__name__)
app.secret_key = 'welcome'

dataset = pd.read_csv("Dataset/Cyber-Threat-Intelligence.csv")
#define object to remove stop words and other text processing
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()
ps = PorterStemmer()


#define function to clean text by removing stop words and other special symbols
def cleanText(doc):
    tokens = doc.split()
    table = str.maketrans('', '', punctuation)
    tokens = [w.translate(table) for w in tokens]
    tokens = [word for word in tokens if word.isalpha()]
    tokens = [w for w in tokens if not w in stop_words]
    tokens = [word for word in tokens if len(word) > 1]
    tokens = [ps.stem(token) for token in tokens]
    tokens = [lemmatizer.lemmatize(token) for token in tokens]
    tokens = ' '.join(tokens)
    return tokens

text = dataset['text'].ravel()
threats = dataset['label_2'].ravel()
labels = np.unique(dataset['label_2'])
X = np.load("model/X.npy")
binary = np.load("model/binary.npy")
multi = np.load("model/multi.npy")
bow = TfidfVectorizer(stop_words=stop_words, use_idf=False, norm=None)
bow_X = bow.fit_transform(X).toarray()
features = bow.get_feature_names()
multi_X_train, multi_X_test, multi_y_train, multi_y_test = train_test_split(bow_X, multi, test_size=0.2)
data = np.load("model/data.npy", allow_pickle=True)
binary_X_train, binary_X_test, binary_y_train, binary_y_test, multi_X_train, multi_X_test, multi_y_train, multi_y_test = data
#select features uisng CHI2 algorithm
selector2 = SelectKBest(score_func=chi2, k=650) 
multi_X_train1 = selector2.fit_transform(multi_X_train, multi_y_train)
multi_X_test1 = selector2.transform(multi_X_test)
dt = DecisionTreeClassifier()
er = ExtraTreesClassifier()
rf =RandomForestClassifier()
#combining multiple algorithms to form a hybrid model
estimators = [('ert', er), ('dt', dt), ('rf', rf)]
multi_hybrid_model = VotingClassifier(estimators = estimators, voting='soft')
multi_hybrid_model.fit(multi_X_train1, multi_y_train)
predict = multi_hybrid_model.predict(multi_X_test1)

@app.route('/Predict', methods=['GET', 'POST'])
def predictView():
    return render_template('Predict.html', msg='')

@app.route('/index', methods=['GET', 'POST'])
def index():
    return render_template('index.html', msg='')

@app.route('/UserLogin', methods=['GET', 'POST'])
def UserLogin():
    return render_template('UserLogin.html', msg='')

@app.route('/UserLoginAction', methods=['GET', 'POST'])
def UserLoginAction():
    if request.method == 'POST' and 't1' in request.form and 't2' in request.form:
        user = request.form['t1']
        password = request.form['t2']
        if user == "admin" and password == "admin":
            return render_template('UserScreen.html', msg="<font size=3 color=blue>Welcome "+user+"</font>")
        else:
            return render_template('UserLogin.html', msg="<font size=3 color=red>Invalid login details</font>")

@app.route('/Logout')
def Logout():
    return render_template('index.html', msg='')

@app.route('/PredictAction', methods=['GET', 'POST'])
def PredictAction():
    if request.method == 'POST':
        global selector2, labels, multi_hybrid_model, bow
        testData = pd.read_csv("Dataset/testData.csv",encoding='ISO-8859-1')#reading test data
        testData = testData.values
        clean = []
        for i in range(len(testData)):
            data = str(testData[i,0])
            data = data.strip("\n").strip().lower()
            if len(data) > 0:
                data = cleanText(data)#cleaning text values
                clean.append(data)
        vector = bow.transform(clean).toarray()  #getting binary bow values   
        vector = selector2.transform(vector)
        predict = multi_hybrid_model.predict(vector)#applying extension hybrid algorithm to predict cyber threat
        output = '<table border=1 align=center width=100%><tr>'
        output += '<th><font size="3" color="black">Test Data</font></th>'
        output += '<th><font size="3" color="blue">Predicted Cyber Threat</font></th></tr>'
        for i in range(len(predict)):
            output += "<tr>"
            output += '<td><font size="3" color="black">'+str(testData[i,0])+'</font></td>'
            output += '<td><font size="4" color="blue">'+labels[predict[i]]+'</font></td>'                 
        output += "</table><br/><br/><br/><br/>"             
        return render_template('UserScreen.html', msg=output)

if __name__ == '__main__':
    app.run()    
