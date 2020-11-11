import re
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns
import string
import nltk
import warnings
from os import listdir
from os.path import isfile, join
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer 
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report


def set_data_train(language):
    '''
    Set the train data.

    Args
    ----------
    language : string
        a string of the language that choosen (english or indonesia)

    Returns
    ----------
    train : pandas.DataFrame
        a data frame of the train data
    '''

    #Choose data train file and store it in train data frame
    train = pd.read_csv('./data_train/train_' + language + '.csv', error_bad_lines=False, index_col=False, dtype='unicode')
    
    #Remove nan value
    train.dropna(subset = ["Sentiment"], inplace=True)

    return train

def set_data_test(language):
    '''
    Set the test data.

    Args
    ----------
    language : string
        a string of the language that choosen (english or indonesia)

    Returns
    ----------
    test : pandas.DataFrame
        a data frame of the test data
    '''

    #Take all the file names from folder data_test
    mypath = "./data_test/" + language + "/"
    onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]

    #Open the data test file by it names (which the name was already store in onlyfiles) and store it in test data frame
    i = 1
    for x in onlyfiles :
        temp = pd.read_csv(mypath + x)
        if(i==1) : 
            test = pd.read_csv(mypath + x)
        else :
            test = test.append(temp, ignore_index=True, sort=True)
        i = i+1
    
    #define the name of the column to test and rename it to Text
    column_name = 'tweet'
    test = test.rename(columns = {column_name: 'Text'}, inplace = False)
    
    return test

def remove_pattern(text,pattern):
    '''
    Remove Twitter Handler (e.g. @user).

    Args
    ----------
    text : string
        a string that will be processed to remove the pattern
    pattern : string
        a string of the pattern to be removed

    Returns
    ----------
    text : string
        a string that already not contains the pattern
    '''

    # re.findall() finds the pattern i.e @user and puts it in a list for further task
    r = re.findall(pattern,str(text))
    
    # re.sub() removes @user from the sentences in the dataset
    for i in r:
        text = re.sub(i,"",str(text))
    
    return text

def set_stopwords(language):
    '''
    Set the list of stopwords.

    Args
    ----------
    language : string
        a string of the language that choosen (english or indonesia)

    Returns
    ----------
    list_stopwords : list
        a list of stopwords in indonesian
    '''

    #Define stopwords in english (Using nltk)
    if language == 'indonesia' :
        language = 'indonesian'

    list_stopword =  set(stopwords.words(language))

    list_stopwords = []

    #In the next step, the stopwords must be a list so it converted from set to list
    for x in list_stopword : 
        list_stopwords.append(x)

    return list_stopwords

def remove_stopwords(text, language):
    '''
    Remove stopword(s) e.g. "yang", "di", "ke", etc.

    Args
    ----------
    text : string
        a string that will be processed to remove the stopword(s)
    language : string
        a string of the language that choosen (english or indonesia)

    Returns
    ----------
    text : string
        a string that already not contains the stopword(s)
    '''

    #Define stopwords in indonesian (Using nltk)
    list_stopwords = set_stopwords(language)

    removed = []
    text = text.translate(str.maketrans('','',string.punctuation)).lower()
    tokens = word_tokenize(text)
    for t in tokens:
        if t not in list_stopwords:
            removed.append(t)
            
    text = " ".join(removed)

    return text

def set_stremmer(language):
    '''
    Set the stemmer (using sastrawi).

    Args
    ----------
    language : string
        a string of the language that choosen (english or indonesia)

    Returns
    ----------
    stemmer : Sastrawi.Stemmer.CachedStemmer.CachedStemmer
        a stemmer to use for stemming the data
    '''

    #Check the language
    if language == 'english' :
        #Create stemmer for stemming english using nltk
        stemmer = PorterStemmer() 
    elif language == 'indonesia' :
        #Create stemmer for stemming bahasa indonesia using sastrawi
        factory = StemmerFactory()
        stemmer = factory.create_stemmer()

    return stemmer

def stemming_sentence(text, language):
    '''
    Stemming each word of the text.

    Args
    ----------
    text : string
        a string that will be processed to stemming
    language : string
        a string of the language that choosen (english or indonesia)

    Returns
    ----------
    text : string
        a string that already stemmed
    '''

    #Get the stemmer
    stemmer = set_stremmer(language)

    stemmed = []
    text = text.translate(str.maketrans('','',string.punctuation)).lower()
    tokens = word_tokenize(text)
    for t in tokens:
        stemmed.append(stemmer.stem(t))
            
    text = " ".join(stemmed)

    return text

def check_train_tidy_tweets(train):
    '''
    Check is the train data has a column that named 'Tidy_Text'.
    This column contains the tweet in train data that already passed the text preprocessing.

    Args
    ----------
    train : pandas.DataFrame
        a data frame of the train data
    Returns
    ----------
    boolean
        a boolean is there a 'Tidy_Text' column

    '''
    #if the first time this module was run and the train data not yet passed the text preprocessing
    if 'Tidy_Text' in train.columns:
        return True
    else :
        return False

def train_text_preprocessing(train, language):
    '''
    Text preprocessing (remove handler, remove all character(s) except a-z/A-Z, remove stopwords, stemming) the train data.
    This function only run once in case the training data not yet passed the text preprocessing (doesn't have a Tidy_Text column).

    Args
    ----------
    train : pandas.DataFrame
        a data frame of the train data
    language : string
        a string of the language that choosen (english or indonesia)

    Returns
    ----------
    train : pandas.DataFrame
        a data frame of the train data that already passed the text preprocessing
    '''

    #Text Preprocessing data train
    train['Tidy_Text'] = np.vectorize(remove_pattern)(train['Text'], "@[\\w]*")
    train['Tidy_Text'] = train['Tidy_Text'].str.replace("[^a-zA-Z#]", " ")
    train['Tidy_Text'] = np.vectorize(remove_stopwords)(train['Tidy_Text'], language)
    train['Tidy_Text'] = np.vectorize(stemming_sentence)(train['Tidy_Text'], language)

    return train

def test_text_preprocessing(test, language):
    '''
    Text preprocessing (remove handler, remove all character(s) except a-z/A-Z, remove stopwords, stemming) the test data.

    Args
    ----------
    test : pandas.DataFrame
        a data frame of the test data
    language : string
        a string of the language that choosen (english or indonesia)

    Returns
    ----------
    test : pandas.DataFrame
        a data frame of the test data that already passed the text preprocessing
    '''

    # test = test.rename(columns = {'tweet': 'Text'}, inplace = False)

    #Text Preprocessing data test
    test['Tidy_Text'] = np.vectorize(remove_pattern)(test['Text'], "@[\rw]*")
    test['Tidy_Text'] = test['Tidy_Text'].str.replace("[^a-zA-Z#]", " ")
    test['Tidy_Text'] = np.vectorize(remove_stopwords)(test['Tidy_Text'], language)
    test['Tidy_Text'] = np.vectorize(stemming_sentence)(test['Tidy_Text'], language)

    return test

def combine_data(train, test):
    '''
    Combine the train data and test data.

    Args
    ----------
    train : pandas.DataFrame
        a data frame of the train data
    test : pandas.DataFrame
        a data frame of the test data

    Returns
    ----------
    combine : pandas.DataFrame
        a data frame of the data that the result of a combine from data train and data test
    '''
    #Combine the data (data train and data test)
    combine = train.append(test,ignore_index=True,sort=True)

    return combine

def word_embedding(train, combine, language):
    '''
    Word embedding with TF-IDF method.

    Args
    ----------
    train : pandas.DataFrame
        a data frame of the train data
    combine : pandas.DataFrame
        a data frame of the data that the result of a combine from data train and data test
    language : string
        a string of the language that choosen (english or indonesia)

    Returns
    ----------
    tfidf_matrix : scipy.sparse.csr.csr_matrix
        a csr_matrix of the Tidy_Text column from combine data
    train_tfidf_matrix : scipy.sparse.csr.csr_matrix
        a csr_matrix of the Tidy_Text column from train data
    '''

    #Word Embedding (TF IDF)
    list_stopwords = set_stopwords(language)

    tfidf=TfidfVectorizer(max_df=0.90, min_df=2,max_features=1000,stop_words=frozenset(list_stopwords))
    # tfidf=TfidfVectorizer(max_df=0.90, min_df=2,max_features=1000,stop_words=frozenset(stopwords))

    tfidf_matrix=tfidf.fit_transform(combine['Tidy_Text'].values.astype('U'))

    # df_tfidf = pd.DataFrame(tfidf_matrix.todense())

    train_tfidf_matrix = tfidf_matrix[:len(train)]

    return (tfidf_matrix, train_tfidf_matrix)

def split_data(train, train_tfidf_matrix):
    '''
    Split the train data.

    Args
    ----------
    train : pandas.DataFrame
        a data frame of the train data
    train_tfidf_matrix : csr_matrix
        a csr_matrix of the Tidy_Text column from train data

    Returns
    ----------
    x_train_tfidf : scipy.sparse.csr.csr_matrix
        a csr_matrix that the value is randomly filled from train_tfidf_matrix 
    x_valid_tfidf : scipy.sparse.csr.csr_matrix
        a csr_matrix that the value is randomly filled from train_tfidf_matrix
    y_train_tfidf : scipy.sparse.csr.csr_matrix
        a csr_matrix that the value is randomly filled from sentiment column in data train
    y_valid_tfidf : scipy.sparse.csr.csr_matrix
        a csr_matrix that the value is randomly filled from sentiment column in data train
    '''

    #Split the training data
    x_train_tfidf, x_valid_tfidf, y_train_tfidf, y_valid_tfidf = train_test_split(train_tfidf_matrix,train['Sentiment'],test_size=0.3,random_state=17)

    return (x_train_tfidf, x_valid_tfidf, y_train_tfidf, y_valid_tfidf)

def logistic_regression_model(x_train_tfidf, x_valid_tfidf, y_train_tfidf, y_valid_tfidf):
    '''
    Set the Logistic Regression model.

    Args
    ----------
    x_train_tfidf : scipy.sparse.csr.csr_matrix
        a csr_matrix that the value is randomly filled from train_tfidf_matrix 
    x_valid_tfidf : scipy.sparse.csr.csr_matrix
        a csr_matrix that the value is randomly filled from train_tfidf_matrix
    y_train_tfidf : scipy.sparse.csr.csr_matrix
        a csr_matrix that the value is randomly filled from sentiment column in data train
    y_valid_tfidf : scipy.sparse.csr.csr_matrix
        a csr_matrix that the value is randomly filled from sentiment column in data train

    Returns
    ----------
    Log_Reg : sklearn.linear_model._logistic.LogisticRegression
        a logistic regression model to use for predict the result
    '''

    #Initiate Logistic Regression model
    Log_Reg = LogisticRegression(random_state=0,solver='lbfgs', max_iter=1000)

    #Fit the data for Logistic Regression model
    Log_Reg = Log_Reg.fit(x_train_tfidf,y_train_tfidf)

    return Log_Reg

def set_predict_with_model(train, test, tfidf_matrix, Log_Reg, language):
    '''
    Predict the result using Logistic Regression model.

    Args
    ----------
    train : pandas.DataFrame
        a data frame of the train data
    test : pandas.DataFrame
        a data frame of the testing data
    tfidf_matrix : scipy.sparse.csr.csr_matrix
        a csr_matrix of the Tidy_Text column from combined data
    Log_Reg : sklearn.linear_model._logistic.LogisticRegression
        a logistic regression model to use for predict the result
    language : string
        a string of the language that choosen (english or indonesia)
    '''

    test_tfidf = tfidf_matrix[len(train):]

    test['Sentiment'] = Log_Reg.predict(test_tfidf)

    combine = train.append(test,ignore_index=True,sort=True)

    store_to_csv(test, combine, language)

def store_to_csv(test, combine, language):
    '''
    Store the result of the test data and combined data with sentiment to local csv

    Args
    ----------
    test : pandas.DataFrame
        a data frame of the test data
    combine : pandas.DataFrame
        a data frame of the data that the result of a combine from data train and data test
    language : string
        a string of the language that choosen (english or indonesia)
    '''

    #Store all combine data to csv
    combine_submission = combine[['Text', 'Sentiment', 'Tidy_Text']]
    combine_submission.to_csv('./data_train/train_' + language + '.csv', index=False)

    test = test.drop('Tidy_Text', axis=1)

    #Store the test data result (with sentiment) to csv
    # submission = test
    test.to_csv('./data_result/result_' + language + '.csv', index=False)

def show_accuracy(x_valid_tfidf, y_valid_tfidf, Log_Reg):
    '''
    Show the accuracy of the Logistic Regression model

    Args
    ----------
    x_valid_tfidf : scipy.sparse.csr.csr_matrix
        a csr_matrix that the value is randomly filled from train_tfidf_matrix
    y_valid_tfidf : scipy.sparse.csr.csr_matrix
        a csr_matrix that the value is randomly filled from sentiment column in data train
    Log_Reg : sklearn.linear_model._logistic.LogisticRegression
        a logistic regression model to use for predict the result
    '''
    y_pred = Log_Reg.predict(x_valid_tfidf)
    print(classification_report(y_valid_tfidf, y_pred))