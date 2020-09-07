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
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report


def set_data_train():
    '''
    Set the train data.

    Returns
    ----------
    train : pandas.DataFrame
        a data frame of the train data
    '''

    #Choose data train file and store it in train data frame
    train = pd.read_csv('./data_train/result.csv', error_bad_lines=False, index_col=False, dtype='unicode')
    
    #Remove nan value
    train.dropna(subset = ["sentimen"], inplace=True)

    return train

def set_data_test():
    '''
    Set the test data.

    Returns
    ----------
    test : pandas.DataFrame
        a data frame of the test data
    '''

    #Take all the file names from folder data_test
    mypath = "./data_test/"
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

def set_stopwords():
    '''
    Set the list of stopwords.

    Returns
    ----------
    list_stopwords : list
        a list of stopwords in indonesian
    '''

    #Define stopwords in indonesian (Using nltk)
    list_stopword =  set(stopwords.words('indonesian'))

    list_stopwords = []

    #In the next step, the stopwords must be a list so it converted from set to list
    for x in list_stopword : 
        list_stopwords.append(x)

    return list_stopwords

def remove_stopwords(text):
    '''
    Remove stopword(s) e.g. "yang", "di", "ke", etc.

    Args
    ----------
    text : string
        a string that will be processed to remove the stopword(s)

    Returns
    ----------
    text : string
        a string that already not contains the stopword(s)
    '''

    #Define stopwords in indonesian (Using nltk)
    list_stopwords = set_stopwords()

    removed = []
    text = text.translate(str.maketrans('','',string.punctuation)).lower()
    tokens = word_tokenize(text)
    for t in tokens:
        if t not in list_stopwords:
            removed.append(t)
            
    text = " ".join(removed)

    return text

def set_stremmer():
    '''
    Set the stemmer (using sastrawi).

    Returns
    ----------
    stemmer : Sastrawi.Stemmer.CachedStemmer.CachedStemmer
        a stemmer to use for stemming the data
    '''

    #Create stemmer for stemming bahasa indonesia using sastrawi    
    factory = StemmerFactory()
    stemmer = factory.create_stemmer()

    return stemmer

def stemming_sentence(text):
    '''
    Stemming each word of the text.

    Args
    ----------
    text : string
        a string that will be processed to stemming

    Returns
    ----------
    text : string
        a string that already stemmed
    '''

    #Get the stemmer
    stemmer = set_stremmer()

    text = stemmer.stem(text)
    return text

def check_train_tidy_tweets(train):
    '''
    Check is the train data has a column that named 'Tidy_Tweets'.
    This column contains the tweet in train data that already passed the text preprocessing.

    Args
    ----------
    train : pandas.DataFrame
        a data frame of the train data
    Returns
    ----------
    boolean
        a boolean is there a 'Tidy_Tweets' column

    '''
    #if the first time this module was run and the train data not yet passed the text preprocessing
    if 'Tidy_Tweets' in train.columns:
        return True
    else :
        return False

def train_text_preprocessing(train):
    '''
    Text preprocessing (remove handler, remove all character(s) except a-z/A-Z, remove stopwords, stemming) the train data.
    This function only run once in case the training data not yet passed the text preprocessing (doesn't have a Tidy_Tweets column).

    Args
    ----------
    train : pandas.DataFrame
        a data frame of the train data

    Returns
    ----------
    train : pandas.DataFrame
        a data frame of the train data that already passed the text preprocessing
    '''

    #Text Preprocessing data train
    train['Tidy_Tweets'] = np.vectorize(remove_pattern)(train['Tweet'], "@[\rw]*")
    train['Tidy_Tweets'] = train['Tidy_Tweets'].str.replace("[^a-zA-Z#]", " ")
    train['Tidy_Tweets'] = np.vectorize(remove_stopwords)(train['Tidy_Tweets'])
    train['Tidy_Tweets'] = np.vectorize(stemming_sentence)(train['Tidy_Tweets'])

    return train

def test_text_preprocessing(test):
    '''
    Text preprocessing (remove handler, remove all character(s) except a-z/A-Z, remove stopwords, stemming) the test data.

    Args
    ----------
    test : pandas.DataFrame
        a data frame of the test data

    Returns
    ----------
    test : pandas.DataFrame
        a data frame of the test data that already passed the text preprocessing
    '''

    test = test.rename(columns = {'tweet': 'Tweet'}, inplace = False)

    #Text Preprocessing data test
    test['Tidy_Tweets'] = np.vectorize(remove_pattern)(test['Tweet'], "@[\rw]*")
    test['Tidy_Tweets'] = test['Tidy_Tweets'].str.replace("[^a-zA-Z#]", " ")
    test['Tidy_Tweets'] = np.vectorize(remove_stopwords)(test['Tidy_Tweets'])
    test['Tidy_Tweets'] = np.vectorize(stemming_sentence)(test['Tidy_Tweets'])

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

def word_embedding(train, combine):
    '''
    Word embedding with TF-IDF method.

    Args
    ----------
    train : pandas.DataFrame
        a data frame of the train data
    combine : pandas.DataFrame
        a data frame of the data that the result of a combine from data train and data test

    Returns
    ----------
    tfidf_matrix : scipy.sparse.csr.csr_matrix
        a csr_matrix of the Tidy_Tweets column from combine data
    train_tfidf_matrix : scipy.sparse.csr.csr_matrix
        a csr_matrix of the Tidy_Tweets column from train data
    '''

    #Word Embedding (TF IDF)
    list_stopwords = set_stopwords()

    tfidf=TfidfVectorizer(max_df=0.90, min_df=2,max_features=1000,stop_words=frozenset(list_stopwords))
    # tfidf=TfidfVectorizer(max_df=0.90, min_df=2,max_features=1000,stop_words=frozenset(stopwords))

    tfidf_matrix=tfidf.fit_transform(combine['Tidy_Tweets'].values.astype('U'))

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
        a csr_matrix of the Tidy_Tweets column from train data

    Returns
    ----------
    x_train_tfidf : scipy.sparse.csr.csr_matrix
        a csr_matrix that the value is randomly filled from train_tfidf_matrix 
    x_valid_tfidf : scipy.sparse.csr.csr_matrix
        a csr_matrix that the value is randomly filled from train_tfidf_matrix
    y_train_tfidf : scipy.sparse.csr.csr_matrix
        a csr_matrix that the value is randomly filled from sentimen column in data train
    y_valid_tfidf : scipy.sparse.csr.csr_matrix
        a csr_matrix that the value is randomly filled from sentimen column in data train
    '''

    #Split the training data
    x_train_tfidf, x_valid_tfidf, y_train_tfidf, y_valid_tfidf = train_test_split(train_tfidf_matrix,train['sentimen'],test_size=0.3,random_state=17)

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
        a csr_matrix that the value is randomly filled from sentimen column in data train
    y_valid_tfidf : scipy.sparse.csr.csr_matrix
        a csr_matrix that the value is randomly filled from sentimen column in data train

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

def set_predict_with_model(train, test, tfidf_matrix, Log_Reg):
    '''
    Predict the result using Logistic Regression model.

    Args
    ----------
    train : pandas.DataFrame
        a data frame of the train data
    test : pandas.DataFrame
        a data frame of the testing data
    tfidf_matrix : scipy.sparse.csr.csr_matrix
        a csr_matrix of the Tidy_Tweets column from combined data
    Log_Reg : sklearn.linear_model._logistic.LogisticRegression
        a logistic regression model to use for predict the result
    '''

    test_tfidf = tfidf_matrix[len(train):]

    test['sentimen'] = Log_Reg.predict(test_tfidf)

    combine = train.append(test,ignore_index=True,sort=True)

    store_to_csv(test, combine)

def store_to_csv(test, combine):
    '''
    Store the result of the test data and combined data with sentiment to local csv

    Args
    ----------
    test : pandas.DataFrame
        a data frame of the test data
    combine : pandas.DataFrame
        a data frame of the data that the result of a combine from data train and data test
    '''

    #Store all combine data to csv
    combine_submission = combine[['Tweet', 'sentimen', 'Tidy_Tweets']]
    combine_submission.to_csv('./data_train/result.csv', index=False)

    # test = test.drop('Tidy_Tweets', axis=1)
    test = test.rename(columns = {'Tweet': 'tweet'}, inplace = False)

    #Store the test data result (with sentimen) to csv
    # submission = test
    test.to_csv('./data_train/result_ikm.csv', index=False)

def show_accuracy(x_valid_tfidf, y_valid_tfidf, Log_Reg):
    '''
    Show the accuracy of the Logistic Regression model

    Args
    ----------
    x_valid_tfidf : scipy.sparse.csr.csr_matrix
        a csr_matrix that the value is randomly filled from train_tfidf_matrix
    y_valid_tfidf : scipy.sparse.csr.csr_matrix
        a csr_matrix that the value is randomly filled from sentimen column in data train
    Log_Reg : sklearn.linear_model._logistic.LogisticRegression
        a logistic regression model to use for predict the result
    '''
    y_pred = Log_Reg.predict(x_valid_tfidf)
    print(classification_report(y_valid_tfidf, y_pred))
    
def main():
    train = set_data_train()
    test = set_data_test()
    
    if(check_train_tidy_tweets(train) == False):
        train = train_text_preprocessing(test)
    else : 
        print("no train text prepocessing")

    test = test_text_preprocessing(test)

    combine = combine_data(train, test)

    train_tfidf_matrix = word_embedding(train, combine)

    x_train_tfidf, x_valid_tfidf, y_train_tfidf, y_valid_tfidf = split_data(train, train_tfidf_matrix[1])

    Log_Reg = logistic_regression_model(x_train_tfidf, x_valid_tfidf, y_train_tfidf, y_valid_tfidf)
    set_predict_with_model(train, test, train_tfidf_matrix[0], Log_Reg)

    show_accuracy(x_valid_tfidf, y_valid_tfidf, Log_Reg)

    # print(y_train_tfidf)
    # print(remove_stopwords.__doc__)


if __name__ == "__main__":
    main()