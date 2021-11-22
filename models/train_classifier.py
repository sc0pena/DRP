# import libraries and download NLTK
import sys
import nltk
import torch
import pickle
import sklearn
nltk.download(['wordnet','punkt','stopwords'])
import pandas as pd
import numpy as np
import scipy as sp
from sqlalchemy import create_engine
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from sklearn.datasets import make_multilabel_classification
from sklearn.multioutput import MultiOutputClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from scipy.sparse import csr_matrix
from sklearn.utils.multiclass import type_of_target
from sklearn.pipeline import Pipeline
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.model_selection import GridSearchCV

#Phase 4: Modeling
###############################################################################
###############################################################################


def load_data(database_filepath):
    '''function to load clean data from database; creates X and Y variables
    for modeling.
    
    Function also creates list of 36 categories for classification report 
    
    '''
    engine = create_engine('sqlite:///' + database_filepath)
    df = pd.read_sql_table('etl_pipeline', con=engine)
    #classification ML model:
    X = df.message #messages for X input, REMOVED .values 11/14/21 to 'elbow'
    Y = df.drop(['id','message','original','genre'],axis=1) #y = classification categories
    Y = Y.values
    category_names = df.drop(['id','message','original','genre'],axis=1)
    category_names = list(category_names.columns)
    return X, Y, category_names
    
    

#tokenize function 
def tokenize(text):   
    '''function to tokenize messages without using stop words
    - tokenize words
    - lemmatize words
    - lowercase words
    - remove whitespace
    - append clean tokens
    ''' 
    #var for word tokenization
    tokens = word_tokenize(text)
    #instantiate lemmatizer (used to split into words and convert words to base forms)
    lemmatizer = WordNetLemmatizer()

    # iterate through tokenized words and append tokenized and lemmatized words to clean_tokens var
    clean_tokens = []
    clean_tok_nostop = []
    for tok in tokens:
        
        # lemmatize, normalize/put into lower case + remove leading/trailing white space
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens


def build_model(X, Y, X_train, X_test, Y_train, Y_test):
    '''function to:
    - pipeline countvectorizer and tfidf transformers, NOT including 
      optimized parameters.....   
      
    - pipeline classify with multinomial naive bayes
    - identify range of parameters for gridsearch analysis, print gridsearch dict
      results
    '''
    
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)), 
        ('tfidf', TfidfTransformer()), 
        ('clf', MultiOutputClassifier(MultinomialNB()))
    ])
    
    #identify best parameters for gridsearch. print gridsearch results  
    parameters = { 
        'vect__ngram_range': ((1, 1), (1, 2)),
        'vect__max_df': (0.5, 0.75, 1.0),
        'vect__max_features': (None, 5000, 10000),
        'tfidf__use_idf': (True, False)
    }

    cv = GridSearchCV(pipeline, param_grid=parameters)

    #model prints best parameters...via testing in a Jupyter notebook, best params
    #(NOT IMPLEMENTED IN THIS MODEL) Were: {'tfidf__use_idf': True, 
    #'vect__max_df': 0.5, 'vect__max_features': 5000, 'vect__ngram_range': (1, 2)}
    return cv


#Phase 5: Evaluation (and some re-modeling to improve scores)
###############################################################################
###############################################################################
def evaluate_model(y_pred, Y_test, category_names):
    '''function to:
    - instantiate classification report for analysis
    '''
    #instantiate classification_report
    c_report =  classification_report(y_pred, Y_test, target_names=category_names)
    
    return c_report
    
def save_model(model, model_filepath):
    '''function to:
    - save model to pickle file
    '''
    # https://medium.com/@maziarizadi/pickle-your-model-in-python-2bbe7dba2bbb
    pickle.dump(model, open(model_filepath,'wb')) 


def main():
    '''function to take in 2 args (db path and pickle file path) at command prompt, 
    print and run each function in the module to run, fit, predict,
    print best parameters in terminal (project rubic states: "GridSearchCV is used 
    to FIND the best parameters for the model."
    Also perform eval, and  save model 
    '''
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model(X, Y, X_train, X_test, Y_train, Y_test)
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        #predict on model
        y_pred = model.predict(X_test)
        
        #print best parameters from gridsearch
        print('Best Parameters using Gridsearch...', model.best_params_)
        
        
        print('Evaluating model...')
        evaluate_model(y_pred, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()