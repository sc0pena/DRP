# import libraries
import pandas as pd
import numpy as np
from sqlalchemy import create_engine
import sys

#Phase 1: Business Understanding:

#Supervised learning model that takes in messages following a disaster and properly
#filters + classifies messages to relevant organizational areas for relief.

#Create an ETL pipeline to handle .csv data (this module), an ML pipeline 
#(train_classifier.py module), then use these modules to create a front end web app 
#where an emergency worker can put a new message and get classification results in several categories. 
#This app will display visualizations of the data (uses run.py module).

#END OF BUSINESS UNDERSTANDING
##############################################################################
##############################################################################
#Phase 2: Data understanding:

def load_data(messages_filepath, categories_filepath):

    '''function to read in CSV files for messages and message categories and
    merge datasets
    '''
    #1.read in CSV files for messages and message categories
    messages = pd.read_csv(messages_filepath)
    #messages.head()
    categories = pd.read_csv(categories_filepath) 
    #categories.head()
    #############################################################################################################################
    #2. merge datasets
    df = messages.merge(categories, how = 'outer', on = 'id')
    #df.head()
    #############################################################################################################################
    return df
def clean_data(df):
    '''function to clean .csv data 
    - split into categories
    - create list of column names 
    - renames columns to this new list
    '''
    # 3. Split categories into separate category columns.
    # create a dataframe of the 36 individual category columns
    categories = df.categories.str.split(';', expand=True)
    #categories.head()
    # select the first row of the categories dataframe
    row = categories[0]

    # use row (above) to extract a list of new column names for categories.
    # one way is to apply a lambda function that takes everything 
    # up to the second to last character of each string with slicing
    # apply lamba funct to categories dataframe, first row, on elements 0 through -2 (removes dash + numb value after)
    category_colnames = categories[0:1].apply(lambda row: row.str[:-2]) 
    #category_colnames
    
    #convert category_list df row to list
    #code example from : https://www.kite.com/python/answers/how-to-convert-the-rows-of-a-pandas-dataframe-to-lists-in-python
    #added .flatten() to put into 1 dimension
    cat_col_list = category_colnames.values.flatten().tolist() 

    # rename columns of categories df to  list of category column values
    categories.columns = cat_col_list
    #categories.head()
  
    ##############################################################################################################################
    #4. Convert category values to just numbers 0 or 1.
    for column in categories:
        # set each value to be the last character of the string; 
        #1. slice last char of each column:
        #example https://www.datasciencemadesimple.com/extract-last-n-characters-from-right-of-the-column-in-pandas-python/
        categories[column] = categories[column].str[-1:] 
        
        # convert column from string to numeric
        categories[column] = pd.to_numeric(categories[column])
    #categories.head()
    
    #pull unique values from each field for double-check. 
    #looks like the first column (results) some rows still have 2. will replace 2's with 1's
    [categories[col_name].unique() for col_name in categories.columns]
    
    #replace 2's with 1's
    categories["related"].replace({2: 1}, inplace=True)
    
    #make sure 2's don't exist
    [categories[col_name].unique() for col_name in categories.columns]
    
    ###############################################################################################################################
    #5. Replace categories column in df with new category columns.
    # drop the original categories column from `df`
    df.drop(columns = 'categories', inplace = True)
    #df.head()
    
    # concatenate the original dataframe with the new `categories` dataframe
    df = pd.concat([df, categories], axis = 1)
    #df.head()
    
    ###############################################################################################################################
    #6. Remove duplicates.
    # check number of duplicates
    np.sum(df.duplicated() == True)
    # drop duplicates
    df.drop_duplicates(inplace = True)
    # check number of duplicates
    np.sum(df.duplicated() == True)
    return df
    
    
def save_data(df, database_filename):
    '''function to save clean dataset to sqlite database
    '''
    engine = create_engine('sqlite:///' + database_filename)  
    df.to_sql('etl_pipeline', con=engine, index=False, if_exists='replace')  


def main():
    '''function to take in 4 args (filenames and db path) at command prompt, 
    print and run each function in the module to clean and save data to db 
    '''
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()