import sys
import nltk
nltk.download(['punkt', 'wordnet'])
nltk.download('stopwords')
import pandas as pd
import numpy as np
from sqlalchemy import create_engine
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import re
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn.datasets import make_multilabel_classification
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, precision_recall_fscore_support


def load_data(messages_filepath, categories_filepath):
    """ 
    Function to import the data needed to train the classifier model.
    This function takes as imputs strings that denote where the .csv files that contain the messages and their corresponding categories.
    The output is a dataframe of these two .csvs merged together using the field 'id'
    """
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    df = messages.merge(categories, how='inner', on='id')
    return df

def clean_data(df):
    """ 
    Function to wrangle and clean the data contained in the dataframe df (output of load_data).
    
    This function takes as imput the name of the dataframe containing the raw message and categories data, which is then processed as follows:
        1. Observations in the categories column is split into it's constituent categories
        2. The name of each category extracted into a series of category names (category_colnames)
        3. The columns of the expanded categories data is then renamed to match the category names
        4. The data in the categories columns is then truncated to be only the final character of the string - a flag either 0 or 1. 
        5. This flag is then cast as a numeric data type.
        6. Anomalies are also dealt with such as any categories with a value that is neither 0 or 1. In this case the anomaly is replaced with the mode of the column
        7. The original categories column is dropped from the dataframe and the new, separated columns are joined on.
        
    The output is a dataframe of the messages, and the wrangled category flags.
    """
    categories = df['categories'].str.split(';',expand=True)
    row = categories.iloc[1,:]
    category_colnames = row.apply(lambda x: x[0:-2])
    categories.columns = category_colnames
    
    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].apply(lambda x: x[-1])
        # convert column from string to numeric
        categories[column] = pd.to_numeric(categories[column])

    # If there are more than 0 or 1 in a category - replace with mode                      
    for index, column in enumerate(categories):
        if max(categories[column]) > 1:
            print(column, min(categories[column]), max(categories[column]))
            to_replace = list(categories[categories['related']>1]['related'].unique())
            replacement = categories[column].mode()[0]
            print(to_replace, 'to be replaced with', replacement)
            categories[column].replace(to_replace,replacement, inplace=True)
            print(column, min(categories[column]), max(categories[column]))
    
    df.drop('categories', axis=1, inplace=True)
    df_new = pd.concat([df, categories], axis=1)
    df_new.drop_duplicates(keep = 'first', inplace = True)
    
    return df_new
    

def save_data(df, database_filename):
    """
    Function to create an SQLite database and save the prepared data to it.
    Inputs are the dataframe to save (output of clean_data(df)) and the name of the database.
    Function has no output but saves the dataframe to the database.
    """
    engine = create_engine(f'sqlite:///{database_filename}') #.db
    sql = 'DROP TABLE IF EXISTS cleaned_messages_categories;'
    result = engine.execute(sql)
    df.to_sql('cleaned_messages_categories', engine, index=False)


def main():
    """
    Runs the above functions, given the input arguments from the system, in the correct order to prepare the data for modelling.
    """
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
#         print(df.head())
        
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