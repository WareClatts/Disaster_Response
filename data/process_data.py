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
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    df = messages.merge(categories, how='inner', on='id')
    return df

def clean_data(df):
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
    engine = create_engine(f'sqlite:///{database_filename}') #.db
    sql = 'DROP TABLE IF EXISTS cleaned_messages_categories;'
    result = engine.execute(sql)
    df.to_sql('cleaned_messages_categories', engine, index=False)


def main():
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