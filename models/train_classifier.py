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
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, precision_recall_fscore_support


def load_data(database_filepath):
    engine = create_engine(f'sqlite:///{database_filepath}')#.db
    df = pd.read_sql_table('cleaned_messages_categories', engine)
    X = df['message']
    Y = df[df.columns[~df.columns.isin(['id', 'message', 'original', 'genre'])]]
    category_names = Y.columns
    
    return X, Y, category_names

def tokenize(text):
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()
    stop_words = stopwords.words("english")
    
    clean_tokens = []
    for tok in tokens:
        if tok not in stop_words:
            clean_tok = lemmatizer.lemmatize(tok).lower().strip()
            if clean_tok not in clean_tokens:
                clean_tokens.append(clean_tok)
                
    return clean_tokens


def build_model():    
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier()))])

    # define parameters for GridSearchCV
    parameters = {
        'clf__estimator__max_depth': [50, 100],
        }

    # create gridsearch object and return as final model pipeline
    cv =  GridSearchCV(pipeline, param_grid=parameters)
    
    return cv


def evaluate_model(model, X_test, Y_test, category_names):
    
#     model = cv.best_estimator_
    
    y_predictions = model.best_estimator_.predict(X_test)

    y_predictions_df = pd.DataFrame(data=y_predictions, columns = Y_test.columns)

    results_df = pd.DataFrame(index = Y_test.columns, columns=['f1 score', 'precision', 'recall'])

    for index, col in enumerate(Y_test.columns):
        y_true = Y_test[col]
        y_pred = y_predictions_df.iloc[:,index]
        result = precision_recall_fscore_support(y_true, y_pred, average='binary')
        for i in [0, 1, 2]:
            results_df.iloc[index,i] = result[i]
    
    print(results_df)
    
    return results_df


def save_model(model, model_filepath):
    # Export model as a pickle file
    import pickle
    pickle.dump(model, open(model_filepath, 'wb'))



def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

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