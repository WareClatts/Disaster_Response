import json
import plotly
import pandas as pd

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar
from plotly.graph_objs import Pie
from sklearn.externals import joblib
from sqlalchemy import create_engine


app = Flask(__name__)

def tokenize(text):
    """
    As in train_classifier.py, this function is for lemmatizing, cleaning and removing stopwords from text, this time input by the user. 
    Input is the text to be processed, output is the clean, lemmatized tokens.
    """
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

# load data
engine = create_engine(f'sqlite:///data/DisasterResponse.db')

df = pd.read_sql_table('cleaned_messages_categories', engine)

# load model
model = joblib.load("models/classifier.pkl")


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    """
    Function to be run on the home or 'index' page of the webapp, showing the visualisations of the training dataset.
    """
    
    # extract data needed for visuals
    # TODO: Below is an example - modify to extract data for your own visuals
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)
    genre_counts = [genre_counts[x] for x in range(len(genre_counts))]
    
    categories = df.columns[~df.columns.isin(['id','message', 'original', 'genre'])].values.tolist()
    category_counts = [df[x].sum() for x in categories]
    categories = [x.replace('_', ' ') for x in categories]
    
    # create visuals
    # TODO: Below is an example - modify to create your own visuals
    graphs = [
        {
            'data': [
                Pie(
                    labels=genre_names,
                    values=genre_counts
                )
            ],

            'layout': {
                'title': 'Distribution of Message Genres',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Genre"
                }
            }
        },
        {
                    'data': [
                Bar(
                    x=categories, 
                    y=category_counts 
                )
            ],

            'layout': {
                'title': 'Distribution of Message Categories',
#                 'margin.pad': 500,
#                 'margin.b': 500,
#                 'orientation': "h",
                'height': 550,
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Message Category",
                    'tickangle': 45,
                    'ticklabelposition': "inside top"
                }
            }
        }
    ]
    
    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)
    
    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)


# web page that handles user query and displays model results
@app.route('/go')
def go():
    """
    Function to be run on the go page of the webapp, when a user inputs a custom message to be classified. Here the tokenizer and model are called and the predictions shown.
    """
    # save user input in query
    query = request.args.get('query', '') 

    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))

    # This will render the go.html Please see that file. 
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )


def main():
    app.run(host='0.0.0.0', port=3001, debug=True)


if __name__ == '__main__':
    main()