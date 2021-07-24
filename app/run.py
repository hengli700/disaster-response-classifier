import json

# system path trick to import tokenize
import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import tokenize

import joblib
import pandas as pd
import plotly
from flask import Flask
from flask import render_template, request
from plotly.graph_objs import Bar
from sqlalchemy import create_engine

# download necessary nltk toolkits
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# create the Flask instance.
app = Flask(__name__)


# load data
dirname = os.path.dirname(__file__)
db_filename = os.path.join(dirname, '../data/DisasterResponse.db')
engine = create_engine('sqlite:///' + db_filename)
df = pd.read_sql_table('DisasterResponse', engine)

# load model
model_filename = os.path.join(dirname, '../models/classifier.pkl')
global model
model = joblib.load(model_filename)


# index webpage displays visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    # visualize data genre distribution
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)

    # visualize category distribution
    df_categories = df[df.columns[4:]]
    df_categories_count = df_categories.sum(axis=0).sort_values(ascending=False).to_frame(name='Count')
    df_categories_count.reset_index(inplace=True)
    df_categories_count = df_categories_count.rename(columns={'index':'Category'})

    # create visuals
    graphs = [
        {
            'data': [
                Bar(
                    x=df_categories_count['Category'],
                    y=df_categories_count['Count'],
                    marker=dict(color=df_categories_count['Count'],
                                colorscale='viridis')
                )
            ],

            'layout': {
                'title': 'Message Category Distribution',
                'font': {
                    'family': 'arial'
                },
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Category",
                    'tickfont': {
                        'size': 9
                    }
                },
                'margin':{
                    't': 120,
                    'b': 200
                }
            }
        },
        {
            'data': [
                Bar(
                    x=genre_names,
                    y=genre_counts,
                    marker=dict(color=genre_counts,
                                colorscale='viridis')
                )
            ],

            'layout': {
                'title': 'Message Genre Distribution',
                'font': {
                    'family': 'arial'
                },
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Genre"
                },
                'autosize': True
            }
        },
    ]

    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)

    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)


# web page that handles user query and displays model results
@app.route('/go')
def go():
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
