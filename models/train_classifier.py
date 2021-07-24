import pickle
import pandas as pd

# system path trick to import tokenize from utils
import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import tokenize

from sqlalchemy import create_engine

from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV


def load_data(database_filepath):
    """
    Load data from store SQLite database.
    :param database_filepath: filepath where SQLite database is stored
    :return: (X, Y, category_names).
    X: numpy.ndarray containing values of 'message' column.
    Y: numpy.ndarray containing values (1 or 0) for individual category specific to the 'message' column.
    category_names: list containing values of category names
    """
    # load data from database
    engine = create_engine('sqlite:///'+database_filepath)
    df = pd.read_sql_table('DisasterResponse', engine)

    # extract message and category columns
    X = df['message'].values
    Y = df[df.columns[4:]].values
    category_names = df.columns[4:]

    return X, Y, category_names


def build_model():
    """
    Build optimizer model to be trained on. The model include a pipeline consisted of CountVectorizer, TfidfTransformer, and
    MultiOutputClassifier. In addition, the model also performs GridSearch with cross validation to find best
    parameters according to provided parameter grid.
    :return: A optimizer with estimator to be trained on
    """
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier()))
    ])

    # parameters = {
    #         'vect__ngram_range': ((1, 1), (1, 2)),
    #         # 'vect__max_df': (0.5, 1),
    #         # 'vect__max_features': (None, 5000),
    #         'tfidf__use_idf': (True, False),
    #         'clf__estimator__n_estimators': (10, 50),
    #         # 'clf__estimator__min_samples_split': (2, 4)
    #     }

    # use only one parameter to facilitate deployment on Heroku due to limited app size.

    parameters = {
        'clf__estimator__n_estimators': [10],
    }

    cv = GridSearchCV(pipeline, param_grid=parameters, verbose=99)

    return cv


def evaluate_model(model, X_test, Y_test, category_names):
    """
    Evaluate trained model and output classification report for each category.
    :param model: Trained optimizer with estimator
    :param X_test: Test set for observations ('message')
    :param Y_test: Test set for labels (individual category)
    :param category_names: Category names
    :return:
    """
    Y_pred = model.predict(X_test)
    for col_index in range(len(category_names)):
        category = category_names[col_index]
        print(f'-->[category= {category}]')
        print(classification_report(Y_test[col_index], Y_pred[col_index]))
        print()


def save_model(model, model_filepath):
    """
    Pickle trained model.
    :param model: Trained optimizer with estimator
    :param model_filepath: Filepath to save model
    """
    pickle.dump(model, open(model_filepath, mode='wb'))


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

        print('Best params for trained model...')
        print(model.best_params_)
        
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