# Disaster Response Pipeline Project

### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/

4. Access web app deployed at Heroku at: <br>
   https://response-classifier-webapp.herokuapp.com/
   
5. Example screenshots of index page and classification:<br>
    <div>
      <h4>Index Page</h4>
      <img src="/images/index_page.png" alt="index page" width="45%"/>
   </div>
   <div>
      <h4>Classification Page</h4>
      <img src="/images/classifying_example.png"  alt="classification example" width="45%"/>
   </div>
    