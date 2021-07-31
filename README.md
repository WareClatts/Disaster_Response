# Disaster Response Pipeline Project
Project 2 of Udacity Data Science nanodegree

### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python app/run.py`

3. Go to http://0.0.0.0:3001/ (https://view6914b2f4-3001.udacity-student-workspaces.com/)

### Folders:
1. data contains: 
    - Training data for classification model:
        -  one csv of messages (disaster_messages.csv)
        -  one of categories that the messages correspond to (disaster_categories.csv)
    - Data processing script process_data.py. This script takes the above csv files as inputs, cleans/tokenizes and combines them before saving to a SQLite database.
    - SQLite Database DisasterResponse.db which contains the data output by the process_data.py script.
2. models contains:
	- train_classifier.py - script to train a multi-output classifier model for the disaster response messages. Takes as input the data held in the SQLite database output by process_data.py and trains a model using a pipeline and gridsearchcv. Please note that due to problems in the gridsearchcv taking a very long time it has not yet been fully optimised and only has two hyperparameters to test. The trained model is then exported as a .pkl file
    - classifier.pkl - the trained model output by train_classifier.py
3. app contains:
	- templates folder - this contains html templates for the 'home' or 'index' page of the web app (master.html), and for the 'go' page (go.html), where the web app user inputs a message to be classified.
    - run.py - script that takes as inputs the trained model as a .pkl file and data from the SQLite database DisasterResponse.db to produce some visualisations for the home page (formatted according to master.html), as well as takes as iput a message typed by the user and outputs classification predictions for all categories (albeit not very well).