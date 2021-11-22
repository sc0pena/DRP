### Table of Contents

1. [Installation](#installation)
2. [Project Overview](#overview)
3. [Background](#background)
4. [File Descriptions](#descriptions)
5. [Instructions](#instructions)
6. [Results](#results)
7. [Licensing, Authors, and Acknowledgements](#licensing)

## Installation<a name="installation"></a>

Libraries used found in the Anaconda distribution of Python. This code works with Python version 3.

## Summary/Project Overview<a name="overview"></a>
Disaster Response Pipeline: Supervised learning model that takes in messages following a disaster and properly filters + classifies messages to relevant organizational areas for relief. Also creates an ETL pipeline to handle .csv data, an ML pipeline, then creates front end web app where an emergency worker can input a new message and get classification results in several categories. This app will display visualizations of the data.

## Background<a name="background"></a> 
Assignment for the Udacity Data Scientist nano degree program. 

## File Descriptions<a name="descriptions"></a> 
- ETL Pipeline Preparation.ipynb - notebook containing E2E data pipeline to create database (used to help create 'process_data.py' module)
- ML Pipeline Preparation.ipynb - notebook containing E2E ML model pipeline + testing, eval, and export to pickle file (used to help create 'train_classifier.py' module)
- process_data.py: module containing E2E data pipeline to create database
- train_classifier.py: module containing E2E ML model pipeline + testing and eval
- classifier.pkl: pickle file containing model
- run.py: module to create website
- disaster_categories.csv - pre-labeled category csv file (used for ETL pipeline)
- disaster_messages.csv - actual disaster messages for training and testing (used for ETL pipeline)
- DisasterResponse.db - file created from 'process_data.py' module
- go.html: performs results classifications on the website
- master.html: main website html page
- verify_webappworks.png: screenshot + URL of working website (https://view6914b2f4-3001.udacity-student-workspaces.com)

## Instructions<a name="instructions"></a>
Project works according to instructions at 'Project Workspace IDE in project 2': 
Here are the instructions in the readme.md provided by Udacity for this project:
NOTE: the workspace IDE has the file 'process_data.py' in the data folder; train_classifier.py in the models folder, run.py in the app folder, go.html and master.html in app/templates. Here are the instructions I followed and was able to view the website:

Run the following commands in the project's root directory to set up your database and model.
    - To run ETL pipeline that cleans data and stores in database:
        - first, download the files 'process_data.py', disaster_categories.csv, disaster_messages.csv to the data diretory in the IDE workspace, then run the command:
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves model to classifier.pkl:
        - first, download the file in github called 'train_classifier.py' to the models diretory in the IDE workspace, then run the command:
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

Run the following command in the app's directory to run your web app.
- first, download the file in github called 'run.py' to the models diretory in the IDE workspace, then run the command:
    `python run.py`

Open another terminal and type env|grep WORK this will give you the spaceid (it will start with view*** and some characters after that)
Now open your browser window and type https://viewa7a4999b-3001.udacity-student-workspaces.com, replace the whole viewa7a4999b with your space id that you got in the step 2
Press enter and the app should now run for you

## Results/Other<a name="results"></a> 
- project analyzed with/without stopwords (see ML Pipeline Preparation.ipynb)
- Chose multinomial Naive Bayes as 1st choice after reviewing the scikit algorithm cheat sheet:
  #https://scikit-learn.org/stable/tutorial/machine_learning_map/index.html
- Project rubic states "GridSearchCV is used to find the best parameters for the model" you will noticed I printed the best parameters (see train_classifier.py) but did not implement them into the         transformers as the rubic did not state this had to be done. However, in ML Pipeline Preparation.ipynb you can see how much better it performs with tuning.
- Multinominal Naive Bayes with tuned parameters peforms better than the K Neareset Neighbor model (see ML Pipeline Preparation.ipynb). Micro Average F1 (accuracy) and recall     scords much better with tuned MNB.   Precision similiar across all models. Will stick with fine tuned MNB as model of choice.
- CRISP-DM process flow, coding, and technical notation found within the Jupyter notebooks/modules

## Licensing, Authors, and Acknowledgements<a name="licensing"></a> 
Stack Overflow and other resources used (links found within the Jupyter notebooks/modules)
