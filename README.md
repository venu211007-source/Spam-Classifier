# Spam Email Classifier

A machine learning project that classifies SMS messages as spam or not spam using Natural Language Processing (NLP) and a Naive Bayes classifier.

## Problem Statement
Spam messages are a major issue in digital communication. This project builds a model that automatically detects spam messages with high accuracy, helping filter unwanted content.

## Dataset
- SMS Spam Collection Dataset from Kaggle
- 5,572 SMS messages labeled as spam or ham (not spam)

## Tech Stack
- Python 3
- pandas — data loading and manipulation
- scikit-learn — machine learning and text vectorization
- CountVectorizer — converts text to numerical features
- Multinomial Naive Bayes — classification algorithm

## How It Works
1. Load and clean the dataset
2. Convert text messages into numerical vectors using CountVectorizer
3. Train a Naive Bayes model on 80% of the data
4. Test the model on the remaining 20%
5. Evaluate accuracy

## Result
Achieved 97.8% accuracy on test data.

## How to Run
1. Clone this repository
2. Install dependencies: pip install -r requirements.txt
3. Place spam.csv in the project folder
4. Run: python SPAM_CLASSIFIER.py

## Project Structure
spam-classifier/
├── SPAM_CLASSIFIER.py
├── spam.csv
├── requirements.txt
└── README.md