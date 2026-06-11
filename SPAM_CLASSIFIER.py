import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report

# Load data
df = pd.read_csv('spam.csv', encoding='latin-1')[['v1', 'v2']]
df.columns = ['label', 'message']

# Convert labels to numbers
df['label'] = df['label'].map({'ham': 0, 'spam': 1})

# Split data
X_train, X_test, y_train, y_test = train_test_split(df['message'], df['label'], test_size=0.2, random_state=42)

# Convert text to numbers
vectorizer = CountVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Train model
model = MultinomialNB()
model.fit(X_train_vec, y_train)

# Test model
predictions = model.predict(X_test_vec)
print(f"Accuracy: {accuracy_score(y_test, predictions) * 100:.2f}%\n")

# NEW: Print a more professional classification report
print("Classification Report:")
print(classification_report(y_test, predictions, target_names=['Ham', 'Spam']))

# NEW: Add an interactive feature to test custom messages
print("\n" + "="*40)
print("Test the Spam Classifier with your own messages!")
print("Type 'exit' to quit.")
print("="*40)

while True:
    user_input = input("\nEnter a message: ")
    if user_input.lower() == 'exit':
        break
    
    # Transform and predict
    input_vec = vectorizer.transform([user_input])
    prediction = model.predict(input_vec)[0]
    
    label_result = "SPAM" if prediction == 1 else "HAM (Not Spam)"
    print(f"Prediction: {label_result}")