import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline

# Training Data
training_data = [
    ("Submit project assignment by 5 PM", "High"),
    ("Emergency doctor appointment", "High"),
    ("Pay electricity bill urgently", "High"),
    ("Buy milk, eggs, and bread", "Low"),
    ("Watch new movie on Netflix", "Low"),
    ("Prepare slides for client presentation", "High"),
    ("Buy groceries for home", "Low"),
    ("Read 10 pages of book", "Low"),
]

X_train = [text for text, priority in training_data]
y_priority = [priority for text, priority in training_data]

# Train ML Model
priority_model = make_pipeline(TfidfVectorizer(), MultinomialNB())
priority_model.fit(X_train, y_priority)

# Save Model
joblib.dump(priority_model, "priority_model.pkl")
print("✅ Model trained & saved as priority_model.pkl successfully!")