import pandas as pd
import re
import math
from collections import defaultdict
from sklearn.model_selection import train_test_split

# Load the dataset
df = pd.read_csv("spam.csv", encoding="ISO-8859-1")[["v1", "v2"]]
df.columns = ["label", "text"]

# Drop empty entries
df.dropna(subset=["label", "text"], inplace=True)

# Convert labels to binary
df['label'] = df['label'].map({'ham': 0, 'spam': 1})

# Text preprocessing
def preprocess(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    return text.split()

df['tokens'] = df['text'].apply(preprocess)

# Train-test split
train_data, test_data = train_test_split(df, test_size=0.2, random_state=42)

# Naive Bayes Classifier
class NaiveBayesClassifier:
    def __init__(self):
        self.vocab = set()
        self.word_freq = {0: defaultdict(int), 1: defaultdict(int)}
        self.class_freq = {0: 0, 1: 0}
        self.class_total_words = {0: 0, 1: 0}
        self.class_prob = {0: 0.0, 1: 0.0}

    def train(self, data):
        total_messages = len(data)

        for _, row in data.iterrows():
            label = row['label']
            tokens = row['tokens']

            self.class_freq[label] += 1
            self.class_total_words[label] += len(tokens)

            for word in tokens:
                self.vocab.add(word)
                self.word_freq[label][word] += 1

        self.class_prob[0] = self.class_freq[0] / total_messages
        self.class_prob[1] = self.class_freq[1] / total_messages

    def predict(self, tokens):
        scores = {}
        vocab_size = len(self.vocab)

        for c in [0, 1]:
            log_prob = math.log(self.class_prob[c])

            for word in tokens:
                word_count = self.word_freq[c][word] + 1
                total_words = self.class_total_words[c] + vocab_size
                log_prob += math.log(word_count / total_words)

            scores[c] = log_prob

        return 1 if scores[1] > scores[0] else 0

# Train and evaluate
model = NaiveBayesClassifier()
model.train(train_data)

correct = 0
total = len(test_data)

for _, row in test_data.iterrows():
    prediction = model.predict(row['tokens'])
    if prediction == row['label']:
        correct += 1

accuracy = correct / total
print(f"Accuracy: {accuracy * 100:.2f}%")

# Custom prediction
custom_message = "HAve you done your homeork?"
custom_tokens = preprocess(custom_message)
print("Spam" if model.predict(custom_tokens) else "Ham")

