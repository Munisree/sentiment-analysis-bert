from transformers import pipeline, DistilBertTokenizer
from datasets import load_dataset
import pandas as pd
import torch
from tqdm import tqdm  # for progress bar

# Enable progress bar for pandas
tqdm.pandas()

# Check if GPU is available for faster processing
device = 0 if torch.cuda.is_available() else -1

# Load IMDb dataset
print("📥 Loading IMDb dataset...")
dataset = load_dataset("imdb")
train_dataset = dataset['train']
train_data = pd.DataFrame(train_dataset)

# Sample a subset of 5,000 reviews (you can adjust this number if you want)
subset_size = 5000
train_data = train_data.sample(n=subset_size, random_state=42)  # Randomly sample 5000 reviews

# --- BERT Sentiment Model ---
print("\n🧠 Loading BERT Sentiment Analysis pipeline...")
tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english")
bert_model = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english", tokenizer=tokenizer, device=device)

# Function to apply BERT model
def get_bert_sentiment(text):
    # Use the tokenizer to tokenize and automatically handle truncation
    inputs = tokenizer(text, padding=True, truncation=True, max_length=512, return_tensors="pt")
    result = bert_model.model(**inputs)  # Get model output
    
    # Access the prediction from the result (assuming result contains the logits)
    label = result[0]  # result is a tensor of shape [1, 2]
    predicted_label = label.argmax().item()  # Get the index of the highest logit
    
    # Map the index to the corresponding label
    sentiment_label = 'POSITIVE' if predicted_label == 1 else 'NEGATIVE'
    return sentiment_label

# Apply BERT Sentiment Model
print("\n🧠 Applying BERT to the subset of reviews...")
train_data['bert_sentiment'] = train_data['text'].progress_apply(get_bert_sentiment)

# Map the IMDb labels (0 = Negative, 1 = Positive)
train_data['true_sentiment'] = train_data['label'].apply(lambda x: "POSITIVE" if x == 1 else "NEGATIVE")

# --- Accuracy ---
bert_correct = (train_data['bert_sentiment'] == train_data['true_sentiment']).sum()
total = len(train_data)

print(f"\n📈 BERT Accuracy: {bert_correct}/{total} = {bert_correct/total:.2f}")

# --- Save to CSV ---
print("\n💾 Saving results to 'bert_sentiment_results.csv'...")
train_data[['text', 'bert_sentiment', 'true_sentiment', 'label']].to_csv("bert_sentiment_results.csv", index=False)

print("✅ Done!")
