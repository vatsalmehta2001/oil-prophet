import os
import json

print("Checking models directory...")
print("Models available:", os.listdir("models"))

print("\nMetadata files available:")
for f in os.listdir("models"):
    if "_metadata" in f:
        print(" -", f)
        
print("\nChecking sentiment model metadata:")
try:
    metadata_path = "models/sentiment_enhanced_lstm_metadata.json"
    if os.path.exists(metadata_path):
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        print(f"Metadata found: {list(metadata.keys())}")
    else:
        print("Metadata file not found")
except Exception as e:
    print(f"Error loading metadata: {str(e)}")

print("\nChecking model existence:")
model_path = "models/sentiment_enhanced_lstm.h5"
if os.path.exists(model_path):
    print(f"Model file exists: {model_path}")
    print(f"File size: {os.path.getsize(model_path) / (1024*1024):.2f} MB")
else:
    print(f"Model file not found: {model_path}")
