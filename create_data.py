import json
import numpy as np
import os

print("Creating sample training data...")

# Create sample data with embeddings
sample_data = {}
for i in range(100):  # Create 100 samples
    # Random embeddings (simulating Cohere embeddings)
    embeddings = np.random.randn(768).tolist()
    
    # Random text samples
    texts = [
        "I'm so happy and excited!",
        "This makes me angry!",
        "I feel sad today.",
        "Wow, that's surprising!",
        "I trust you completely.",
        "I'm afraid of the dark.",
        "This food is disgusting.",
        "I anticipate great things.",
        "Joy fills my heart.",
        "Fear grips my soul."
    ]
    
    # Random labels
    label_sets = [
        ["Joy", "Anticipation"],
        ["Anger", "Disgust"],
        ["Sadness", "Fear"],
        ["Surprise", "Joy"],
        ["Trust"],
        ["Fear"],
        ["Disgust"],
        ["Anticipation"],
        ["Joy"],
        ["Fear", "Sadness"]
    ]
    
    idx = i % len(texts)
    sample_data[str(i)] = {
        "text": texts[idx],
        "labels_text": label_sets[idx],
        "embeddings": embeddings
    }

# Save to file
output_path = os.path.join("data", "xed_with_embeddings.json")
with open(output_path, 'w') as f:
    json.dump(sample_data, f, indent=2)

print(f"✅ Sample data created at: {output_path}")
print(f"📊 Number of samples: {len(sample_data)}")
print(f"🎯 Emotions included: Joy, Anticipation, Anger, Disgust, Sadness, Fear, Surprise, Trust")