from transformers import pipeline

# Create the pipeline
emotion_classifier = pipeline(
    "text-classification", model="AdamCodd/tinybert-emotion-balanced"
)

# Now you can use the pipeline to classify emotions
result = emotion_classifier(
    "We are delighted that you will be coming to visit us. It will be so nice to have you here."
)
print(result)
# [{'label': 'joy', 'score': 0.9895486831665039}]


"""
Plan: recieve sentences from transcriber and output analysis when done
    
"""
