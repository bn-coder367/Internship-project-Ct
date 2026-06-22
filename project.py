#Task 1
import nltk

# Download tokenizer files
nltk.download('punkt')
nltk.download('punkt_tab')

from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lsa import LsaSummarizer

# Take user input
text = input("Paste your story or article:\n\n")

# Parse the text
parser = PlaintextParser.from_string(text, Tokenizer("english"))

# Create summarizer
summarizer = LsaSummarizer()

# Generate summary (3 sentences)
summary = summarizer(parser.document, 3)

# Print summary
print("\nSUMMARY:\n")

for sentence in summary:
    print(sentence)

 
#task 2
import speech_recognition as sr

# Create recognizer
recognizer = sr.Recognizer()

# Load audio file
with sr.AudioFile("voice.wav") as source:
    print("Listening to audio...")
    
    audio = recognizer.record(source)

# Convert speech to text
try:
    text = recognizer.recognize_google(audio)

    print("\nTRANSCRIBED TEXT:\n")
    print(text)

except sr.UnknownValueError:
    print("Could not understand audio")

except sr.RequestError:
    print("Internet connection error")

#task 3
import tensorflow as tf
import tensorflow_hub as hub
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

# Function to load image
def load_image(path):
    img = Image.open(path)
    img = img.resize((512, 512))
    img = np.array(img) / 255.0
    img = img.astype(np.float32)
    img = img[np.newaxis, :]
    return tf.constant(img)

# Load images
content_image = load_image("my.jpg")
style_image = load_image("style.jpg")

# Load AI style transfer model
model = hub.load(
    'https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/2'
)

# Generate styled image
stylized_image = model(content_image, style_image)[0]

# Show output image
plt.imshow(stylized_image[0])
plt.axis('off')
plt.show()
 
#task 4 

from transformers import pipeline

# Load GPT-2 text generator
generator = pipeline(
    "text-generation",
    model="gpt2"
)

# User prompt
prompt = input("Enter your topic or prompt:\n\n")

# Generate text
result = generator(
    prompt,
    max_length=120,
    num_return_sequences=1
)

# Print generated text
print("\nGENERATED TEXT:\n")

print(result[0]['generated_text'])
