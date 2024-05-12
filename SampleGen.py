from fontpreview import FontPreview
from wonderwords import RandomSentence
import os
import json
import random


#########################################################################
# Generates a sample image for testing the models from .tff font styles
#########################################################################
fonts_folder = 'fonts'
files = [file for file in os.listdir(fonts_folder) if file.endswith('.ttf')]

random_file = random.choice(files)
random_file_path = os.path.join(fonts_folder, random_file)
font_name = os.path.splitext(random_file)[0]

# Generate a random sentence
random_sentence_generator = RandomSentence()
random_text = random_sentence_generator.sentence()[:-1] # Remove period '.'
font_preview = FontPreview(random_file_path, font_text = random_text)
img_name = 'sample.png'
font_preview.save(img_name)

print(f'Sample Image generated of font-type {font_name}')
