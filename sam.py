from fontpreview import FontPreview
from wonderwords import RandomSentence
import os
import json

random_sentence_generator = RandomSentence()
random_text = random_sentence_generator.sentence()[:-1]
font_preview = FontPreview('/Users/satya/Downloads/AYR-ml_project_files/fonts/GreatVibes.ttf', font_text = random_text)
img_name = 'sample.png'
font_preview.save(img_name)

