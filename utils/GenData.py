from fontpreview import FontPreview
from wonderwords import RandomSentence
import os


#########################################################################
# Generates 100 sample images for each font style - increase dataset
#########################################################################
font_directory = '../fonts'
output_directory = '../data'

# Number of samples per font
num_samples_per_font = 100

# Initialize RandomSentence object
random_sentence_generator = RandomSentence()

os.makedirs(output_directory, exist_ok=True)

for font_file in os.listdir(font_directory):
    if font_file.endswith('.ttf'):
        font_path = os.path.join(font_directory, font_file)
        font_name = os.path.splitext(font_file)[0]

        font_output_dir = os.path.join(output_directory, font_name)
        os.makedirs(font_output_dir, exist_ok=True)

        for i in range(1, num_samples_per_font + 1):
            random_text = random_sentence_generator.sentence()[:-1]
            # print(random_text) # Generate random sentence

            font_preview = FontPreview(font_path, font_text = random_text)
            # font_preview.font_text = random_text
            image_filename = f'{font_name}_gen_{i}.png'
            image_path = os.path.join(font_output_dir, image_filename)

            # Save image
            font_preview.save(image_path)
