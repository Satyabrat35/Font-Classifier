import gdown


################################################################################
# Provide the sharable link for ViT Model tensor file
################################################################################
vit_url = 'https://drive.google.com/file/d/1N9_Z40lXykbc4cfoxwB4AshMcC7MnRNo/view'

output_path = 'model checkpoint/vit/model.safetensors'

gdown.download(vit_url, output_path, quiet=False, fuzzy=True)