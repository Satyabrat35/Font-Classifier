# Font Classifier
This repository contains the code for a Font Classifier, a machine learning model that takes an image of text as input and predicts the font used in the image from a predefined list of 10 fonts. 
Four different classifier models were explored for this task: a CNN baseline, a modified CNN, a fine-tuned ResNet-18, and a fine-tuned Vision Transformer (ViT) model.

## Repository Structure
1. `data` - Contains a collection of 1792 images showcasing 10 distinct font styles.
2. `fonts` - Includes TrueType Font (.ttf) files utilized for generating additional samples.
3. `ipynb files` - Comprises IPython Notebook files corresponding to each employed model. These files were executed on the Kaggle platform.
4. `model checkpoint` - Houses the weights for all models, excluding the ViT model, which can be obtained through the provided steps.
5. `models` - Contains skeleton code for the CNN Baseline, CNN Modified, and Resnet Modified models, along with image processing techniques.
6. `test` - Includes two test files utilized for testing the developed or fine-tuned models with a sample image.
7. `utils` - Contains utility files, such as code for generating test sample images, data samples, and downloading ViT model weights.
8. `label_mapping.json` - Represents the mapping of label fonts to numeric values, utilizing Label Encoder.
9. `sample.png` - A sample image employed for model testing purposes.
10. `Inference.xlsx` - A documented record detailing the inference results obtained from each model.
11. `img` - Contains few images wrt to model and inference time.

## Instructions to Test the models
Download the repository code - 

``git clone git@github.com:Satyabrat35/Font-Classifier.git``

Conda Environment (optional) - 

``conda create --name <env_name>``

``conda activate <env_name>``

Install the required libraries (project root folder) - 

``pip install -r requirements.txt``

Download ViT weights (utils folder) - 

``cd utils``

``python DownloadTensorFile.py``

Create a sample image (utils folder) -

``cd utils``

``python SampleGen.py``

Test the CNN Baseline, CNN Modified and Resnet 18 Modified model (test folder) -

``cd test``

Uncomment the model and its weights code which you wish to test, by default its set to Resnet18

``python TestModel.py``

Test the ViT Model (same dir) - 

``python TestModelViT.py``

Example - 
![img.png](img/modeltest.png)


## Inference.xlsx
It contains the inference time i.e.average epoch time,
test evaluation time, accuracy score for each models in GPU and CPU setting (except ViT).
Also contains the model size, no of parameters and the test accuracy of each class
against each model.

Check - [Inference.xlsx](Inference.xlsx)

![img.png](img/inference.png)

## Metrics
Used averaged accuracy as the main metric for model evaluation.
Also calculated precision, recall and accuracy for each class for the 4 models.
One noticeable observation - CNN variants and Resnet18 model tend to perform 
poorly for the `Allura` font class.

Test Accuracy for each class - 
![img_1.png](img/AccuracyClassWise.png)


## Data split
1. Total images - 1792
2. Train set - 1254
3. Validation set - 269
4. Test set - 269