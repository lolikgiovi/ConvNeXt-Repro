# Reproducing ConvNeXt Models
This repository contains the code for reproducing the ConvNeXt models using the official implementation via Google Colaboratory. In this project, I trained the model from scratch (random weights), not fine-tuning from already established models. The dataset used is a small dataset but similar to the ConvNeXt default (ImageNet).

## Dataset
The dataset used for this project is Imagenette2, which is a smaller version of the popular ImageNet dataset. Imagenette2 contains 10 classes and a total of 15,000 images, with each class having 600-1,000 images. The images are resized to 160x160 pixels.

## Models
I trained two models in this project: ConvNeXt_tiny and ConvNeXt_small. Both models use the architecture and hyperparameters as described in the official ConvNeXt paper. The models were trained from scratch using the Imagenette2 dataset.

## Training
To train the models, I used Google Colaboratory, which provides free access to a GPU. The models were trained for 100 epochs with a batch size of 32 and a learning rate of 4e-3. The optimizer used was AdamW, and the dropout rate was set to 0.1. The input size of the images was 160x160 pixels.

## Results
The final accuracy achieved by ConvNeXt_tiny was 84.13%, while the final accuracy achieved by ConvNeXt_small was 88.15%. The training logs and trained models are available in the logs and checkpoints directories, respectively.

## Reproducing
To reproduce the results, follow these steps:

Clone this repository: git clone https://github.com/[USERNAME]/[REPOSITORY_NAME].git
Open the ConvNeXt_Training.ipynb notebook in Google Colaboratory.
Follow the instructions in the notebook to train the models.
Note that training the models can take several hours depending on the compute resources available.

Acknowledgements
The official implementation of ConvNeXt is available at https://github.com/facebookresearch/ConvNeXt. The Imagenette2 dataset was created by fast.ai.
