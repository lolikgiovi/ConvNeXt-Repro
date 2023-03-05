# Reproducing ConvNeXt Models
This repository contains my approach for reproducing the ConvNeXt models using the official implementation via Google Colaboratory. In this project, I tried to train the ConvNeXt model from scratch (random weights), not fine-tuning from pretrained models. The dataset used is a small dataset but similar to the ConvNeXt default (ImageNet).

## Dataset
The dataset used for this project is [Imagenette2](https://github.com/fastai/imagenette), which is a smaller version of the popular ImageNet dataset. Imagenette2 contains 10 classes and a total of 15,000 images, with each class having 600-1,000 images. The images are resized to 160x160 pixels. You can get the data [from here](https://s3.amazonaws.com/fast-ai-imageclas/imagenette2-160.tgz).

## Model Training
I trained two models in this project: ConvNeXt_tiny and ConvNeXt_small. Both models use the architecture and hyperparameters as described in the official ConvNeXt paper. The models were trained from scratch using the Imagenette2 dataset. The reason I am taking this two models is the architecture of them are less complicated than the bigger ones. Since I tried to train the model with a relatively small dataset, the less complicated architecture will fit better.

## Training
To train the models, I used Google Colaboratory, which provides free access to a GPU. The models were trained for 100 epochs with a batch size of 32 and a learning rate of 4e-3. The optimizer used was AdamW, and the dropout rate was set to 0.1. The input size of the images was 160x160 pixels. I followed the official [Installation](https://github.com/facebookresearch/ConvNeXt/blob/main/INSTALL.md) and [Training](https://github.com/facebookresearch/ConvNeXt/blob/main/TRAINING.md) from ConvNeXt repository.

## GPU Requirement
I tried to train the model on my local environment (I am using a M1 based MacBook), but the training always failed due to the unavailability of CUDA. Then, I switched to use Google Colaboratory with a Colab Pro subscription. I tried to choose to train using CPU than GPU by adding parameter argument when training, or even adding the code to train using CPU. The problem might be that my configuration is too high for CPU Training. More or less, I ended up have to train using a single GPU on colab. When I checked on the paper once more, I think the model is designed to train on GPU more or less.

## Monitoring the Training
Since ConvNeXt has also implemented Weight & Biases tools, it is more convenient for me to set-up the W&B and record my training logs easier.

## Results
The final accuracy achieved by ConvNeXt_tiny was 84.13%, while the final accuracy achieved by ConvNeXt_small was 88.15%. The training logs and trained models are available in the logs and checkpoints directories, respectively.

## Reproducing
To reproduce the results, follow these steps:

Clone this repository: git clone https://github.com/lolikgiovi/ConvNeXt-Repro.git
Open the ConvNeXt_Training.ipynb notebook in Google Colaboratory.
Follow the instructions in the notebook to train the models.
Note that training the models can take several hours depending on the compute resources available.

Acknowledgements
The official implementation of ConvNeXt is available at https://github.com/facebookresearch/ConvNeXt. The Imagenette2 dataset was created by fast.ai.
