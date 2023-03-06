# Reproducing ConvNeXt Models
This repository contains my approach for reproducing the [ConvNeXt](https://arxiv.org/abs/2201.03545) models using the [official implementation](https://github.com/facebookresearch/ConvNeXt) via Google Colaboratory. In this project, I tried to train the ConvNeXt model from scratch (random weights), not fine-tuning from pretrained models. The dataset used is a small dataset but similar to the ConvNeXt default (ImageNet).

## Dataset
The dataset used for this project is [Imagenette2](https://github.com/fastai/imagenette), which is a smaller version of the popular ImageNet dataset. Imagenette2 contains 10 classes and a total of 15,000 images, with each class having 600-1,000 images. The images are resized to 160x160 pixels. You can get the data [from here](https://s3.amazonaws.com/fast-ai-imageclas/imagenette2-160.tgz).

## Model Training
I trained two variants of ConvNeXt architectures in this project: ConvNeXt-Tiny and ConvNeXt-Small. Both models use the architecture and hyperparameters as described in the official ConvNeXt paper. The models were trained from scratch using the Imagenette2 dataset. The reason I am taking this two models is the architecture of them are less complicated than the bigger ones. Since I tried to train the model with a relatively small dataset, the less complicated architecture should fit better.

## Training
To train the models, I used Google Colaboratory, which provides free access to a GPU. The models were trained for 100 epochs with a batch size of 32 and 64, and a learning rate of 4e-3. The optimizer used was AdamW, and the dropout rate was set to 0.1. The input size of the images was 160x160 pixels. I followed the official [Installation](https://github.com/facebookresearch/ConvNeXt/blob/main/INSTALL.md) and [Training](https://github.com/facebookresearch/ConvNeXt/blob/main/TRAINING.md) from ConvNeXt repository. I documented my training process [in this notebook](https://github.com/lolikgiovi/ConvNeXt-Repro/blob/main/Training_History.ipynb).

## GPU Requirement
I tried to train the model on my local environment (I am using a M1 based MacBook), but the training always failed due to the unavailability of CUDA. Then, I switched to use Google Colaboratory with a Colab Pro subscription. I tried to choose to train using CPU than GPU by adding parameter argument when training, or even adding the code to train using CPU. The problem might be that my configuration is too high for CPU Training. More or less, I ended up have to train using a single GPU on colab. When I checked on the paper once more, I think the model is designed to train on GPU more or less.

## Monitoring the Training
Since ConvNeXt has also implemented Weight & Biases tools, it is more convenient for me to set-up the W&B and record my training logs easier.

## Results
Here is the result of models after being evaluated with Test Data. 

| Architecture     | Specific Configuration                         | Top-1 Accuracy | Model |
|------------------|------------------------------------------------|----------------|-------|
| ConvNeXt-T    | Batch size 32, color jitter 0.5, smoothing 0.2 | 85.885         | [Link](https://drive.google.com/file/d/1WgZHE80WELCdxjGSLj94UTfo_EueDoBm/view?usp=share_link) |
| ConvNeXt-S   | Batch size 32                                  | 85.172         | [Link](https://drive.google.com/file/d/1SIVKlTS_6kJ5yRIA8rK3pu1UYGT6pUJq/view?usp=share_link) |
| ConvNeXt-T    | Batch size 32                                  | 83.389         | [Link](https://drive.google.com/file/d/1me7f5HmgqM5f4-AxGnKQ-3LcUzk7NAvs/view?usp=share_link) |
| ConvNeXt-T    | Batch size 64                                  | 83.312         | [Link](https://drive.google.com/file/d/1klxyPyjyYi2LLMj8VUTchXsoPMOwVcU_/view?usp=share_link) |

Training Accuracy Trend:
![Training Acc](https://user-images.githubusercontent.com/59627864/222997332-5c6932c6-fb37-4dd8-ac62-dc2a9683384b.png)


From the result I got, I think it is more suitable to use smaller batch size and implement data augmentation when you try to train a model from scratch with relatively small dataset such as Imagenette.

## Reproducing
To reproduce the results, follow these steps in [this notebook](https://github.com/lolikgiovi/ConvNeXt-Repro/blob/main/Training_ConvNeXt.ipynb).

Acknowledgements
The official implementation of ConvNeXt is available at https://github.com/facebookresearch/ConvNeXt. The Imagenette2 dataset was created by fast.ai.
