# Multi-Label-Image-Classification-of-CelebA

## Description: 

The project disintegrates an image into various classes related to facial attributes indicating the probability of each attribute being present in the input image, based off of the dataset CelebA <https://www.kaggle.com/jessicali9530/celeba-dataset> consisting of 200k images and 40 classes each pointing to a specific facial attributes.

## Pre-processing, EDA and Challenges:

- In the dataset provided, we observe that each of the attributes do not have a high volume of images under it with a tolerance that is almost close to the number of images in each of the other attributes. In simpler terms, there is an imbalance noted in the equaity and high quantity of images available for each attribute. Owing to this imbalance, the probabiltiy of - the attributes whose data frequency are more in the dataset, will be high, consequentially leading to over fitting and decreasing the gross accuracy of the model. The fix to the imbalance causing the anomalies is a compensation – adding equal and quantifiable number of datum (images) for attributes with lower frequency.
- The size of each image is reduced to an aspect ratio of **150*150** as a feeder to the neural network created in the model. The primary factor to choose the aspect ratio being hardware availability.
- The channel of the image can be changed to **gray scale** to make the feature extraction process faster in the CNN
- Data augmentation has been performed to avoid **over fitting**
- **Normalization** ensures that the pixel values range between [0,1] which necessarily means that every pixel is given an almost equal amount of consideration for computation, thereby contributing to higher accuracy. In this process of normalization, all negative values in the CSV are mapped against zero. Without normalization, accuracy takes a toll.
- 
## Implementation Scheme and Model Selection:

- With a variety of CNNs available for multiple layers of feature extraction like RESNET101, RESNET50 VGG16, etc., VGG16 has been used to achieve **feature extraction** of input images.
- A non-linear function like **relu** or **tanh** can be used for the activation layers.Here I have used relu 
- There are three dense layers having 4096, 512 and 40 units. The first two layers employ non-linear functions like relu or tanh for layer activation. The third layer, being multi-label, needs to prominently use **sigmoid and not softmax**.
- To live-up-to the expectation of the probability of an image belonging to each class, loss and accuracy functions have been utilized as means to accomplish ***binary cross entropy*** and ***binary accuracies*** respectively.
- After training the model is deployed and dockerized using FAST api.

## Usage:

- Install the requirements using pip.
```
pip install -r FAST/requirements.txt
```
- File ``` train.py``` is used for training . All the changeable parameters are at the begining of the file. Save the model and copy inside the folder ```FAST```  
```
IMG_DIR = <path to image dir>
DATASET_DIR = <path to list_attr_celeba.csv>
SIZE = <size of image>
n_rows = <Number of rows to extract from CSV file>
BATCH_SIZE = <batch size >
CLASSES = 40
DEPTH = <total number of channel for the image >
EPOCHS = <Total number of epochs>
```
- For testing with a single image we can use pred() and best_pred() function inside ```train.py```
- In order to run the FAST api use the following code and visit <https://localhost/home/your_name>
```
cd FAST
uvicorn app.main:app
```
- In order to run using docker use the following code and visit <http://127.0.0.1:80/home/your_name>
```
cd FAST
sudo python3 -m venv <name of the environment>
pip3 install -r requirements.txt
sudo docker build -t celeba .
sudo docker run -d --name mycontainer -p 80:80 celeba 
```

## Result
- Accruacy Graph for VGG16

![Alt text](./acc.png?raw=true "Accruacy Graph for VGG16")

- Loss Graph for VGG16

![Alt text](./loss.png?raw=true "Loss Graph for VGG16")

- The model was trained for 20 epochs on GPU with a binary accuracy 0.89
