# Multi-Label-Image-Classification-of-CelebA
In this project we are classifying an image into 40 classes which are the facial attribute.

## Preprocessing and EDA

- In the dataset we can observe that all the classes doesn't have equal number of image. So there is a high chance that the classes whose data frequency are more in the dataset will be dominated and potentially decrease the accuracy of model. We can add more data for the classes having low frequency to fix the inaccuracy.
 
- The size of each image in the dataset is **218 X 178**. This can be modified based on the model with which we are training or the hardware components.
- The channel of the image can be changed to **gray scale** to make the feature extraction process faster in the CNN
- Data augmentation can be performed to avoid **over fitting**
- **Normalization** each pixel to make sure it ranges between [0,1] 
- Converting all the "-1" values in the CSV file to "0"

## Implementation and Model Selection
- Since we are dealing with images we need to perform feature extraction on then we can choose variety for CNN having 2 layers or multiple layers like RESNET101 and etc.
- A non-linear function like **relu** or **tanh** can be used for the activation layers.
- At the end we have 3 dense layers having units 4096 ,512 ,40. It is important for the activation function of the final dense layer to be **sigmoid** and not softmax as we are dealing with multi label.
- Here I have used loss function as binary cross entropy and accuracy function binary accuracy because we are expecting a probability of the image belonging on each classes.
- After training the model is deployed and dockerized using FAST api.
## Usage
- Install the requirements using pip.
```
pip install -r requirements.txt
```
- File ``` train.py``` is used for training . All the changeable parameters are at the begining of the file. 
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
sudo docker build -t celeba
sudo docker run -d --name mycontainer -p 80:80 celeba
```

