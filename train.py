import tensorflow as tf

physical_devices = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)


import matplotlib.pyplot as plt
import matplotlib.image as imread
from keras.models import Sequential
from keras_preprocessing.image import ImageDataGenerator
from keras.layers import Dense, Activation, Flatten, Dropout, BatchNormalization
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions
from keras.layers import Conv2D, MaxPooling2D
from keras import regularizers, optimizers
from tensorflow.keras.preprocessing.image import img_to_array
from keras.preprocessing import image
import pandas as pd
import numpy as np
import keras
import cv2
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, TensorBoard, EarlyStopping

COL = ['5_o_Clock_Shadow',
 'Arched_Eyebrows',
 'Attractive',
 'Bags_Under_Eyes',
 'Bald',
 'Bangs',
 'Big_Lips',
 'Big_Nose',
 'Black_Hair',
 'Blond_Hair',
 'Blurry',
 'Brown_Hair',
 'Bushy_Eyebrows',
 'Chubby',
 'Double_Chin',
 'Eyeglasses',
 'Goatee',
 'Gray_Hair',
 'Heavy_Makeup',
 'High_Cheekbones',
 'Male',
 'Mouth_Slightly_Open',
 'Mustache',
 'Narrow_Eyes',
 'No_Beard',
 'Oval_Face',
 'Pale_Skin',
 'Pointy_Nose',
 'Receding_Hairline',
 'Rosy_Cheeks',
 'Sideburns',
 'Smiling',
 'Straight_Hair',
 'Wavy_Hair',
 'Wearing_Earrings',
 'Wearing_Hat',
 'Wearing_Lipstick',
 'Wearing_Necklace',
 'Wearing_Necktie',
 'Young']


IMG_DIR = './archive/img/img/'
DATASET_DIR ='archive/list_attr_celeba.csv'
SIZE = 150
n_rows = 20000
BATCH_SIZE = 16
CLASSES = 40
DEPTH = 3
EPOCHS = 20





class Models:
    @staticmethod
    def build_VGG(width , height , depth , classes):
        
        vgg16 = keras.applications.vgg16
        conv_model = vgg16.VGG16(weights='imagenet', include_top=False, input_shape=(height,width,depth))
        x = keras.layers.Flatten()(conv_model.output)
        for layer in conv_model.layers[0:-10]:
             layer.trainable = False
        x = keras.layers.Dense(4096, activation='relu')(x)
        x = keras.layers.Dense(512, activation='relu')(x)
        x = keras.layers.Dropout(0.5)(x)
        predictions = keras.layers.Dense(classes, activation='sigmoid')(x)
        full_model = keras.models.Model(inputs=conv_model.input, outputs=predictions)
        full_model.summary()

        full_model.compile(loss='binary_crossentropy',
              optimizer='adadelta',
              metrics='binary_accuracy')

        return full_model
    @staticmethod 
    def build_CNN(width , height , depth , classes):
        model = Sequential()
        
        model.add(Conv2D(filters=16, kernel_size=(5, 5), activation="relu", input_shape=(height,width,depth)))
        model.add(BatchNormalization())
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.2))
        
        model.add(Conv2D(filters=32, kernel_size=(5, 5), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(BatchNormalization())
        model.add(Dropout(0.2))
        
        model.add(Conv2D(filters=64, kernel_size=(5, 5), activation="relu"))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(BatchNormalization())
        model.add(Dropout(0.2))
        
        model.add(Conv2D(filters=64, kernel_size=(5, 5), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(BatchNormalization())
        model.add(Dropout(0.2))
        
        model.add(Flatten())
        model.add(Dense(128, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(64, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(classes, activation='sigmoid'))
        model.compile(loss='binary_crossentropy',
                      optimizer='adadelta',
                      metrics='binary_accuracy')
        return model 
    
    @staticmethod 
    def build_RESNET50(width , height , depth , classes):
        from keras.applications.resnet50 import ResNet50
        model = Sequential()
        model.add(ResNet50(include_top = False, pooling = 'avg', weights = 'imagenet'))
        model.add(Dense(4096, activation = 'relu'))
        model.add(Dropout(0.5))
        model.add(Dense(1024, activation = 'relu'))
        model.add(Dropout(0.5))
        model.add(Dense(classes , activation = 'sigmoid'))
        model.layers[0].trainable = False
        model.compile(loss='binary_crossentropy',
                      optimizer='adadelta',
                      metrics='binary_accuracy')
        return model

def preprocess(DATASET_DIR, IMG_DIR, SIZE, n_rows, BATCH_SIZE):
    df = pd.read_csv(DATASET_DIR)
    df = df.head(n_rows)
    df = df.replace([-1],0)
    rows , cols = df.shape
    columns = []
    for cl in df.columns:
        if cl != 'image_id':
            columns.append(cl)
    
    datagen = ImageDataGenerator(rotation_range=20, 
                                   rescale=1./255, 
                                   width_shift_range=0.2, 
                                   height_shift_range=0.2, 
                                   shear_range=0.2, 
                                   zoom_range=0.2, 
                                   horizontal_flip=True, 
                                   fill_mode='nearest')
    
    # test_datagen=ImageDataGenerator(rescale=1./255.)
    
    train_generator=datagen.flow_from_dataframe(
    dataframe=df,
    directory=IMG_DIR,
    x_col="image_id",
    y_col=columns,
    batch_size=BATCH_SIZE,
    shuffle=True,
    class_mode="other",
    target_size=(SIZE,SIZE))
    
    valid_generator=datagen.flow_from_dataframe(
    dataframe=df,
    directory=IMG_DIR,
    x_col="image_id",
    y_col=columns,
    batch_size=BATCH_SIZE,
    shuffle=True,
    class_mode="other",
    target_size=(SIZE,SIZE))
    
    return train_generator , valid_generator , columns

def pred(img_dir , model):
    img = imread.imread(img_dir)
    plt.imshow(img)
    img = image.load_img(img_dir, target_size=(SIZE,SIZE,3))
    img = image.img_to_array(img)
    img = img/255.
    img = np.expand_dims(img, axis=0)
    img = preprocess_input(img)
    prd = model.predict(img)      
    pred_dict = {COL[i]: prd[0][i] for i in range(len(prd[0]))}
    
    return pred_dict
    
def best_pred(dic):
    best_prd = []
    ser = list(dic.items())
    for i in range(len(ser)):
        if ser[i][1] > 0.75 :
            best_prd.append((ser[i][0] , ser[i][1]))
    
    return best_prd
            



print("[INFO] PREPROCESSING")
train , val , col= preprocess(DATASET_DIR, IMG_DIR, SIZE, n_rows, BATCH_SIZE)
print("[INFO] DONE")

print("[INFO] COMPILING AND LOADING NEURAL NET")
Net = Models()
model = Net.build_VGG(SIZE , SIZE , DEPTH , CLASSES)
STEP_SIZE_TRAIN=train.n//train.batch_size
STEP_SIZE_VALID=val.n//val.batch_size

earlystopper = EarlyStopping(monitor='val_loss', min_delta=0, patience=2, verbose=0, mode='auto', baseline=None, restore_best_weights=True)

print("[INFO] TRAINING")
history = model.fit_generator(generator=train,
                    steps_per_epoch=STEP_SIZE_TRAIN,
                    validation_data=val , 
                    validation_steps=STEP_SIZE_VALID,
                    epochs=EPOCHS,
                    callbacks = [earlystopper]
)
print("[INFO] FINISHED TRAINING")


plt.plot(history.history['binary_accuracy'])
plt.plot(history.history['val_binary_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()
# "Loss"
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()
