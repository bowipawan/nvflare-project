import tensorflow as tf
import numpy as np
import pandas as pd
import cv2
import math

from net import Net
from nvflare.apis.event_type import EventType
from nvflare.apis.fl_constant import FLConstants, ShareableKey, ShareableValue
from nvflare.apis.fl_context import FLContext
from nvflare.apis.shareable import Shareable
from nvflare.apis.trainer import Trainer
from tensorflow.keras.callbacks import ReduceLROnPlateau

def convert_data(df,path):
    IMG_SIZE = 224
    ds = []
    for e in df.iloc:
        img = cv2.imread(path+e['image'])
        img = cv2.resize(img, (IMG_SIZE,IMG_SIZE))
        ds.append(img)
    ds = np.array(ds)
    y = pd.get_dummies(df.labels)
    return [ds,y]

class SimpleTrainer(Trainer):
    def __init__(self, epochs_per_round):
        super().__init__()
        self.epochs_per_round = epochs_per_round
        self.train_images, self.train_labels = None, None
        self.test_images, self.test_labels = None, None
        self.model = None

    def handle_event(self, event_type: str, fl_ctx: FLContext):
        if event_type == EventType.START_RUN:
            self.setup(fl_ctx)

    def setup(self, fl_ctx):
        PATH_NAME = '/workspace/nvflare/fl-test/data/'

        train = pd.read_csv(PATH_NAME+'train.csv')
        train.columns = ['image','labels']

        val = pd.read_csv(PATH_NAME+'val.csv')
        val.columns=['image','labels']
        
        IMG_SIZE = 224
        NUM_CLASSES = 2
        
        X_train,y_train = convert_data(train, PATH_NAME)
        X_val,y_val = convert_data(val, PATH_NAME)
        
#         train_data_gen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1.0/255.0,rotation_range=30, horizontal_flip=True, zoom_range=[0.7,1.0], brightness_range=[0.5,1.0])
#         val_data_gen = tf.keras.preprocessing.image.ImageDataGenerator(rescale= 1.0 / 255.0)     

        train_data_gen = tf.keras.preprocessing.image.ImageDataGenerator(
            rotation_range=30, horizontal_flip=True, zoom_range=[0.7,1.0], brightness_range=[0.5,1.0], featurewise_std_normalization=True,rescale= 1.0 / 255.0,featurewise_center=True)
        val_data_gen = tf.keras.preprocessing.image.ImageDataGenerator(featurewise_std_normalization=True,rescale= 1.0 / 255.0,featurewise_center=True)
        train_data_gen.fit(X_train)
        val_data_gen.fit(X_val)     
        
        self.train_it = train_data_gen.flow(X_train, y_train)
        self.val_it = val_data_gen.flow(X_val, y_val)        
        self.class_weight = {0: 1., 1: 4.}

#         self.train_images
#         self.train_labels
#         self.test_images
#         self.test_labels

        model = Net()
        
        loss_fn = "categorical_crossentropy"
        optimizer = tf.keras.optimizers.Adam(clipnorm=1.,learning_rate=1e-4)
#         optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)
        model.compile(optimizer=optimizer, loss=loss_fn, metrics=["accuracy"])
#         _ = model(tf.keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3)))

        self.model = model

    def train(self, shareable: Shareable, fl_ctx: FLContext, abort_signal) -> Shareable:
        """
        This function is an extended function from the super class.
        As a supervised learning based trainer, the train function will run
        evaluate and train engines based on model weights from `shareable`.
        After finishing training, a new `Shareable` object will be submitted
        to server for aggregation.

        Args:
            shareable: the `Shareable` object acheived from server.
            fl_ctx: the `FLContext` object achieved from server.
            abort_signal: if triggered, the training will be aborted.

        Returns:
            a new `Shareable` object to be submitted to server for aggregation.
        """

        # retrieve model weights download from server's shareable
        model_weights = shareable[ShareableKey.MODEL_WEIGHTS]

        # use previous round's client weights to replace excluded layers from server
        prev_weights = {
            str(key): value for key, value in enumerate(self.model.get_weights())
        }
        for key, value in model_weights.items():
            if np.all(value == 0):
                model_weights[key] = prev_weights[key]
                
#         for key,value in model_weights.items():
#             if np.any(math.isnan(value)):
#                 self.logger.info(f"weight is nan")

        # update local model weights with received weights
        
        self.model.set_weights(list(model_weights.values()))

        # adjust LR or other training time info as needed
        # such as callback in the fit function
#         self.model.fit(
#             self.train_images,
#             self.train_labels,
#             epochs=self.epochs_per_round,
#             validation_data=(self.test_images, self.test_labels),
#         )

        reduce_lr = ReduceLROnPlateau(monitor="val_accuracy", mode="max", patience=3, min_lr=0.000001)
        model_checkpoint = tf.keras.callbacks.ModelCheckpoint(
        filepath="/workspace/nvflare/fl-test2/model.h5",
        save_weights_only=True,
        monitor='val_loss',
        mode='min',
        save_best_only=True)

        self.model.fit(
            self.train_it,
            epochs = self.epochs_per_round,
            validation_data = self.val_it,
#             class_weight = self.class_weight,
            callbacks = [reduce_lr,model_checkpoint]
        )
        
        self.model.load_weights("/workspace/nvflare/fl-test2/model.h5")
        
        # report updated weights in shareable
        shareable = Shareable()
        shareable[ShareableKey.TYPE] = ShareableValue.TYPE_WEIGHTS
        shareable[ShareableKey.DATA_TYPE] = ShareableValue.DATA_TYPE_UNENCRYPTED
        shareable[ShareableKey.MODEL_WEIGHTS] = {
            str(key): value for key, value in enumerate(self.model.get_weights())
        }
#         self.logger.info(f"Sending shareable to server: \n {shareable[ShareableKey.MODEL_WEIGHTS]}")
        self.logger.info(f"Sending shareable to server: \n ")
        return shareable
