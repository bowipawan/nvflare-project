import tensorflow as tf
# from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers

def build_model(NUM_CLASSES,IMG_SIZE,top_dropout_rate):
    inputs = layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
    model = EfficientNetB0(include_top=False, input_tensor=inputs, weights="imagenet")
#     model = ResNet50(include_top=False, input_tensor=inputs)
    # Freeze the pretrained weights
    model.trainable = True

    # Rebuild top
    x = layers.GlobalAveragePooling2D(name="avg_pool")(model.output)
    x = layers.BatchNormalization()(x)

    x = layers.Dropout(top_dropout_rate, name="top_dropout")(x)
    outputs = layers.Dense(NUM_CLASSES, activation="softmax", name="pred")(x)

    # Compile
    model = tf.keras.Model(inputs, outputs, name="Resnet")
    optimizer = tf.keras.optimizers.Adam(clipnorm=1.,learning_rate=1e-4)
    model.compile(
        optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy"]
    )
    return model

class Net(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.NUM_CLASSES = 2
        self.IMG_SIZE = 224
        self.top_dropout_rate = 0.2
#         self.inputs = layers.Input(shape=(self.IMG_SIZE, self.IMG_SIZE, 3))
#         self.model = EfficientNetB0(include_top=False, input_tensor=self.inputs, weights="imagenet")
        self.model = build_model(2,224,0.3)
        self.model.trainable = True
#         self.pooling2D = layers.GlobalAveragePooling2D(name="avg_pool")
#         self.normalization = layers.BatchNormalization()
#         self.dropout = layers.Dropout(self.top_dropout_rate, name="top_dropout")
#         self.outputs = layers.Dense(self.NUM_CLASSES, activation="softmax", name="pred")
        
    def call(self, x):
        x = self.model(x)
#         x = self.normalization(x)
#         x = self.dropout(x)
#         x = self.outputs(x)
        return x 
    
