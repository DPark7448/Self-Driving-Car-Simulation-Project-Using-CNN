import tensorflow as tf
from tensorflow.keras import layers, regularizers

def nvidia_model(l2_reg: float = 1e-4) -> tf.keras.Model:
    l2 = regularizers.l2(l2_reg) if l2_reg else None
    model = tf.keras.Sequential([
        layers.Input(shape=(66, 200, 3)),
        layers.Conv2D(24, (5, 5), strides=(2, 2), activation="elu", kernel_regularizer=l2),
        layers.Conv2D(36, (5, 5), strides=(2, 2), activation="elu", kernel_regularizer=l2),
        layers.Conv2D(48, (5, 5), strides=(2, 2), activation="elu", kernel_regularizer=l2),
        layers.Conv2D(64, (3, 3), activation="elu", kernel_regularizer=l2),
        layers.Conv2D(64, (3, 3), activation="elu", kernel_regularizer=l2),
        layers.Flatten(),
        layers.Dense(100, activation="elu", kernel_regularizer=l2),
        layers.Dropout(0.2),
        layers.Dense(50, activation="elu", kernel_regularizer=l2),
        layers.Dense(10, activation="elu", kernel_regularizer=l2),
        layers.Dense(1)
    ])
    return model
