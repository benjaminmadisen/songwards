import tensorflow as tf

def simple_two_layer_model(input_size:int, layer_size:int):
    """ A simple two layer MLP.

    Args:
        input_size: size of input tensor.
        layer_size: size of middle layers.

    """
    return tf.keras.Sequential([
                tf.keras.Input(shape=(input_size,)),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.Dense(layer_size, activation='relu'),
                tf.keras.layers.Dense(layer_size, activation='relu'),
                tf.keras.layers.Dense(2, activation='softmax')])