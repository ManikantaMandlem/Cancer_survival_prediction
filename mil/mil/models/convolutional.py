import tensorflow as tf


def convolutional_model(D, ragged=False):
    inputs = tf.cond(
        ragged,
        lambda: tf.keras.Input([None, None, D], ragged=True),
        lambda: tf.keras.Input([None, None, D]),
    )

    c1 = tf.cond(
        ragged,
        lambda: tf.keras.layers.Conv2D(256, 3, activation="relu", padding="same")(
            inputs.to_tensor()
        ),
        lambda: tf.keras.layers.Conv2D(256, 3, activation="relu", padding="same")(
            inputs
        ),
    )

    c2 = tf.keras.layers.Conv2D(128, 3, activation="relu", padding="same")(c1)
    a1 = tf.keras.layers.Conv2D(1, 1, activation="relu", padding="same")(
        c2
    )  # (h, w, 1) attention weights
    h1 = tf.keras.layers.GlobalAveragePooling2D()(
        c1 * a1
    ) / tf.keras.layers.GlobalAveragePooling2D()(
        a1
    )  # weighted average pooling
    s1 = tf.keras.layers.Dense(1, activation="softmax")(h1)
    scores = tf.keras.layers.Concatenate(axis=1, name="softmax")([s1, 1 - s1])

    return tf.keras.Model(inputs=inputs, outputs=scores)
