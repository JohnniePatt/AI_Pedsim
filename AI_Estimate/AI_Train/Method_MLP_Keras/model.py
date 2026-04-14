import keras
from keras import layers


def build_model(input_dim, config):
    """Build a MLP time estimator using the Keras Functional API."""
    model_cfg = config.get("model", {})
    hidden_dims = model_cfg.get("hidden_dims", [128, 64, 32])
    dropout = float(model_cfg.get("dropout", 0.1))
    output_dim = len(config["features"]["target"])

    inputs = keras.Input(shape=(input_dim,), name="features")
    x = inputs
    for dim in hidden_dims:
        x = layers.Dense(dim, activation="relu")(x)
        x = layers.LayerNormalization()(x)
        x = layers.Dropout(dropout)(x)
    outputs = layers.Dense(output_dim, name="scaled_time_outputs")(x)

    return keras.Model(inputs=inputs, outputs=outputs, name="time_estimator_mlp_keras")
