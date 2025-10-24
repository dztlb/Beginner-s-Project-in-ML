# -*- coding: utf-8 -*-
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def build_tcn_attention_bigru(input_shape):
    try:
        import tensorflow as tf
        from tensorflow import keras
    except Exception as e:
        raise RuntimeError("TensorFlow not available") from e

    inputs = keras.Input(shape=input_shape)

    def tcn_block(x, filters, kernel_size, dilation_rate):
        y = keras.layers.Conv1D(filters, kernel_size, padding="causal",
                                dilation_rate=dilation_rate, activation="relu")(x)
        y = keras.layers.BatchNormalization()(y)
        if x.shape[-1] != filters:
            x = keras.layers.Conv1D(filters, 1, padding="same")(x)
        return keras.layers.Add()([x, y])

    x = inputs
    for i in range(3):
        x = tcn_block(x, filters=64, kernel_size=3, dilation_rate=2**i)

    # Simple self-attention
    q = keras.layers.Dense(64)(x)
    k = keras.layers.Dense(64)(x)
    v = keras.layers.Dense(64)(x)
    attn_scores = keras.layers.Lambda(lambda z: tf.matmul(z[0], z[1], transpose_b=True) / tf.math.sqrt(tf.cast(64, tf.float32)))([q,k])
    attn_weights = keras.layers.Softmax(axis=-1)(attn_scores)
    attn_out = keras.layers.Lambda(lambda z: tf.matmul(z[0], z[1]))([attn_weights, v])

    # BiGRU stack
    g1 = keras.layers.Bidirectional(keras.layers.GRU(128, return_sequences=True))(attn_out)
    g2 = keras.layers.Bidirectional(keras.layers.GRU(64))(g1)

    d = keras.layers.Dense(128, activation="relu")(g2)
    d = keras.layers.Dropout(0.3)(d)
    d = keras.layers.Dense(64, activation="relu")(d)
    outputs = keras.layers.Dense(1)(d)

    model = keras.Model(inputs, outputs)
    model.compile(optimizer=keras.optimizers.Adam(1e-3), loss="mse", metrics=["mae"])
    return model

def train_dl_model(model, X_train, y_train, X_test, y_test, save_prefix="run1"):
    from tensorflow import keras

    ckpt = keras.callbacks.ModelCheckpoint(f"results/models/{save_prefix}_dl.keras",
                                           save_best_only=True, monitor="val_loss", mode="min")
    es = keras.callbacks.EarlyStopping(monitor="val_loss", patience=8, restore_best_weights=True)
    rl = keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=4, min_lr=1e-6)

    hist = model.fit(
        X_train, y_train,
        validation_split=0.2,
        epochs=50,
        batch_size=32,
        callbacks=[ckpt, es, rl],
        verbose=1
    )

    preds = model.predict(X_test).reshape(-1)
    mse = mean_squared_error(y_test, preds)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, preds)
    r2  = r2_score(y_test, preds)

    metrics = {"MSE": mse, "RMSE": rmse, "MAE": mae, "R2": r2}
    return hist.history, preds, metrics
