# Main2 autoencoder model (with ResNet CNN)
# Definition of residual block
def residual_block(x, filters, kernel_size=3, stride=1):
    shortcut = x  # Save original input

    # First convolutional layer
    x = Conv1D(filters, kernel_size, strides=stride, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    # Second convolutional layer
    x = Conv1D(filters, kernel_size, strides=stride, padding='same')(x)
    x = BatchNormalization()(x)

    # Adjust shortcut channel number if different from x after convolution
    if shortcut.shape[-1] != filters:
        shortcut = Conv1D(filters, 1, strides=stride, padding='same')(shortcut)

    # Add shortcut to x after convolution
    x = Add()([shortcut, x])
    x = Activation('relu')(x)

    return x

# Build ResNet autoencoder model
def build_resnet_autoencoder(input_shape, encoding_dim=64):
    input_data = Input(shape=input_shape)

    # Encoder part
    x = Conv1D(32, 3, activation='relu', padding='same')(input_data)
    x = BatchNormalization()(x)
    x = MaxPooling1D(2)(x)

    # Residual Block 1
    x = residual_block(x, 32)
    x = MaxPooling1D(2)(x)

    # Residual Block 2
    x = residual_block(x, 64)
    x = MaxPooling1D(2)(x)

    x = Flatten()(x)
    encoded = Dense(encoding_dim, activation='relu')(x)

    # Decoder part
    flattened_shape = (input_shape[0] // 8) * 64  # Calculate flattened shape
    x = Dense(flattened_shape, activation='relu')(encoded)
    x = Reshape((input_shape[0] // 8, 64))(x)
    x = Conv1D(64, 3, activation='relu', padding='same')(x)
    x = UpSampling1D(2)(x)
    x = Conv1D(32, 3, activation='relu', padding='same')(x)
    x = UpSampling1D(2)(x)

    # Ensure output length matches input length
    current_length = x.shape[1]
    target_length = input_shape[0]

    if current_length < target_length:
        x = tf.keras.layers.ZeroPadding1D((0, target_length - current_length))(x)
    elif current_length > target_length:
        x = tf.keras.layers.Cropping1D((0, current_length - target_length))(x)

    decoded = Conv1D(input_shape[1], 3, activation='sigmoid', padding='same')(x)

    autoencoder = Model(input_data, decoded)
    encoder = Model(input_data, encoded)
    autoencoder.compile(optimizer='adam', loss='mse')

    return autoencoder, encoder

# Plot training history
def plot_training_history(history):
    plt.figure(figsize=(10, 6))
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Training and Validation Loss', fontsize=16)
    plt.xlabel('Epochs', fontsize=14)
    plt.ylabel('Loss', fontsize=14)
    plt.legend()
    plt.grid(True)
    plt.show()

# Main program: Train ResNet autoencoder and save model
def main2(encoding_dim=64, batch_size=32, epochs=50):
    base_dir = "E:\japan"

    # Load preprocessed data from disk
    try:
        freq_features = np.load(os.path.join(base_dir, "freq_features.npy"))
        if freq_features is None or len(freq_features) == 0:
            raise ValueError("Frequency features are empty.")
    except Exception as e:
        print(f"Error loading frequency features: {e}")
        return

    # Data standardization
    scaler = StandardScaler()
    freq_features = scaler.fit_transform(freq_features.reshape(-1, freq_features.shape[-1])).reshape(freq_features.shape)

    # Build and train ResNet autoencoder model
    input_shape = freq_features.shape[1:]  # (Time series length, number of features)
    autoencoder, encoder = build_resnet_autoencoder(input_shape, encoding_dim)

    # Save the best model and dynamically adjust learning rate
    checkpoint = tf.keras.callbacks.ModelCheckpoint(os.path.join(base_dir, "best_autoencoder.h5"), save_best_only=True, monitor='val_loss', verbose=1)
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=1e-6, verbose=1)
    early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, verbose=1)

    history = autoencoder.fit(
        freq_features, freq_features,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=0.2,
        callbacks=[checkpoint, reduce_lr, early_stop],
        verbose=1
    )

    # Plot training history
    plot_training_history(history)

    # Save encoder model
    encoder.save(os.path.join(base_dir, "encoder_model.h5"))
    print("Encoder model has been saved to disk")

if __name__ == "__main__":
    main2()
