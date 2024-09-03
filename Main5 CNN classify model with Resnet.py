# Main5  CNN classify model with Resnet blocks
# Residual Block
def residual_block(x, filters, kernel_size=3, stride=1, l2_reg=0.001):
    shortcut = x  # Original input

    # First convolutional layer
    x = Conv1D(filters, kernel_size, strides=stride, padding='same', kernel_regularizer=l2(l2_reg))(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    # Second convolutional layer
    x = Conv1D(filters, kernel_size, strides=stride, padding='same', kernel_regularizer=l2(l2_reg))(x)
    x = BatchNormalization()(x)

    # If the number of channels in shortcut is different from x after convolution, adjust it with a 1x1 convolution
    if shortcut.shape[-1] != filters:
        shortcut = Conv1D(filters, 1, strides=stride, padding='same', kernel_regularizer=l2(l2_reg))(shortcut)
    
    # Add the shortcut to the output of the convolution
    x = Add()([shortcut, x])
    x = Activation('relu')(x)

    return x

# Build a CNN classifier with ResNet
def build_cnn_resnet_classifier(input_shape, num_classes=10, learning_rate=0.001, dropout_rate=0.5, l2_reg=0.001):
    input_data = Input(shape=input_shape)

    # First convolutional layer
    x = Conv1D(64, 3, activation='relu', padding='same', kernel_regularizer=l2(l2_reg))(input_data)
    x = BatchNormalization()(x)
    x = MaxPooling1D(2)(x)
    
    # Residual Block 1
    x = residual_block(x, 64, l2_reg=l2_reg)
    x = MaxPooling1D(2)(x)

    # Residual Block 2
    x = residual_block(x, 128, l2_reg=l2_reg)
    x = MaxPooling1D(2)(x)
    
    # Residual Block 3
    x = residual_block(x, 256, l2_reg=l2_reg)
    x = MaxPooling1D(2)(x)

    # Fully connected layers
    x = Flatten()(x)
    x = Dense(256, activation='relu', kernel_regularizer=l2(l2_reg))(x)
    x = Dropout(dropout_rate)(x)
    x = Dense(128, activation='relu', kernel_regularizer=l2(l2_reg))(x)
    x = Dropout(dropout_rate)(x)

    # Output layer
    output = Dense(num_classes, activation='softmax')(x)

    model = Model(inputs=input_data, outputs=output)
    model.compile(optimizer=Adam(learning_rate=learning_rate), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

# Plot loss and accuracy curves during training
def plot_training_metrics(history, base_dir):
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Training and Validation Loss', fontsize=16)
    plt.xlabel('Epochs', fontsize=14)
    plt.ylabel('Loss', fontsize=14)
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Training and Validation Accuracy', fontsize=16)
    plt.xlabel('Epochs', fontsize=14)
    plt.ylabel('Accuracy', fontsize=14)
    plt.legend()
    plt.grid(True)

    metrics_file = os.path.join(base_dir, "training_metrics.png")
    plt.savefig(metrics_file)
    plt.show()
    print(f"Training metrics plot saved to: {metrics_file}")

# Cross-validate the model
def cross_validate_model(X_train, y_train, input_shape, num_classes, learning_rate, dropout_rate, l2_reg, n_splits=5):
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    fold_no = 1
    val_accuracies = []

    for train_index, val_index in kf.split(X_train):
        print(f"Training fold {fold_no}...")
        X_train_fold, X_val_fold = X_train[train_index], X_train[val_index]
        y_train_fold, y_val_fold = y_train[train_index], y_train[val_index]

        model = build_cnn_resnet_classifier(input_shape, num_classes, learning_rate, dropout_rate, l2_reg)

        history = model.fit(
            X_train_fold, y_train_fold,
            epochs=10,
            validation_data=(X_val_fold, y_val_fold),
            verbose=0
        )

        val_accuracy = model.evaluate(X_val_fold, y_val_fold, verbose=0)[1]
        val_accuracies.append(val_accuracy)
        print(f"Fold {fold_no} Validation Accuracy: {val_accuracy:.4f}")

        fold_no += 1

    avg_val_accuracy = np.mean(val_accuracies)
    print(f"Average Validation Accuracy across {n_splits} folds: {avg_val_accuracy:.4f}")
    return avg_val_accuracy

# Main program: Train the CNN classifier with ResNet
def main5():
    base_dir = "E:\japan"
    
    # Load preprocessed data from disk
    X_train = np.load(os.path.join(base_dir, "X_train.npy"))
    X_val = np.load(os.path.join(base_dir, "X_val.npy"))
    y_train = np.load(os.path.join(base_dir, "y_train.npy"))
    y_val = np.load(os.path.join(base_dir, "y_val.npy"))

    # Hyperparameters
    input_shape = X_train.shape[1:]
    num_classes = 10
    learning_rate = 0.001
    dropout_rate = 0.5
    l2_reg = 0.001
    epochs = 50
    batch_size = 12

    # Cross-validate the model to evaluate performance
    cross_validate_model(X_train, y_train, input_shape, num_classes, learning_rate, dropout_rate, l2_reg, n_splits=5)

    # Define callbacks: learning rate adjustment, early stopping, and model checkpoint
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=1e-6)
    early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    checkpoint = ModelCheckpoint(filepath=os.path.join(base_dir, "best_cnn_classifier.h5"), monitor='val_accuracy', save_best_only=True, verbose=1)

    # Build and train the model
    model = build_cnn_resnet_classifier(input_shape, num_classes, learning_rate, dropout_rate, l2_reg)
    history = model.fit(
        X_train, y_train,
        epochs=epochs,
        validation_data=(X_val, y_val),
        callbacks=[reduce_lr, early_stop, checkpoint],
        batch_size=batch_size,
        verbose=1
    )

    # Plot loss and accuracy curves during training
    plot_training_metrics(history, base_dir)

    # Save the final model
    model.save(os.path.join(base_dir, "cnn_classifier_final.h5"))
    np.save(os.path.join(base_dir, "training_history.npy"), history.history)
    print("Final model and training history saved to disk")

if __name__ == "__main__":
    main5()
