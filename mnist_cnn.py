import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, callbacks # pyright: ignore[reportMissingImports]
from tensorflow.keras.datasets import mnist # pyright: ignore[reportMissingImports]
from tensorflow.keras.utils import to_categorical # pyright: ignore[reportMissingImports]
import os


np.random.seed(42)
tf.random.set_seed(42)
os.makedirs('results', exist_ok=True)

print("="*70)
print("MNIST CNN - STARTING")
print("="*70)

# ====================
# 1. LOAD DATA
# ====================
print("\n[1/8] Loading MNIST data...")
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# Preprocess
X_train = X_train.astype('float32') / 255.0
X_test = X_test.astype('float32') / 255.0
X_train = X_train.reshape(-1, 28, 28, 1)
X_test = X_test.reshape(-1, 28, 28, 1)
y_train_cat = to_categorical(y_train, 10)
y_test_cat = to_categorical(y_test, 10)

print(f" Training samples: {X_train.shape[0]}")
print(f" Test samples: {X_test.shape[0]}")

# ====================
# 2. BUILD MODEL
# ====================
print("\n[2/8] Building CNN model...")

model = models.Sequential([
    # Conv Block 1
    layers.Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(28, 28, 1)),
    layers.BatchNormalization(),
    layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),
    layers.Dropout(0.25),
    
    # Conv Block 2
    layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
    layers.BatchNormalization(),
    layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),
    layers.Dropout(0.25),
    
    # Conv Block 3
    layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
    layers.BatchNormalization(),
    layers.Dropout(0.3),
    
    # Fully Connected
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.BatchNormalization(),
    layers.Dropout(0.5),
    layers.Dense(10, activation='softmax')
])

print("\n" + "="*70)
model.summary()
print("="*70)

# ====================
# 3. COMPILE MODEL
# ====================
print("\n[3/8] Compiling model...")
model.compile(
    optimizer=keras.optimizers.Adam(0.001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)
print(" Optimizer: Adam (lr=0.001)")
print(" Loss: Categorical Cross-Entropy")

# ====================
# 4. TRAIN MODEL
# ====================
print("\n[4/8] Training model...")

early_stop = callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
reduce_lr = callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-7)

history = model.fit(
    X_train, y_train_cat,
    batch_size=128,
    epochs=50,
    validation_split=0.1,
    callbacks=[early_stop, reduce_lr],
    verbose=1
)

# ====================
# 5. EVALUATE MODEL
# ====================
print("\n[5/8] Evaluating model...")
test_loss, test_acc = model.evaluate(X_test, y_test_cat, verbose=0)
y_pred_probs = model.predict(X_test, verbose=0)
y_pred = np.argmax(y_pred_probs, axis=1)

print("\n" + "="*70)
print("RESULTS")
print("="*70)
print(f" Test Accuracy: {test_acc*100:.2f}%")
print(f" Test Loss: {test_loss:.4f}")
print(f" Total Parameters: {model.count_params():,}")
print(f" Epochs Trained: {len(history.history['accuracy'])}")

train_acc = history.history['accuracy'][-1]
val_acc = history.history['val_accuracy'][-1]
gap = (train_acc - val_acc) * 100
print(f" Training Accuracy: {train_acc*100:.2f}%")
print(f" Validation Accuracy: {val_acc*100:.2f}%")
print(f" Accuracy Gap: {gap:.2f}%")

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# ====================
# 6. SAVE VISUALIZATIONS
# ====================
print("\n[6/8] Creating visualizations...")

# Training History
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
axes[0].plot(history.history['accuracy'], label='Training', linewidth=2)
axes[0].plot(history.history['val_accuracy'], label='Validation', linewidth=2)
axes[0].set_title('Model Accuracy', fontsize=14, fontweight='bold')
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Accuracy')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

axes[1].plot(history.history['loss'], label='Training', linewidth=2)
axes[1].plot(history.history['val_loss'], label='Validation', linewidth=2)
axes[1].set_title('Model Loss', fontsize=14, fontweight='bold')
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('Loss')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('results/training_history.png', dpi=300, bbox_inches='tight')
plt.close()


# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=range(10), yticklabels=range(10))
plt.title('Confusion Matrix', fontsize=16, fontweight='bold')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.tight_layout()
plt.savefig('results/confusion_matrix.png', dpi=300, bbox_inches='tight')
plt.close()


# Correct Predictions
correct_idx = np.where(y_pred == y_test)[0][:10]
fig, axes = plt.subplots(2, 5, figsize=(15, 6))
fig.suptitle('Correct Predictions', fontsize=16, fontweight='bold')
for i, ax in enumerate(axes.flat):
    idx = correct_idx[i]
    ax.imshow(X_test[idx].squeeze(), cmap='gray')
    ax.set_title(f'True: {y_test[idx]}, Pred: {y_pred[idx]}\nConf: {y_pred_probs[idx][y_pred[idx]]*100:.1f}%')
    ax.axis('off')
plt.tight_layout()
plt.savefig('results/correct_predictions.png', dpi=300, bbox_inches='tight')
plt.close()


# Incorrect Predictions
incorrect_idx = np.where(y_pred != y_test)[0][:10]
fig, axes = plt.subplots(2, 5, figsize=(15, 6))
fig.suptitle('Incorrect Predictions', fontsize=16, fontweight='bold', color='red')
for i, ax in enumerate(axes.flat):
    if i < len(incorrect_idx):
        idx = incorrect_idx[i]
        ax.imshow(X_test[idx].squeeze(), cmap='gray')
        ax.set_title(f'True: {y_test[idx]}, Pred: {y_pred[idx]}\nConf: {y_pred_probs[idx][y_pred[idx]]*100:.1f}%', color='red')
        ax.axis('off')
plt.tight_layout()
plt.savefig('results/incorrect_predictions.png', dpi=300, bbox_inches='tight')
plt.close()


# ====================
# 7. VISUALIZE FILTERS
# ====================
print("\n[7/8] Visualizing filters.")

# Get first conv layer
first_conv = model.layers[0]
filters, biases = first_conv.get_weights()
f_min, f_max = filters.min(), filters.max()
filters_norm = (filters - f_min) / (f_max - f_min)

fig, axes = plt.subplots(4, 8, figsize=(16, 8))
fig.suptitle('First Layer Convolutional Filters', fontsize=16, fontweight='bold')
for i, ax in enumerate(axes.flat):
    if i < 32:
        ax.imshow(filters_norm[:, :, 0, i], cmap='viridis')
        ax.set_title(f'Filter {i+1}', fontsize=8)
        ax.axis('off')
plt.tight_layout()
plt.savefig('results/conv_filters.png', dpi=300, bbox_inches='tight')
plt.close()


# Feature Maps
layer_outputs = [layer.output for layer in model.layers if 'conv' in layer.name]
activation_model = models.Model(inputs=model.input, outputs=layer_outputs[0])
activations = activation_model.predict(X_test[0:1], verbose=0)

fig, axes = plt.subplots(4, 8, figsize=(16, 8))
fig.suptitle('Feature Maps - First Conv Layer', fontsize=16, fontweight='bold')
for i, ax in enumerate(axes.flat):
    if i < 32:
        ax.imshow(activations[0, :, :, i], cmap='viridis')
        ax.set_title(f'Map {i+1}', fontsize=8)
        ax.axis('off')
plt.tight_layout()
plt.savefig('results/feature_maps.png', dpi=300, bbox_inches='tight')
plt.close()


# ====================
# 8. SAVE MODEL
# ====================

model.save('results/mnist_cnn_model.h5')


# ====================
# FINAL SUMMARY
# ====================
print("\n" + "="*70)
print("="*70)
print("\nRESULTS:")
print(f"  • Test Accuracy: {test_acc*100:.2f}%")
print(f"  • Test Loss: {test_loss:.4f}")
print(f"  • Total Parameters: {model.count_params():,}")
print(f"  • Epochs Trained: {len(history.history['accuracy'])}")
print(f"  • Training Accuracy: {train_acc*100:.2f}%")
print(f"  • Validation Accuracy: {val_acc*100:.2f}%")
print(f"  • Accuracy Gap: {gap:.2f}%")
print(f"\n All visualizations saved in 'results/' folder")
print(f" 6 image files ready for your report")
print("="*70)