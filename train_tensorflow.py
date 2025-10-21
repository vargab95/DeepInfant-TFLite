import os
import tensorflow as tf
import numpy as np
import pandas as pd
import librosa
from pathlib import Path
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
import json

class DeepInfantDataset:
    def __init__(self, data_dir, transform=False):
        self.data_dir = Path(data_dir)
        self.transform = transform
        self.samples = []
        self.labels = []
        
        # Updated label mapping based on new classes
        self.label_map = {
            'bp': 0,  # belly pain
            'bu': 1,  # burping
            'ch': 2,  # cold/hot
            'dc': 3,  # discomfort
            'hu': 4,  # hungry
            'lo': 5,  # lonely
            'sc': 6,  # scared
            'ti': 7,  # tired
            'un': 8,  # unknown
        }
        
        # Load metadata if available
        metadata_file = Path(data_dir).parent / 'metadata.csv'
        if metadata_file.exists():
            self._load_from_metadata(metadata_file)
        else:
            self._load_dataset()
    
    def _load_from_metadata(self, metadata_file):
        df = pd.read_csv(metadata_file)
        for _, row in df.iterrows():
            if row['split'] == self.data_dir.name:  # 'train' or 'test'
                audio_path = self.data_dir / row['filename']
                if audio_path.exists():
                    self.samples.append(str(audio_path))
                    self.labels.append(self.label_map[row['class_code']])
    
    def _load_dataset(self):
        for audio_file in self.data_dir.glob('*.*'):
            if audio_file.suffix in ['.wav', '.caf', '.3gp']:
                # Parse filename for label
                label = audio_file.stem.split('-')[-1][:2]  # Get reason code
                if label in self.label_map:
                    self.samples.append(str(audio_file))
                    self.labels.append(self.label_map[label])
    
    def _process_audio(self, audio_path):
        # Load audio with 16kHz sample rate
        waveform, sample_rate = librosa.load(audio_path, sr=16000)
        
        # Add basic audio augmentation (during training)
        if self.transform:
            # Random time shift (-100ms to 100ms)
            shift = np.random.randint(-1600, 1600)
            if shift > 0:
                waveform = np.pad(waveform, (shift, 0))[:len(waveform)]
            else:
                waveform = np.pad(waveform, (0, -shift))[(-shift):]
            
            # Random noise injection
            if np.random.random() < 0.3:
                noise = np.random.normal(0, 0.005, len(waveform))
                waveform = waveform + noise
        
        # Ensure consistent length (7 seconds)
        target_length = 7 * 16000
        if len(waveform) > target_length:
            waveform = waveform[:target_length]
        else:
            waveform = np.pad(waveform, (0, target_length - len(waveform)))
        
        # Generate mel spectrogram with adjusted parameters
        mel_spec = librosa.feature.melspectrogram(
            y=waveform,
            sr=sample_rate,
            n_fft=1024,  # Reduced from 2048 for better temporal resolution
            hop_length=256,  # Reduced from 512
            n_mels=80,  # Standard for speech/audio
            fmin=20,  # Minimum frequency
            fmax=8000  # Maximum frequency, suitable for infant cries
        )
        
        # Convert to log scale
        mel_spec = librosa.power_to_db(mel_spec, ref=np.max)
        
        # Normalize to [0, 1]
        mel_spec = (mel_spec - mel_spec.min()) / (mel_spec.max() - mel_spec.min())
        
        return mel_spec
    
    def get_data(self):
        X = []
        y = []
        
        for i, audio_path in enumerate(self.samples):
            try:
                mel_spec = self._process_audio(audio_path)
                X.append(mel_spec)
                y.append(self.labels[i])
            except Exception as e:
                print(f"Error processing {audio_path}: {e}")
                continue
        
        return np.array(X), np.array(y)

def create_deepinfant_model(input_shape=(80, 437), num_classes=9):
    """
    Create DeepInfant model architecture in TensorFlow/Keras
    """
    inputs = layers.Input(shape=input_shape + (1,))  # Add channel dimension
    
    # CNN layers with residual connections
    x = layers.Conv2D(64, (3, 3), padding='same', activation='relu')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)
    
    x = layers.Conv2D(128, (3, 3), padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)
    
    x = layers.Conv2D(256, (3, 3), padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)
    
    # Squeeze-and-excitation block
    se = layers.GlobalAveragePooling2D()(x)
    se = layers.Dense(16, activation='relu')(se)
    se = layers.Dense(256, activation='sigmoid')(se)
    se = layers.Reshape((1, 1, 256))(se)
    x = layers.Multiply()([x, se])
    
    # Reshape for LSTM
    shape = tf.shape(x)
    x = layers.Reshape((shape[2], shape[1] * shape[3]))(x)  # (time, features)
    
    # Bi-directional LSTM for temporal modeling
    x = layers.Bidirectional(
        layers.LSTM(512, return_sequences=True, dropout=0.3)
    )(x)
    x = layers.Bidirectional(
        layers.LSTM(512, dropout=0.3)
    )(x)
    
    # Final classification layers
    x = layers.Dense(512, activation='relu')(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    
    model = models.Model(inputs, outputs)
    return model

def data_generator(X, y, batch_size=32, shuffle=True):
    """
    Generator function for training data
    """
    indices = np.arange(len(X))
    
    while True:
        if shuffle:
            np.random.shuffle(indices)
        
        for start_idx in range(0, len(X), batch_size):
            batch_indices = indices[start_idx:start_idx + batch_size]
            batch_X = X[batch_indices]
            batch_y = y[batch_indices]
            
            # Add channel dimension
            batch_X = np.expand_dims(batch_X, -1)
            
            # Convert labels to categorical
            batch_y = tf.keras.utils.to_categorical(batch_y, num_classes=9)
            
            yield batch_X, batch_y

def convert_to_tflite(model, model_path, quantize=True):
    """
    Convert Keras model to TensorFlow Lite format
    """
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    
    if quantize:
        # Enable optimization for smaller model size
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        
        # Optional: Use dynamic range quantization
        converter.target_spec.supported_types = [tf.float16]
    
    tflite_model = converter.convert()
    
    # Save the model
    tflite_path = model_path.replace('.h5', '.tflite')
    with open(tflite_path, 'wb') as f:
        f.write(tflite_model)
    
    print(f"TFLite model saved to: {tflite_path}")
    
    # Save model info
    model_info = {
        'input_shape': model.input_shape,
        'output_shape': model.output_shape,
        'num_classes': 9,
        'label_map': {
            0: 'belly_pain',
            1: 'burping', 
            2: 'cold_hot',
            3: 'discomfort',
            4: 'hungry',
            5: 'lonely',
            6: 'scared',
            7: 'tired',
            8: 'unknown'
        }
    }
    
    info_path = tflite_path.replace('.tflite', '_info.json')
    with open(info_path, 'w') as f:
        json.dump(model_info, f, indent=2)
    
    return tflite_path

def train_model():
    # Set random seeds for reproducibility
    tf.random.set_seed(42)
    np.random.seed(42)
    
    # Create datasets
    print("Loading training data...")
    train_dataset = DeepInfantDataset('processed_dataset/train', transform=True)
    X_train, y_train = train_dataset.get_data()
    
    print("Loading validation data...")
    val_dataset = DeepInfantDataset('processed_dataset/test', transform=False)
    X_val, y_val = val_dataset.get_data()
    
    print(f"Training samples: {len(X_train)}")
    print(f"Validation samples: {len(X_val)}")
    print(f"Input shape: {X_train[0].shape}")
    
    # Create model
    model = create_deepinfant_model(input_shape=X_train[0].shape, num_classes=9)
    
    # Compile model
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Print model summary
    model.summary()
    
    # Create callbacks
    callbacks = [
        ModelCheckpoint(
            'deepinfant_tensorflow.h5',
            monitor='val_accuracy',
            save_best_only=True,
            mode='max',
            verbose=1
        ),
        EarlyStopping(
            monitor='val_accuracy',
            patience=10,
            mode='max',
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-7,
            verbose=1
        )
    ]
    
    # Prepare data generators
    batch_size = 32
    train_gen = data_generator(X_train, y_train, batch_size=batch_size, shuffle=True)
    val_gen = data_generator(X_val, y_val, batch_size=batch_size, shuffle=False)
    
    # Calculate steps
    train_steps = len(X_train) // batch_size
    val_steps = len(X_val) // batch_size
    
    # Train model
    print("Starting training...")
    history = model.fit(
        train_gen,
        steps_per_epoch=train_steps,
        epochs=50,
        validation_data=val_gen,
        validation_steps=val_steps,
        callbacks=callbacks,
        verbose=1
    )
    
    # Load best model
    model.load_weights('deepinfant_tensorflow.h5')
    
    # Convert to TFLite
    print("Converting to TensorFlow Lite...")
    tflite_path = convert_to_tflite(model, 'deepinfant_tensorflow.h5', quantize=True)
    
    # Also create unquantized version
    tflite_path_unquantized = convert_to_tflite(model, 'deepinfant_tensorflow_unquantized.h5', quantize=False)
    
    print(f"Training completed!")
    print(f"Keras model saved to: deepinfant_tensorflow.h5")
    print(f"TFLite model (quantized) saved to: {tflite_path}")
    print(f"TFLite model (unquantized) saved to: {tflite_path_unquantized}")
    
    return model, history

def main():
    # Check for GPU
    print("GPUs available:", tf.config.list_physical_devices('GPU'))
    
    # Train model
    model, history = train_model()
    
    # Print final metrics
    print("\nFinal metrics:")
    print(f"Training accuracy: {history.history['accuracy'][-1]:.4f}")
    print(f"Validation accuracy: {history.history['val_accuracy'][-1]:.4f}")

if __name__ == '__main__':
    main()