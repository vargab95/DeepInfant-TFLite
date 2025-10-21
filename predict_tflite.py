import tensorflow as tf
import librosa
import numpy as np
import json
from pathlib import Path

class InfantCryTFLitePredictor:
    def __init__(self, model_path='deepinfant_tensorflow.tflite', model_info_path=None):
        """
        Initialize TFLite predictor for infant cry classification
        
        Args:
            model_path: Path to the TFLite model file
            model_info_path: Path to the model info JSON file (optional)
        """
        self.model_path = model_path
        
        # Load TFLite model
        self.interpreter = tf.lite.Interpreter(model_path=model_path)
        self.interpreter.allocate_tensors()
        
        # Get input and output tensors
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        
        # Load model info if available
        if model_info_path is None:
            model_info_path = model_path.replace('.tflite', '_info.json')
        
        if Path(model_info_path).exists():
            with open(model_info_path, 'r') as f:
                self.model_info = json.load(f)
                self.label_map = self.model_info['label_map']
        else:
            # Default label mapping
            self.label_map = {
                '0': 'belly_pain',
                '1': 'burping', 
                '2': 'cold_hot',
                '3': 'discomfort',
                '4': 'hungry',
                '5': 'lonely',
                '6': 'scared',
                '7': 'tired',
                '8': 'unknown'
            }
        
        print(f"Model loaded successfully from {model_path}")
        print(f"Input shape: {self.input_details[0]['shape']}")
        print(f"Output shape: {self.output_details[0]['shape']}")
    
    def _process_audio(self, audio_path):
        """
        Process audio file to mel spectrogram format expected by the model
        """
        # Load audio with 16kHz sample rate
        waveform, sample_rate = librosa.load(audio_path, sr=16000)
        
        # Ensure consistent length (7 seconds)
        target_length = 7 * 16000
        if len(waveform) > target_length:
            waveform = waveform[:target_length]
        else:
            waveform = np.pad(waveform, (0, target_length - len(waveform)))
        
        # Generate mel spectrogram
        mel_spec = librosa.feature.melspectrogram(
            y=waveform,
            sr=sample_rate,
            n_fft=1024,
            hop_length=256,
            n_mels=80,
            fmin=20,
            fmax=8000
        )
        
        # Convert to log scale
        mel_spec = librosa.power_to_db(mel_spec, ref=np.max)
        
        # Normalize to [0, 1]
        mel_spec = (mel_spec - mel_spec.min()) / (mel_spec.max() - mel_spec.min())
        
        # Add batch and channel dimensions
        mel_spec = np.expand_dims(mel_spec, axis=0)  # Add batch dimension
        mel_spec = np.expand_dims(mel_spec, axis=-1)  # Add channel dimension
        
        return mel_spec.astype(np.float32)
    
    def predict(self, audio_path):
        """
        Predict the class of a single audio file
        
        Returns:
            tuple: (predicted_label, confidence, all_probabilities)
        """
        # Process audio
        mel_spec = self._process_audio(audio_path)
        
        # Set input tensor
        self.interpreter.set_tensor(self.input_details[0]['index'], mel_spec)
        
        # Run inference
        self.interpreter.invoke()
        
        # Get output
        output_data = self.interpreter.get_tensor(self.output_details[0]['index'])
        probabilities = output_data[0]  # Remove batch dimension
        
        # Get predicted class and confidence
        pred_class = np.argmax(probabilities)
        confidence = probabilities[pred_class]
        
        predicted_label = self.label_map[str(pred_class)]
        
        return predicted_label, confidence, probabilities
    
    def predict_batch(self, audio_dir, file_extensions=('.wav', '.caf', '.3gp')):
        """
        Predict classes for all audio files in a directory
        
        Returns:
            list: List of tuples (filename, predicted_label, confidence)
        """
        results = []
        audio_dir = Path(audio_dir)
        
        for audio_file in audio_dir.glob('*.*'):
            if audio_file.suffix.lower() in file_extensions:
                try:
                    label, confidence, _ = self.predict(str(audio_file))
                    results.append((audio_file.name, label, confidence))
                except Exception as e:
                    print(f"Error processing {audio_file.name}: {e}")
                    results.append((audio_file.name, "error", 0.0))
        
        return results
    
    def get_model_info(self):
        """
        Get information about the loaded model
        """
        info = {
            'model_path': self.model_path,
            'input_shape': self.input_details[0]['shape'].tolist(),
            'output_shape': self.output_details[0]['shape'].tolist(),
            'input_dtype': str(self.input_details[0]['dtype']),
            'output_dtype': str(self.output_details[0]['dtype']),
            'labels': self.label_map
        }
        return info

def benchmark_model(model_path, test_audio_path, num_runs=100):
    """
    Benchmark the TFLite model performance
    """
    import time
    
    predictor = InfantCryTFLitePredictor(model_path)
    
    # Warm up
    for _ in range(5):
        predictor.predict(test_audio_path)
    
    # Benchmark
    start_time = time.time()
    for _ in range(num_runs):
        predictor.predict(test_audio_path)
    end_time = time.time()
    
    avg_time = (end_time - start_time) / num_runs
    print(f"Average inference time: {avg_time*1000:.2f} ms")
    print(f"Inferences per second: {1/avg_time:.2f}")

def main():
    """
    Example usage of the TFLite predictor
    """
    # Initialize predictor
    model_path = "deepinfant_tensorflow.tflite"
    
    if not Path(model_path).exists():
        print(f"Model file {model_path} not found!")
        print("Please run train_tensorflow.py first to generate the TFLite model.")
        return
    
    predictor = InfantCryTFLitePredictor(model_path)
    
    # Print model info
    print("\nModel Information:")
    info = predictor.get_model_info()
    for key, value in info.items():
        print(f"{key}: {value}")
    
    # Example single file prediction
    print("\nExample usage:")
    print("predictor = InfantCryTFLitePredictor('deepinfant_tensorflow.tflite')")
    print("label, confidence, probs = predictor.predict('path/to/audio.wav')")
    print("print(f'Prediction: {label} (confidence: {confidence:.2%})')")
    
    # Example batch prediction
    print("\nBatch prediction:")
    print("results = predictor.predict_batch('path/to/audio/directory')")
    print("for filename, label, confidence in results:")
    print("    print(f'{filename}: {label} ({confidence:.2%})')")

if __name__ == "__main__":
    main()