"""
Script to convert PyTorch DeepInfant model to TensorFlow/TFLite format
This is useful if you have an existing trained PyTorch model that you want to convert.
"""

import torch
import tensorflow as tf
import numpy as np
import json
from pathlib import Path
from train import DeepInfantModel  # PyTorch model
from train_tensorflow import create_deepinfant_model  # TensorFlow model

def extract_pytorch_weights(pytorch_model_path):
    """
    Extract weights from PyTorch model
    """
    device = torch.device('cpu')  # Use CPU for conversion
    pytorch_model = DeepInfantModel()
    pytorch_model.load_state_dict(torch.load(pytorch_model_path, map_location=device))
    pytorch_model.eval()
    
    weights = {}
    for name, param in pytorch_model.named_parameters():
        weights[name] = param.detach().numpy()
    
    return weights, pytorch_model

def create_sample_input():
    """
    Create a sample input for model conversion testing
    """
    # Sample mel spectrogram shape: (batch, freq_bins, time_steps)
    sample_input = np.random.randn(1, 80, 437, 1).astype(np.float32)
    return sample_input

def convert_model_architecture(pytorch_weights, input_shape=(80, 437)):
    """
    Create TensorFlow model with equivalent architecture and transfer weights where possible
    """
    # Create TensorFlow model
    tf_model = create_deepinfant_model(input_shape=input_shape, num_classes=9)
    
    print("Created TensorFlow model architecture")
    print("Note: Weight transfer is complex due to architectural differences.")
    print("It's recommended to retrain the model using train_tensorflow.py")
    
    return tf_model

def test_model_equivalence(pytorch_model, tf_model, sample_input):
    """
    Test if models produce similar outputs (for validation)
    """
    # Convert input for PyTorch (batch, channels, freq, time)
    pytorch_input = torch.FloatTensor(sample_input.transpose(0, 3, 1, 2))
    
    # PyTorch forward pass
    with torch.no_grad():
        pytorch_output = pytorch_model(pytorch_input)
        pytorch_probs = torch.nn.functional.softmax(pytorch_output, dim=1)
    
    # TensorFlow forward pass
    tf_output = tf_model(sample_input)
    
    print(f"PyTorch output shape: {pytorch_output.shape}")
    print(f"TensorFlow output shape: {tf_output.shape}")
    
    # Compare outputs (they won't be identical due to different weights)
    pytorch_pred = torch.argmax(pytorch_probs, dim=1).item()
    tf_pred = np.argmax(tf_output[0])
    
    print(f"PyTorch prediction: {pytorch_pred}")
    print(f"TensorFlow prediction: {tf_pred}")
    
    return pytorch_pred, tf_pred

def convert_to_tflite_from_pytorch(pytorch_model_path, output_path="converted_deepinfant"):
    """
    Main conversion function
    """
    print(f"Converting PyTorch model from {pytorch_model_path}")
    
    # Extract PyTorch weights
    try:
        pytorch_weights, pytorch_model = extract_pytorch_weights(pytorch_model_path)
        print("Successfully loaded PyTorch model")
    except Exception as e:
        print(f"Error loading PyTorch model: {e}")
        return None
    
    # Create TensorFlow model
    tf_model = convert_model_architecture(pytorch_weights)
    
    # Compile the model
    tf_model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Test with sample input
    sample_input = create_sample_input()
    pytorch_pred, tf_pred = test_model_equivalence(pytorch_model, tf_model, sample_input)
    
    # Save TensorFlow model
    tf_model_path = f"{output_path}.h5"
    tf_model.save(tf_model_path)
    print(f"TensorFlow model saved to: {tf_model_path}")
    
    # Convert to TFLite
    converter = tf.lite.TFLiteConverter.from_keras_model(tf_model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_model = converter.convert()
    
    # Save TFLite model
    tflite_path = f"{output_path}.tflite"
    with open(tflite_path, 'wb') as f:
        f.write(tflite_model)
    print(f"TFLite model saved to: {tflite_path}")
    
    # Save model info
    model_info = {
        'converted_from': pytorch_model_path,
        'tensorflow_model': tf_model_path,
        'tflite_model': tflite_path,
        'input_shape': tf_model.input_shape,
        'output_shape': tf_model.output_shape,
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
        },
        'note': 'Model converted from PyTorch. Weights are not transferred - model needs retraining.'
    }
    
    info_path = f"{output_path}_info.json"
    with open(info_path, 'w') as f:
        json.dump(model_info, f, indent=2)
    print(f"Model info saved to: {info_path}")
    
    return tflite_path, tf_model_path, info_path

def main():
    """
    Example usage
    """
    pytorch_model_path = "deepinfant.pth"
    
    if not Path(pytorch_model_path).exists():
        print(f"PyTorch model {pytorch_model_path} not found!")
        print("\nThis script converts existing PyTorch models to TensorFlow/TFLite.")
        print("If you don't have a trained PyTorch model, use train_tensorflow.py instead.")
        print("\nNote: This conversion creates the model architecture but doesn't transfer weights.")
        print("The converted model will need to be retrained using train_tensorflow.py")
        return
    
    print("Converting PyTorch model to TensorFlow/TFLite...")
    print("=" * 50)
    
    try:
        result = convert_to_tflite_from_pytorch(
            pytorch_model_path, 
            "converted_deepinfant"
        )
        
        if result is not None:
            tflite_path, tf_path, info_path = result
            print("\n" + "=" * 50)
            print("Conversion completed successfully!")
            print(f"TensorFlow model: {tf_path}")
            print(f"TFLite model: {tflite_path}")
            print(f"Model info: {info_path}")
            print("\nIMPORTANT: The converted model needs to be retrained using train_tensorflow.py")
            print("This script only converts the architecture, not the learned weights.")
        else:
            print("Conversion failed - see error messages above.")
        
    except Exception as e:
        print(f"Conversion failed: {e}")
        print("\nRecommendation: Train a new model using train_tensorflow.py instead.")

if __name__ == "__main__":
    main()