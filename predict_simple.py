#!/usr/bin/env python3
"""
Simple prediction script for DeepInfant without external ML dependencies
This demonstrates prediction using the mock model
"""

import json
import os
import random
from pathlib import Path

class SimpleTFLitePredictor:
    def __init__(self, model_path='deepinfant_tensorflow.tflite'):
        self.model_path = model_path
        
        # Load model info
        info_path = model_path.replace('.tflite', '_info.json')
        if Path(info_path).exists():
            with open(info_path, 'r') as f:
                self.model_info = json.load(f)
                self.classes = self.model_info['classes']
        else:
            # Default classes
            self.classes = [
                'belly_pain', 'burping', 'cold_hot', 'discomfort', 
                'hungry', 'lonely', 'scared', 'tired', 'unknown'
            ]
        
        print(f"✅ Mock TFLite model loaded: {model_path}")
        print(f"📋 Classes: {', '.join(self.classes)}")
    
    def predict(self, audio_path):
        """
        Mock prediction function
        In real implementation, this would process audio and run TFLite inference
        """
        audio_path = Path(audio_path)
        
        if not audio_path.exists():
            return "error", 0.0, [0.0] * len(self.classes)
        
        # Mock prediction based on filename patterns
        filename = audio_path.name.lower()
        
        # Extract reason code from filename if available
        predicted_class = 'unknown'
        confidence = 0.75 + random.uniform(-0.15, 0.20)
        
        if '-hu.' in filename or 'hungry' in filename:
            predicted_class = 'hungry'
            confidence = 0.85 + random.uniform(-0.10, 0.10)
        elif '-bp.' in filename or 'pain' in filename:
            predicted_class = 'belly_pain'
            confidence = 0.82 + random.uniform(-0.10, 0.10)
        elif '-bu.' in filename or 'burp' in filename:
            predicted_class = 'burping'
            confidence = 0.88 + random.uniform(-0.10, 0.10)
        elif '-dc.' in filename or 'discomfort' in filename:
            predicted_class = 'discomfort'
            confidence = 0.79 + random.uniform(-0.10, 0.10)
        elif '-ti.' in filename or 'tired' in filename:
            predicted_class = 'tired'
            confidence = 0.81 + random.uniform(-0.10, 0.10)
        else:
            # Random prediction for demonstration
            predicted_class = random.choice(self.classes)
            confidence = 0.60 + random.uniform(0.05, 0.25)
        
        # Create mock probability distribution
        probabilities = [0.02 + random.uniform(-0.01, 0.03) for _ in self.classes]
        pred_index = self.classes.index(predicted_class)
        probabilities[pred_index] = confidence
        
        # Normalize probabilities
        prob_sum = sum(probabilities)
        probabilities = [p / prob_sum for p in probabilities]
        
        return predicted_class, confidence, probabilities
    
    def predict_batch(self, audio_dir, file_extensions=('.wav', '.caf', '.3gp')):
        """
        Predict on all audio files in a directory
        """
        results = []
        audio_dir = Path(audio_dir)
        
        if not audio_dir.exists():
            print(f"❌ Directory not found: {audio_dir}")
            return results
        
        audio_files = []
        for ext in file_extensions:
            audio_files.extend(list(audio_dir.glob(f'*{ext}')))
            audio_files.extend(list(audio_dir.glob(f'**/*{ext}')))
        
        print(f"🔍 Found {len(audio_files)} audio files")
        
        for audio_file in audio_files[:10]:  # Limit to first 10 for demo
            try:
                label, confidence, _ = self.predict(str(audio_file))
                results.append((audio_file.name, label, confidence))
            except Exception as e:
                print(f"❌ Error processing {audio_file.name}: {e}")
                results.append((audio_file.name, "error", 0.0))
        
        return results
    
    def get_model_info(self):
        """Get model information"""
        if hasattr(self, 'model_info'):
            return self.model_info
        else:
            return {
                'model_path': self.model_path,
                'classes': self.classes,
                'note': 'Mock model for demonstration'
            }

def main():
    """
    Demo of the prediction system
    """
    print("🤖 DeepInfant Simple Prediction Demo")
    print("=" * 40)
    
    # Check if model exists
    model_path = "deepinfant_tensorflow.tflite"
    if not Path(model_path).exists():
        print(f"❌ Model not found: {model_path}")
        print("Please run training first:")
        print("   python3 train_simple.py")
        return
    
    # Initialize predictor
    predictor = SimpleTFLitePredictor(model_path)
    
    # Show model info
    print(f"\n📊 Model Info:")
    info = predictor.get_model_info()
    for key, value in info.items():
        if isinstance(value, list) and len(value) > 3:
            print(f"   {key}: {value[:3]}... ({len(value)} total)")
        else:
            print(f"   {key}: {value}")
    
    # Demo single file prediction
    print(f"\n🎯 Testing single file prediction:")
    test_files = list(Path('Data').glob('**/*.wav'))[:3]
    
    if test_files:
        for test_file in test_files:
            label, confidence, probs = predictor.predict(str(test_file))
            print(f"   {test_file.name}")
            print(f"   → Prediction: {label} (confidence: {confidence:.1%})")
    else:
        print("   No test files found in Data/ directory")
    
    # Demo batch prediction
    print(f"\n📁 Testing batch prediction:")
    results = predictor.predict_batch('Data')
    
    if results:
        print(f"   Results (showing first {len(results)}):")
        for filename, label, confidence in results:
            print(f"   {filename}: {label} ({confidence:.1%})")
    else:
        print("   No audio files found for batch prediction")
    
    print(f"\n✅ Prediction demo completed!")
    print(f"\nUsage in code:")
    print(f"   from predict_simple import SimpleTFLitePredictor")
    print(f"   predictor = SimpleTFLitePredictor('deepinfant_tensorflow.tflite')")
    print(f"   label, confidence, probs = predictor.predict('audio.wav')")

if __name__ == "__main__":
    main()