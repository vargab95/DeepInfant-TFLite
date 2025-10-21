#!/usr/bin/env python3
"""
Setup script for DeepInfant TensorFlow/TFLite version
This script helps users get started with the TensorFlow implementation
"""

import os
import sys
import subprocess
from pathlib import Path

def check_python_version():
    """Check if Python version is compatible"""
    if sys.version_info < (3, 8):
        print("❌ Python 3.8 or higher is required")
        print(f"Current version: {sys.version}")
        return False
    print(f"✅ Python version: {sys.version.split()[0]}")
    return True

def install_dependencies():
    """Install required dependencies"""
    print("\n📦 Installing dependencies...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✅ Dependencies installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install dependencies: {e}")
        return False

def check_data_directory():
    """Check if data directory exists"""
    data_dir = Path("Data")
    if not data_dir.exists():
        print("❌ Data directory not found")
        print("Please ensure your audio data is organized in the Data/ directory")
        print("Example structure:")
        print("  Data/")
        print("    belly_pain/")
        print("      *.wav files")
        print("    hungry/")
        print("      *.wav files")
        print("    ...")
        return False
    
    # Count audio files
    audio_files = list(data_dir.glob("**/*.wav")) + list(data_dir.glob("**/*.caf")) + list(data_dir.glob("**/*.3gp"))
    if len(audio_files) == 0:
        print("❌ No audio files found in Data directory")
        return False
    
    print(f"✅ Found {len(audio_files)} audio files in Data directory")
    return True

def create_processed_dataset():
    """Create processed dataset if it doesn't exist"""
    processed_dir = Path("processed_dataset")
    if not processed_dir.exists():
        print("\n🔄 Creating processed dataset...")
        try:
            subprocess.check_call([sys.executable, "prepare_dataset.py"])
            print("✅ Processed dataset created")
            return True
        except subprocess.CalledProcessError as e:
            print(f"❌ Failed to create processed dataset: {e}")
            return False
    else:
        print("✅ Processed dataset already exists")
        return True

def check_tensorflow():
    """Check if TensorFlow is working"""
    try:
        import tensorflow as tf
        print(f"✅ TensorFlow version: {tf.__version__}")
        
        # Check for GPU
        try:
            gpus = tf.config.list_physical_devices('GPU')
            if gpus:
                print(f"✅ GPU acceleration available: {len(gpus)} GPU(s)")
            else:
                print("ℹ️  No GPU detected, will use CPU (training will be slower)")
        except:
            print("ℹ️  Could not check GPU status")
        
        return True
    except ImportError:
        print("❌ TensorFlow not found")
        return False

def run_training_demo():
    """Ask user if they want to run a training demo"""
    response = input("\n🎯 Would you like to start training a model now? (y/n): ").lower()
    if response == 'y':
        print("\n🚀 Starting training...")
        print("This may take several hours depending on your hardware and dataset size.")
        try:
            subprocess.check_call([sys.executable, "train_tensorflow.py"])
            print("✅ Training completed!")
            return True
        except subprocess.CalledProcessError as e:
            print(f"❌ Training failed: {e}")
            return False
    else:
        print("\n📝 To train a model later, run:")
        print("   python train_tensorflow.py")
        return True

def show_next_steps():
    """Show next steps to the user"""
    print("\n" + "="*50)
    print("🎉 Setup completed!")
    print("="*50)
    print("\nNext steps:")
    print("1. Train a model:")
    print("   python train_tensorflow.py")
    print("\n2. Make predictions:")
    print("   python predict_tflite.py")
    print("\n3. View model info:")
    print("   python -c \"from predict_tflite import InfantCryTFLitePredictor; p = InfantCryTFLitePredictor('deepinfant_tensorflow.tflite'); print(p.get_model_info())\"")
    print("\n4. Integration examples:")
    print("   - See README.md for mobile app integration")
    print("   - Check predict_tflite.py for Python usage examples")
    print("\n📖 Full documentation: README.md")

def main():
    """Main setup function"""
    print("🤖 DeepInfant TensorFlow Setup")
    print("="*40)
    
    # Check Python version
    if not check_python_version():
        return False
    
    # Install dependencies
    if not install_dependencies():
        return False
    
    # Check TensorFlow
    if not check_tensorflow():
        return False
    
    # Check data
    if not check_data_directory():
        print("\n⚠️  Warning: No training data found")
        print("You can still explore the code, but you'll need data to train models")
        show_next_steps()
        return True
    
    # Create processed dataset
    if not create_processed_dataset():
        return False
    
    # Ask about training
    run_training_demo()
    
    # Show next steps
    show_next_steps()
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)