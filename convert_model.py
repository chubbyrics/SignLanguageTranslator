import tensorflow as tf
import os

def convert_model():
    try:
        # Load your existing model
        print("Loading your existing model...")
        model = tf.keras.models.load_model('Improved_SignLanguage_Model', compile=False)
        
        # Save as .h5 format
        print("Converting to .h5 format...")
        model.save('Improved_SignLanguage_Model.h5')
        
        # Also save as .keras format (new Keras 3 format)
        print("Converting to .keras format...")
        model.save('Improved_SignLanguage_Model.keras')
        
        print("✅ Conversion successful!")
        print("Created files:")
        print("- Improved_SignLanguage_Model.h5")
        print("- Improved_SignLanguage_Model.keras")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        print("\nTrying alternative conversion methods...")
        alternative_conversion()

def alternative_conversion():
    """Alternative method if the first one fails"""
    try:
        # Try loading with custom objects if needed
        model = tf.keras.models.load_model(
            'Improved_SignLanguage_Model', 
            compile=False,
            custom_objects={}
        )
        model.save('Improved_SignLanguage_Model.h5')
        print("✅ Alternative conversion successful!")
    except Exception as e:
        print(f"❌ Alternative conversion failed: {e}")
        print("\nPlease try the manual fix below.")

if __name__ == "__main__":
    convert_model()