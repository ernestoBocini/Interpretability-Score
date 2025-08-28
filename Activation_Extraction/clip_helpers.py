import numpy as np
from PIL import Image

def preprocess_image(img_path):
    """
    Loads an image, resizes to 288x288, converts to float32 np.array.
    """
    try:
        img = Image.open(img_path)
        img = img.resize((288, 288), Image.LANCZOS)
        img = img.convert('RGB')
        img_array = np.array(img).astype(np.float32)
        return img_array
    except Exception as e:
        print(f"Error loading {img_path}: {e}")
        return None