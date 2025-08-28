import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import pandas as pd
import sys


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

# Set TensorFlow environment variables
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true' 
os.environ['TF_ENABLE_NHWC'] = '1'
os.environ['TF_AVGPOOL_ALLOW_NHWC'] = '1'

# export path where model is (root):
sys.path.append('../')

def get_clip_neuron_activation_tf(image_path, neuron_idx=89, show_image=True):
    """
    Get the activation of a specific neuron in the TensorFlow CLIP model for an image.
    
    Args:
        image_path: Path to the image file
        neuron_idx: Index of the neuron to analyze (default: 89)
        show_image: Whether to display the image
    
    Returns:
        Activation value for the specified neuron
    """
    # Check if the image exists
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")
    
    # Import the CLIPImage module from the specified location
    # Make sure the model.py file is in the current directory or in the Python path
    try:
        from model import CLIPImage
    except ImportError:
        raise ImportError("Could not import CLIPImage. Make sure model.py is in your path.")
    
    # Configure TensorFlow session
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    session = tf.Session(config=config)

    # Silence TensorFlow warnings
    if hasattr(tf, 'get_logger'):
        tf.get_logger().setLevel('ERROR')
    
    # Load and preprocess the image
    try:
        # Load and resize image to 288x288 (CLIP TensorFlow input size)
        img = Image.open(image_path)
        img = img.resize((288, 288), Image.LANCZOS)
        img = img.convert('RGB')
        
        # Display the image if requested
        if show_image:
            plt.figure(figsize=(3, 3))
            plt.imshow(img)
            plt.title(f"Image: {os.path.basename(image_path)}")
            plt.axis('off')
            plt.show()
        
        # Convert to numpy array 
        img_array = np.array(img).astype(np.float32)
        
        # Add batch dimension [batch_size=1, height, width, channels]
        img_batch = np.expand_dims(img_array, axis=0)
    except Exception as e:
        raise ValueError(f"Error processing image {image_path}: {e}")
    
    # Load CLIP model
    model = CLIPImage()
    
    # Create TensorFlow graph
    tf.reset_default_graph()
    with tf.Graph().as_default() as g:
        with tf.Session(config=config) as sess:
            # Input placeholder
            input_ph = tf.placeholder(tf.float32, shape=[None, 288, 288, 3])
            
            # Import model
            tf.import_graph_def(
                model.graph_def,
                {model.input_name: input_ph},
                name='clip'
            )
            
            # Get reference to the desired layer - this is the same approach used in your Lucid code
            layer_name = [n.name for n in model.graph_def.node if "image_block_4" in n.name][-1]
            layer_tensor = g.get_tensor_by_name(f"clip/{layer_name}:0")
            
            # Run forward pass
            activations = sess.run(layer_tensor, feed_dict={input_ph: img_batch})
            
            # Process activations (average over spatial dimensions if needed)
            if len(activations.shape) == 4:
                if activations.shape[1] > activations.shape[2]:
                    # Format is (batch_size, channels, height, width)
                    activations = np.mean(activations, axis=(2, 3))
                else:
                    # Format is (batch_size, height, width, channels)
                    activations = np.mean(activations, axis=(1, 2))
            
            # Extract activation for the specific neuron
            neuron_activation = float(activations[0, neuron_idx])
    
    # print(f"TensorFlow CLIP - Neuron {neuron_idx} activation for {os.path.basename(image_path)}: {neuron_activation:.6f}")
    return neuron_activation
