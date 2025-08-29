import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm
import glob
import cv2
from scipy import stats
import random
from sklearn.model_selection import train_test_split
import tensorflow as tf



def get_local_image_path(image_url):
    """
    Convert S3 URL to local file path based on the mapping rules.
    
    Args:
        image_url: S3 URL from the dataframe
        
    Returns:
        str: Local file path
    """
    
    GLOBAL_PATH = "Data/Image_Data"
    
    # Extract the part after 'EXPERIMENT_HUMAN_CORRELATION'
    if 'EXPERIMENT_HUMAN_CORRELATION' not in image_url:
        raise ValueError(f"URL doesn't contain expected path: {image_url}")
    
    # Get the part after 'EXPERIMENT_HUMAN_CORRELATION/'
    path_part = image_url.split('EXPERIMENT_HUMAN_CORRELATION/')[1]
    
    # Apply mapping rules
    if path_part.startswith('clean_images/'):
        # clean_images/raised_hand_clean_1.jpg -> clean_images/raised_hand_clean_1.jpg
        local_path = os.path.join(GLOBAL_PATH, path_part)
        
    elif path_part.startswith('level1/'):
        # level1/fire_noisy_level1_2.jpg -> noisy_images/level1/fire_noisy_level1_2.jpg
        filename = path_part.split('level1/')[1]
        local_path = os.path.join(GLOBAL_PATH, f"noisy_images/level1/{filename}")
        
    elif path_part.startswith('level2/'):
        # level2/sailboat_noisy_level2_2.jpg -> noisy_images/level2/sailboat_noisy_level2_2.jpg
        filename = path_part.split('level2/')[1]
        local_path = os.path.join(GLOBAL_PATH, f"noisy_images/level2/{filename}")
        
    elif path_part.startswith('level3/'):
        # level3/something_noisy_level3_1.jpg -> noisy_images/level3/something_noisy_level3_1.jpg
        filename = path_part.split('level3/')[1]
        local_path = os.path.join(GLOBAL_PATH, f"noisy_images/level3/{filename}")

    elif path_part.startswith('deepdream/'):
        # deepdream/neuron_967_gradient.jpg -> deepdream/neuron_967/neuron_967_gradient.jpg
        filename = path_part.split('deepdream/')[1]
        # Extract neuron number from filename (e.g., neuron_967_gradient.jpg -> 967)
        neuron_num = filename.split('_')[1]  # neuron_967_gradient.jpg -> 967
        local_path = os.path.join(GLOBAL_PATH, f"deepdream/neuron_{neuron_num}/{filename}")

    else:
        raise ValueError(f"Unknown path pattern: {path_part}")
    
    return local_path




####################################################################################
#########################################
########## SELECTIVITY HELPERS ##########
#########################################
####################################################################################

def preprocess_image(image_path, target_size=(288, 288)):
    """
    Preprocess an image for CLIP model input
    """
    try:
        # Load and resize image
        img = Image.open(image_path)
        img = img.resize(target_size, Image.LANCZOS)
        img = img.convert('RGB')
        
        # Convert to numpy array and normalize
        img_array = np.array(img).astype(np.float32)
        
        return img_array
    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        return None

def collect_random_images(imagenet_path, n_samples=400, exclude_paths=None):
    """
    Collect random images from ImageNet
    """
    # Find all image files in ImageNet (using a subset of categories for efficiency)
    all_categories = os.listdir(imagenet_path)
    random_categories = random.sample(all_categories, min(50, len(all_categories)))
    
    image_paths = []
    for category in random_categories:
        category_path = os.path.join(imagenet_path, category)
        if os.path.isdir(category_path):
            image_paths.extend(glob.glob(os.path.join(category_path, "*.JPEG")))
    
    print(f"Found {len(image_paths)} candidate images in ImageNet subset")
    
    # Exclude specified paths
    if exclude_paths:
        exclude_set = set(exclude_paths)
        image_paths = [p for p in image_paths if p not in exclude_set]
    
    # Randomly sample
    return random.sample(image_paths, n_samples)

def get_unit_activations(image_paths, neuron_idx, model, batch_size=16):
    """
    Get activations of a specific neuron for a list of images
    """
    # Create TensorFlow graph for batch processing
    tf.reset_default_graph()
    with tf.Graph().as_default() as g:
        with tf.Session(config=tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True))) as sess:
            # Input placeholder
            input_ph = tf.placeholder(tf.float32, shape=[None, 288, 288, 3])
            
            # Import model
            tf.import_graph_def(
                model.graph_def, 
                {model.input_name: input_ph},
                name='clip'
            )
            
            # Get reference to the desired layer
            layer_name = [n.name for n in model.graph_def.node if "image_block_4" in n.name][-1]
            layer_tensor = g.get_tensor_by_name(f"clip/{layer_name}:0")
            
            # Process images in batches
            all_activations = []
            valid_paths = []
            
            for start_idx in tqdm(range(0, len(image_paths), batch_size), desc="Processing images"):
                batch_paths = image_paths[start_idx:start_idx + batch_size]
                batch_images = []
                batch_valid_paths = []
                
                for img_path in batch_paths:
                    img_array = preprocess_image(img_path)
                    if img_array is not None:
                        batch_images.append(img_array)
                        batch_valid_paths.append(img_path)
                
                if not batch_images:
                    continue
                
                # Convert to numpy batch
                batch_array = np.stack(batch_images)
                
                # Run forward pass
                batch_activations = sess.run(layer_tensor, feed_dict={input_ph: batch_array})
                
                # Process activations (average over spatial dimensions)
                if len(batch_activations.shape) == 4:
                    if batch_activations.shape[1] > batch_activations.shape[2]:
                        # Format is (batch_size, channels, height, width)
                        batch_activations = np.mean(batch_activations, axis=(2, 3))
                    else:
                        # Format is (batch_size, height, width, channels)
                        batch_activations = np.mean(batch_activations, axis=(1, 2))
                
                # Extract activations for the specific neuron
                neuron_activations = batch_activations[:, neuron_idx]
                all_activations.extend(neuron_activations)
                valid_paths.extend(batch_valid_paths)
            
            return np.array(all_activations), valid_paths
        
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def compute_selectivity_score(concept_activations, random_activations):
    """
    Computes the sigmoid-bounded Selectivity Score from raw activations.
    """
    mu_X = np.mean(concept_activations)
    mu_notX = np.mean(random_activations)
    std_X = np.std(concept_activations)
    std_notX = np.std(random_activations)
    
    pooled_std = np.sqrt((std_X ** 2 + std_notX ** 2) / 2)
    if pooled_std == 0:
        return 0.5  # Neutral score if no variance

    effect_size = (mu_X - mu_notX) / pooled_std
    return sigmoid(effect_size)

def statistical_testing(concept_activations, random_activations, n_iterations=1000, alpha=0.05):
    # Match sizes
    min_size = min(len(concept_activations), len(random_activations))
    concept_activations = concept_activations[:min_size]
    random_activations = random_activations[:min_size]

    # Compute Selectivity Score
    selectivity_score = compute_selectivity_score(concept_activations, random_activations)

    # Basic stats
    concept_mean = np.mean(concept_activations)
    random_mean = np.mean(random_activations)
    concept_std = np.std(concept_activations)
    random_std = np.std(random_activations)
    
    pooled_std = np.sqrt((concept_std**2 + random_std**2) / 2)
    effect_size = (concept_mean - random_mean) / pooled_std if pooled_std > 0 else 0

    # Non-parametric test
    u_stat, p_value = stats.mannwhitneyu(concept_activations, random_activations, alternative='greater')
    
    # Bootstrapping
    bootstrap_diffs = [
        np.mean(np.random.choice(concept_activations, size=min_size, replace=True)) -
        np.mean(np.random.choice(random_activations, size=min_size, replace=True))
        for _ in range(n_iterations)
    ]
    conf_interval = np.percentile(bootstrap_diffs, [2.5, 97.5])
    
    return {
        'selectivity_score': selectivity_score,
        'concept_mean': concept_mean,
        'random_mean': random_mean,
        'concept_std': concept_std,
        'random_std': random_std,
        'effect_size': effect_size,
        'p_value': p_value,
        'significant': p_value < alpha,
        'conf_interval': conf_interval,
        'histogram_data': {
            'concept': concept_activations,
            'random': random_activations
        }
    }

####################################################################################
#########################################
########## CAUSALITY HELPERS ############
#########################################
####################################################################################


####################################################################################
#########################################
########## ROUBUSTNESS HELPERS ##########
#########################################
####################################################################################


####################################################################################
#########################################
########## HUMAN ALIGN. HELPERS #########
#########################################
####################################################################################