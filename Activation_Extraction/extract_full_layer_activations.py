import os
import numpy as np
import pandas as pd
from tqdm import tqdm
import sys

# TensorFlow setup
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
os.environ['TF_ENABLE_NHWC'] = '1'
os.environ['TF_AVGPOOL_ALLOW_NHWC'] = '1'

from clip_helpers import preprocess_image

# export path where model is (root):
sys.path.append('../')


def extract_clip_layer_activations(
        image_paths,
        output_npz='activations_output.npz',
        output_csv='activations_output.csv',
        layer_name=None,
        batch_size=16
    ):
    """
    Processes all provided images, extracts activations from a specified CLIP layer,
    and saves results as both .npz and .csv.

    Args:
        image_paths: List of image file paths to process.
        output_npz: Filepath to save activations as npz.
        output_csv: Filepath to save activations as csv.
        layer_name: Name of the target CLIP layer. If None, auto-detects 'image_block_4'.
        batch_size: Number of images per batch.

    Returns:
        DataFrame with activations, filenames, and optionally S3 links.
    """

    # Import model
    try:
        from model import CLIPImage
    except ImportError:
        raise ImportError("Could not import CLIPImage. Make sure model.py is in your path.")

    # Setup TensorFlow session
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    # Load CLIP model
    model = CLIPImage()

    # Auto-detect layer if needed
    if layer_name is None:
        # Use last occurrence of 'image_block_4'
        ultimate_layer = [n.name for n in model.graph_def.node if "image_block_4" in n.name][-1]
        layer_name = ultimate_layer

    print(f"Using CLIP layer: {layer_name}")

    tf.reset_default_graph()
    with tf.Graph().as_default() as g:
        with tf.Session(config=config) as sess:
            input_ph = tf.placeholder(tf.float32, shape=[None, 288, 288, 3])

            # Import model into the graph
            tf.import_graph_def(
                model.graph_def,
                {model.input_name: input_ph},
                name='clip'
            )
            layer_tensor = g.get_tensor_by_name(f"clip/{layer_name}:0")

            all_activations = []
            all_filenames = []

            # Batch processing
            for start_idx in tqdm(range(0, len(image_paths), batch_size), desc="Processing images in batches"):
                batch_paths = image_paths[start_idx:start_idx + batch_size]
                batch_images = []
                batch_valid_paths = []

                # Preprocess images
                for img_path in batch_paths:
                    arr = preprocess_image(img_path)
                    if arr is not None:
                        batch_images.append(arr)
                        batch_valid_paths.append(img_path)

                if not batch_images:
                    continue

                batch_array = np.stack(batch_images)  # (batch_size, 288, 288, 3)

                # Run forward pass
                batch_activations = sess.run(layer_tensor, feed_dict={input_ph: batch_array})

                # Average over spatial dims if 4D
                if len(batch_activations.shape) == 4:
                    if batch_activations.shape[1] > batch_activations.shape[2]:
                        # (batch, channels, H, W)
                        batch_activations = np.mean(batch_activations, axis=(2, 3))
                    else:
                        # (batch, H, W, channels)
                        batch_activations = np.mean(batch_activations, axis=(1, 2))

                # Append activations and filenames
                all_activations.extend(batch_activations)
                all_filenames.extend(batch_valid_paths)

    # Convert to np array and save as .npz
    all_activations = np.stack(all_activations)
    np.savez(output_npz, activations=all_activations, filenames=all_filenames, layer_name=layer_name)
    print(f"Saved activations to {output_npz}. Shape: {all_activations.shape}")

    # Create DataFrame
    neuron_cols = [f'activation_{i}' for i in range(all_activations.shape[1])]
    df_acts = pd.DataFrame(all_activations, columns=neuron_cols)
    df_acts['filename'] = [os.path.basename(p) for p in all_filenames]

    # Save as CSV
    df_acts.to_csv(output_csv, index=False)
    print(f"Saved activations CSV to {output_csv}")

    return df_acts