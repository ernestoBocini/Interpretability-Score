import os
import numpy as np
import tensorflow as tf
from PIL import Image

import sys
sys.path.append('../')
from model import CLIPImage

# Suppress TensorFlow logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

def extract_intermediate_activations(image_path, debug=False):
    """
    Extract activations from the graph up to the specified intervention point.
    This is the first step of the intervention pipeline.
    """
    
    if debug:
        print("Extracting intermediate activations")
    
    # Load and preprocess image (same as before)
    try:
        img = Image.open(image_path)
        img = img.resize((288, 288), Image.LANCZOS)
        img = img.convert('RGB')
        img_array = np.array(img).astype(np.float32)
        img_batch = np.expand_dims(img_array, axis=0)
        
        if debug:
            print(f"Image loaded and preprocessed: {img_batch.shape}")
    except Exception as e:
        raise ValueError(f"Error processing image: {e}")
    
    # Load CLIP model
    model = CLIPImage()
    
    # Configure TensorFlow session
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.log_device_placement = False
    
    # Create graph and get activations UP TO intervention point
    tf.reset_default_graph()
    with tf.Graph().as_default() as g:
        with tf.Session(config=config) as sess:
            # Input placeholder
            input_ph = tf.placeholder(tf.float32, shape=[None, 288, 288, 3])
            
            # Import the complete model (we'll only use part of it)
            tf.import_graph_def(
                model.graph_def,
                {model.input_name: input_ph},
                name='clip'
            )
            
            # Get our target intervention layer
            target_layer_name = 'image_block_4/5/Relu_2'
            intervention_tensor = g.get_tensor_by_name(f"clip/{target_layer_name}:0")
            
            if debug:
                print(f"Target intervention tensor: {intervention_tensor.shape}")
            
            # Execute ONLY up to the intervention point
            feed_dict = {input_ph: img_batch}
            intervention_activations = sess.run(intervention_tensor, feed_dict=feed_dict)
            
            if debug:
                print(f"Intervention activations shape: {intervention_activations.shape}")
                print(f"Activation range: [{np.min(intervention_activations):.6f}, {np.max(intervention_activations):.6f}]")
                print(f"Mean activation: {np.mean(intervention_activations):.6f}")
    
    # Return the raw activations for further processing
    return {
        'intervention_activations': intervention_activations,
        'shape': intervention_activations.shape,
        'success': True
    }

def apply_neuron_intervention_and_reshape(intervention_activations, neuron_idx, 
                                        intervention_type='none', intervention_scale=1.0, debug=False):
    """
    Apply intervention to specific neurons and perform required reshape operations.
    """
    
    if debug:
        print("Applying neuron intervention and reshape operations")
        print(f"Input shape: {intervention_activations.shape}")
    
    # Step 2a: Apply intervention (same as before)
    intervened_activations = intervention_activations.copy()
    
    if intervention_type == 'ablate':
        intervened_activations[0, neuron_idx, :, :] = 0.0
        if debug:
            print(f"Ablated neuron {neuron_idx}")
    elif intervention_type == 'amplify':
        intervened_activations[0, neuron_idx, :, :] *= intervention_scale
        if debug:
            print(f"Amplified neuron {neuron_idx} by {intervention_scale}x")
    
    # Verify intervention
    original_neuron_val = np.mean(intervention_activations[0, neuron_idx, :, :])
    intervened_neuron_val = np.mean(intervened_activations[0, neuron_idx, :, :])
    if debug:
        print(f"Neuron {neuron_idx} mean: {original_neuron_val:.6f} -> {intervened_neuron_val:.6f}")
    
    # Step 2b: Reshape_2 operation
    # Target: (1, 2560, 9, 9) → (1, 2560, 81)
    batch_size, channels, height, width = intervened_activations.shape
    
    # Flatten spatial dimensions while keeping channel dimension first
    reshaped_activations = intervened_activations.reshape(batch_size, channels, height * width)
    
    if debug:
        print(f"Reshape_2: {intervened_activations.shape} -> {reshaped_activations.shape}")
    
    # Step 2c: transpose_1 operation
    # Target: (1, 2560, 81) → (1, 81, 2560)
    transposed_activations = reshaped_activations.transpose(0, 2, 1)
    
    if debug:
        print(f"transpose_1: {reshaped_activations.shape} -> {transposed_activations.shape}")
        print(f"Final processed shape: {transposed_activations.shape}")
        print(f"Value range: [{np.min(transposed_activations):.6f}, {np.max(transposed_activations):.6f}]")
        
        # Verify against original shapes
        print(f"Matches original reshape_2? Should be (1, 2560, 81): {reshaped_activations.shape}")
        print(f"Matches original transpose_1? Should be (1, 81, 2560): {transposed_activations.shape}")
    
    return {
        'original_activations': intervention_activations,
        'intervened_activations': intervened_activations,
        'reshaped_activations': reshaped_activations,  # (1, 2560, 81)
        'final_processed_activations': transposed_activations,  # (1, 81, 2560)
        'intervention_success': True,
        'shapes': {
            'original': intervention_activations.shape,
            'intervened': intervened_activations.shape,
            'reshaped': reshaped_activations.shape,
            'final': transposed_activations.shape
        }
    }


def execute_remaining_attention_pipeline(processed_activations, debug=False):
    """
    Execute the remaining graph operations with complete attention mechanism.
    
    Input: processed_activations with shape (1, 81, 2560) from transpose_1
    Output: Final CLIP embedding (1, 640)
    """
    
    if debug:
        print("Executing remaining attention pipeline")
        print(f"Input shape: {processed_activations.shape}")
    
    model = CLIPImage()
    
    # Configure TensorFlow session
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.log_device_placement = False
    
    tf.reset_default_graph()
    with tf.Graph().as_default() as g:
        with tf.Session(config=config) as sess:
            # Create placeholder for our processed activations
            processed_input = tf.placeholder(tf.float32, shape=[None, 81, 2560], name='processed_input')
            
            # Import the model graph
            tf.import_graph_def(
                model.graph_def,
                {model.input_name: tf.placeholder(tf.float32, shape=[None, 288, 288, 3])},
                name='clip'
            )
            
            try:
                # Steps 1-3: Same as before (mean + concat + add)
                # Step 1: Apply Mean operation: (1, 81, 2560) → (1, 1, 2560)
                mean_output = tf.reduce_mean(processed_input, axis=1, keepdims=True)
                
                # Step 2: Get image_post/wp parameters and apply concat
                wp_tensor = g.get_tensor_by_name("clip/image_post/wp/read:0")
                wp_values = sess.run(wp_tensor)
                
                # Concat: [(1, 1, 2560), (1, 81, 2560)] → (1, 82, 2560)
                concat_output = tf.concat([mean_output, processed_input], axis=1)
                
                # Step 3: Add operation
                add_output = tf.add(concat_output, wp_values)  # (1, 82, 2560)
                
                if debug:
                    print(f"Mean + Concat + Add complete: {add_output.shape}")
                
                # Steps 4-7: Complete attention mechanism
                # Step 4: Q,K,V projections
                q_weights = g.get_tensor_by_name("clip/image_block/attn/q/w/read:0")
                q_bias = g.get_tensor_by_name("clip/image_block/attn/q/b/read:0")
                k_weights = g.get_tensor_by_name("clip/image_block/attn/k/w/read:0")
                k_bias = g.get_tensor_by_name("clip/image_block/attn/k/b/read:0")
                v_weights = g.get_tensor_by_name("clip/image_block/attn/v/w/read:0")
                v_bias = g.get_tensor_by_name("clip/image_block/attn/v/b/read:0")
                
                # Get weight values
                q_w, q_b, k_w, k_b, v_w, v_b = sess.run([q_weights, q_bias, k_weights, k_bias, v_weights, v_bias])
                
                # Apply Q,K,V projections
                add_reshaped = tf.reshape(add_output, [-1, 2560])  # (82, 2560)
                q_projected = tf.reshape(tf.matmul(add_reshaped, q_w) + q_b, [1, 82, 2560])
                k_projected = tf.reshape(tf.matmul(add_reshaped, k_w) + k_b, [1, 82, 2560])
                v_projected = tf.reshape(tf.matmul(add_reshaped, v_w) + v_b, [1, 82, 2560])
                
                if debug:
                    print(f"Q,K,V projections complete: {q_projected.shape} each")
                
                # Step 5: Multi-head reshape and transpose
                num_heads, head_dim = 40, 64
                q_heads = tf.transpose(tf.reshape(q_projected, [1, 82, num_heads, head_dim]), [0, 2, 1, 3])
                k_heads = tf.transpose(tf.reshape(k_projected, [1, 82, num_heads, head_dim]), [0, 2, 1, 3])
                v_heads = tf.transpose(tf.reshape(v_projected, [1, 82, num_heads, head_dim]), [0, 2, 1, 3])
                
                if debug:
                    print(f"Multi-head reshape complete: {q_heads.shape} each")
                
                # Step 6: Scaled dot-product attention
                scale = 1.0 / np.sqrt(head_dim)
                k_transposed = tf.transpose(k_heads, [0, 1, 3, 2])  # (1, 40, 64, 82)
                attention_scores = tf.matmul(q_heads, k_transposed) * scale  # (1, 40, 82, 82)
                attention_probs = tf.nn.softmax(attention_scores, axis=-1)
                attention_output = tf.matmul(attention_probs, v_heads)  # (1, 40, 82, 64)
                
                if debug:
                    print(f"Attention computation complete: {attention_output.shape}")
                
                # Step 7: Output projection + final slice
                c_proj_weights = g.get_tensor_by_name("clip/image_block/attn/c_proj/w/read:0")
                c_proj_bias = g.get_tensor_by_name("clip/image_block/attn/c_proj/b/read:0")
                c_proj_w, c_proj_b = sess.run([c_proj_weights, c_proj_bias])
                
                # Reshape attention output back to sequence format
                attn_reshaped = tf.reshape(tf.transpose(attention_output, [0, 2, 1, 3]), [1, 82, 2560])
                
                # Apply output projection
                attn_flat = tf.reshape(attn_reshaped, [-1, 2560])
                projected = tf.matmul(attn_flat, c_proj_w) + c_proj_b
                c_proj_output = tf.reshape(projected, [1, 82, 640])
                
                # Final slice: take first token
                final_embedding = c_proj_output[:, 0, :]  # (1, 640)
                
                if debug:
                    print(f"Output projection complete: {final_embedding.shape}")
                
                # Execute the complete pipeline
                feed_dict = {processed_input: processed_activations}
                final_result = sess.run(final_embedding, feed_dict=feed_dict)
                
                if debug:
                    print("Attention pipeline completed successfully")
                    print(f"Final embedding: {final_result.shape}, range: [{np.min(final_result):.6f}, {np.max(final_result):.6f}]")
                
                return {
                    'final_embedding': final_result,  # (1, 640)
                    'success': True
                }
                
            except Exception as e:
                if debug:
                    print(f"Error in attention pipeline: {e}")
                return {'error': str(e), 'success': False}

def run_complete_intervention_pipeline(image_path, neuron_idx=89, intervention_type='amplify', intervention_scale=2.0):
    """
    Execute the complete intervention pipeline from image input to final embedding.
    """
    # Step 1: Get activations
    step1_result = extract_intermediate_activations(image_path, debug=False)
    
    # Step 2: Apply intervention + manual ops
    step2_result = apply_neuron_intervention_and_reshape(
        step1_result['intervention_activations'], 
        neuron_idx, intervention_type, intervention_scale, debug=False
    )
    
    # Step 3: Execute remaining graph with complete attention
    step3_result = execute_remaining_attention_pipeline(
        step2_result['final_processed_activations'], debug=False
    )
    
    return step3_result