from lucid.modelzoo.vision_base import Model
from lucid.optvis import render
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
from lucid.misc.io import load, save
import os

# First, set up a global session and configuration
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2'  # Make all GPUs visible
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
os.environ['TF_ENABLE_NHWC'] = '1'

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.gpu_options.per_process_gpu_memory_fraction = 0.95  # Use 95% of each GPU
config.allow_soft_placement = True
config.gpu_options.visible_device_list = "0,1,2"

# Create a global session
global_session = tf.Session(config=config)
tf.keras.backend.set_session(global_session)

# Create a distribution strategy
strategy = tf.distribute.MirroredStrategy()

class CLIPImage(Model):
    image_value_range = (0, 255)
    input_name = 'input_image'
    
    def __init__(self):
        # Use the global session instead of creating a new one
        self.sess = global_session
        
        with tf.gfile.GFile('./image32.pb', "rb") as f:
            self.graph_def = tf.GraphDef()
            self.graph_def.ParseFromString(f.read())
        self.model_name = "RN50_4x"
        self.image_shape = [288, 288, 3]
        self.model_path = "https://openaipublic.blob.core.windows.net/clip/tf/RN50_4x/084ee9c176da32014b0ebe42cd7ca66e/image32.pb"

    def load(self, inp=None):
        # Use the global session
        with self.sess.as_default():
            if inp is None:
                inp = tf.placeholder(shape=(None, self.image_shape[0], self.image_shape[1], 3), 
                                   dtype=tf.float32, name="clip_image_input")
            
            # Import the model - TensorFlow will automatically distribute operations
            # across available GPUs when executing operations
            self.T = render.import_model(self, inp, inp)
            return inp, self.T


class CLIPText(Model):
    input_name = 'tokens'

    def __init__(self):
        # Use the global session
        self.sess = global_session
        
        self.model_name = "RN50_4x_text"
        self.model_path = "https://openaipublic.blob.core.windows.net/clip/tf/RN50_4x/da21bc82c7bba068aa8163333438354c/text32.pb"

    def load(self, O=None):
        # Use the global session
        with self.sess.as_default():
            if O is None:
                O = tf.placeholder(tf.int32, [None, None], name="clip_text_input")
            
            # Import the graph definition
            tf.import_graph_def(self.graph_def, {self.input_name: O}, name="text")
            gph = tf.get_default_graph()
            self.T = lambda x: gph.get_tensor_by_name("text/" + x + ":0")
            return O, self.T