import tensorflow as tf
import tf2onnx

class CustomDense(tf.keras.Model):
    def __init__(self):
        super(CustomDense, self).__init__()
        # Define a fully connected layer with 128 input features and 64 output features
        self.fc = tf.keras.layers.Dense(64, input_shape=(128,))

    @tf.function(input_signature=[tf.TensorSpec(shape=[None, 128], dtype=tf.float32)])
    def call(self, x):
        # Pass the input through the fully connected layer
        return self.fc(x)

# Create an instance of the model
model_dense = CustomDense()

concrete_func = model_dense.call.get_concrete_function(tf.TensorSpec(shape=[None, 128], dtype=tf.float32))

# You can save the model using the TensorFlow SavedModel format
tf.saved_model.save(model_dense, "test_dense_tf", signatures={"serving_default": concrete_func})



# To export this model to ONNX, you would need to use an additional tool like tf2onnx

# python3 -m tf2onnx.convert --saved-model test_dense_tf --output test_dense_tf.onnx --opset 13
