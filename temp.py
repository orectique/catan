import tensorflow as tf

# Check if GPU is available
if tf.test.is_gpu_available():
    print("GPU is available")
else:
    print("GPU is not available")

# Print the list of available GPUs
gpus = tf.config.list_physical_devices('GPU')
for gpu in gpus:
    print("GPU:", gpu)