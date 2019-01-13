import numpy as np
import tensorflow as tf
import argparse

tf.logging.set_verbosity(tf.logging.INFO)
args = {}

def parse_args():
    '''
    Parses the arguments.
    '''
    parser = argparse.ArgumentParser(description="CNN Model")
    parser.add_argument('--mode', default='evaluate', help='evaluate the previously trained weights')
    parser.add_argument('--conv1_kernel_size', type=int, nargs='+', default=[5,5], help='conv1 filter size')
    parser.add_argument('--conv2_1_kernel_size', type=int, nargs='+', default=[5,5], help='conv2_1 filter size')
    parser.add_argument('--conv2_2_kernel_size', type=int, nargs='+', default=[5,5], help='conv2_2 filter size')
    parser.add_argument('--conv3_1_kernel_size', type=int, nargs='+', default=[5,5], help='conv3_1 filter size')
    parser.add_argument('--conv3_2_kernel_size', type=int, nargs='+', default=[5,5], help='conv3_2 filter size')
    return parser.parse_args()


def cnn_model(features, labels, mode):
    """Model function"""

    # Reshape X to 4-D tensor: [batch_size, width, height, channels]
    input_layer = tf.reshape(features["x"], [-1, 28, 28, 1])

    # Conv1 computes 32 features using a 5x5 filter with ReLU activation.
    # Input Tensor Shape: [batch_size, 28, 28, 1]
    # Output Tensor Shape: [batch_size, 28, 28, 32]
    conv1 = tf.layers.conv2d(
        inputs=input_layer,
        filters=32,
        kernel_size=args.conv1_kernel_size,
        padding="same",
        activation=tf.nn.relu)

    # Pooling1 max pooling layer with a 2x2 filter and stride of 2
    # Input Tensor Shape: [batch_size, 28, 28, 32]
    # Output Tensor Shape: [batch_size, 14, 14, 32]
    pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)

    # Conv2_1 computes 64 features using a 5x5 filter.
    # Input Tensor Shape: [batch_size, 14, 14, 32]
    # Output Tensor Shape: [batch_size, 14, 14, 64]
    conv2_1 = tf.layers.conv2d(
        inputs=pool1,
        filters=64,
        kernel_size=args.conv2_1_kernel_size,
        padding="same",
        activation=tf.nn.relu)

    # pool2_1 max pooling layer with a 2x2 filter and stride of 2
    # Input Tensor Shape: [batch_size, 14, 14, 64]
    # Output Tensor Shape: [batch_size, 7, 7, 64]
    pool2_1 = tf.layers.max_pooling2d(inputs=conv2_1, pool_size=[2, 2], strides=2)

    # Conv2_2 computes 64 features using a 5x5 filter.
    # Input Tensor Shape: [batch_size, 14, 14, 32]
    # Output Tensor Shape: [batch_size, 14, 14, 64]
    conv2_2 = tf.layers.conv2d(
        inputs=pool1,
        filters=64,
        kernel_size=args.conv2_2_kernel_size,
        padding="same",
        activation=tf.nn.relu)

    # pool2_2 max pooling layer with a 2x2 filter and stride of 2
    # Input Tensor Shape: [batch_size, 14, 14, 64]
    # Output Tensor Shape: [batch_size, 7, 7, 64]
    pool2_2 = tf.layers.max_pooling2d(inputs=conv2_2, pool_size=[2, 2], strides=2)

    # Conv3_1 computes features using a 5x5 filter.
    # Input Tensor Shape: [batch_size, 7, 7, 64]
    # Output Tensor Shape: [batch_size, 7, 7, 256]
    conv3_1 = tf.layers.conv2d(
        inputs=pool2_1,
        filters=256,
        kernel_size=args.conv3_1_kernel_size,
        padding="same",
        activation=tf.nn.relu)

    # Conv3_2 computes features using a 5x5 filter.
    # Input Tensor Shape: [batch_size, 7, 7, 64]
    # Output Tensor Shape: [batch_size, 7, 7, 256]
    conv3_2 = tf.layers.conv2d(
        inputs=pool2_2,
        filters=256,
        kernel_size=args.conv3_2_kernel_size,
        padding="same",
        activation=tf.nn.relu)

    conv3 = tf.concat([conv3_1, conv3_2],axis=-1)

    # Flatten tensor into a batch of vectors
    # Input Tensor Shape: [batch_size, 7, 7, 64]
    # Output Tensor Shape: [batch_size, 7 * 7 * 64]
    conv3_flat = tf.reshape(conv3, [-1, 7 * 7 * 512])

    # fc1 layer with 1000 neurons
    # Input Tensor Shape: [batch_size, 7 * 7 * 64]
    # Output Tensor Shape: [batch_size, 1000]
    fc1 = tf.layers.dense(inputs=conv3_flat, units=1000, activation=tf.nn.relu)

    # fc2 layer with 500 neurons
    # Input Tensor Shape: [batch_size, 1000]
    # Output Tensor Shape: [batch_size, 500]
    fc2 = tf.layers.dense(inputs=fc1, units=500, activation=tf.nn.relu)

    # Logits layer
    # Input Tensor Shape: [batch_size, 500]
    # Output Tensor Shape: [batch_size, 10]
    logits = tf.layers.dense(inputs=fc2, units=10)

    predictions = {
        # Generate predictions
        "classes": tf.argmax(input=logits, axis=1),
        # Add `softmax_tensor` to the graph. It is used for PREDICT and by the logging_hook.
        "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
    }
    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

    # Calculate Loss (for both TRAIN and EVAL modes)
    loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)

    # Configure the Training Op (for TRAIN mode)
    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
        train_op = optimizer.minimize(loss=loss, global_step=tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

    # Add evaluation metrics (for EVAL mode)
    eval_metric_ops = {"accuracy": tf.metrics.accuracy(labels=labels, predictions=predictions["classes"])}

    return tf.estimator.EstimatorSpec(mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)


def main(unused_arg):
    # Load training and eval data
    mnist = tf.contrib.learn.datasets.load_dataset("mnist")
    train_data = mnist.train.images  # Returns np.array
    train_labels = np.asarray(mnist.train.labels, dtype=np.int32)
    eval_data = mnist.test.images  # Returns np.array
    eval_labels = np.asarray(mnist.test.labels, dtype=np.int32)

    # Create the Estimator
    mnist_classifier = tf.estimator.Estimator(
        model_fn=cnn_model, model_dir="./training_weights/mnist_convnet_model")

    # Log the values in the "Softmax" tensor with label "probabilities"
    tensors_to_log = {"probabilities": "softmax_tensor"}
    logging_hook = tf.train.LoggingTensorHook(
        tensors=tensors_to_log, every_n_iter=50)

    if args.mode == 'train':
        # Train the model
        train_input_fn = tf.estimator.inputs.numpy_input_fn(
            x={"x": train_data},
            y=train_labels,
            batch_size=100,
            num_epochs=None,
            shuffle=True)
        mnist_classifier.train(
            input_fn=train_input_fn,
            steps=13000,
            hooks=[logging_hook])

    if args.mode == 'evaluate':
        # Evaluate the model and print results
        eval_input_fn = tf.estimator.inputs.numpy_input_fn(
            x={"x": eval_data},
            y=eval_labels,
            num_epochs=1,
            shuffle=False)
        eval_results = mnist_classifier.evaluate(input_fn=eval_input_fn)
        print(eval_results)

    if args.mode == 'save':
        # save the final model and weights
        image = tf.placeholder(tf.float32, shape=[None, 28,28])
        input_fn = tf.estimator.export.build_raw_serving_input_receiver_fn({'x': image})
        mnist_classifier.export_savedmodel('./models', input_fn, strip_default_attrs=True)
        print("model exported")


if __name__ == "__main__":
    args = parse_args()
    tf.app.run()
