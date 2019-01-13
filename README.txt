CNN Model

Language: Python 2.7.15
Package manager: pip 18.1

Installing dependencies:
pip install -r requirements.txt


# 1) Trainig the model, tuning hyperparametrs, saving the final model

To evaluate the model on previously trained weights -
python mnist.py --mode='evaluate'

To save the weights of model for production server -
python mnist.py --mode='save'

To tune hyperparametrs and train the model-
python mnist.py --mode='train' --conv1_kernel_size 5 5 --conv2_1_kernel_size 5 5 --conv2_2_kernel_size 5 5 --conv3_1_kernel_size 5 5 --conv3_2_kernel_size 5 5

You can alternatively check the usage using command-
python3 mnist.py -h


# 2) To deploy production server-

Start the server using -
simple_tensorflow_serving --model_base_path="./models/mnist_classifier/" &

Send test requests using - (this returns the classes and their probabilities in a dictionary)
python client.py
