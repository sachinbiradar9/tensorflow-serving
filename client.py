import requests
import numpy as np

def main():
  # make sure the server is running, refer to README.txt file to start the server
  endpoint = "http://127.0.0.1:8500"

  # You can input your data here, make sure your data is of shape (x,28,28)
  images = np.random.randn(2,28,28).tolist()
  json_data = {"model_name": "default", "data": {"x": images} }
  result = requests.post(endpoint, json=json_data)

  # this is the final prediction with probabilities
  print(result.text)

if __name__ == "__main__":
  main()
