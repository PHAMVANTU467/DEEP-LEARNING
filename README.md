This is a simple Convolutional Neural Network for image classification using PyTorch.

As an example, we use Kaggle dataset for classification of image of cats and dogs("https://download.microsoft.com/download/3/E/1/3E1C3F21-ECDB-4869-8368-6DEBA77B919F/kagglecatsanddogs_3367a.zip"). The example download and process the data into training and validation sets.
The model for this example is set up in "catvsdog_example.py".
For running, it requires to install the next libraries:

  -PyTorch
  -torchvision
  -tqdm
  -zipfile
  -cv2
You can create other model for image classification, you only need to provide the dataset in the variable "parameters['dataset']"
