# Tuberculosis-Detection-using-CNN-Models
uberculosis Detection Using Deep Learning
This project focuses on developing and evaluating a deep learning model to detect tuberculosis (TB) from chest X-ray images. The model leverages pre-trained convolutional neural networks (CNNs) to classify images and assist in TB diagnosis.

Project Structure
Data Downloading and Preparation:
The dataset is automatically downloaded from a specified source, uncompressed, and prepared for training and validation.

Model Creation:
A function create_model is provided to build and compile the deep learning model. This function uses a pre-trained base model, applies global average pooling, and adds a dense layer with softmax activation for classification into TB-positive or TB-negative categories.

Model Evaluation:
A function evaluate_model is used to assess the model's performance. It makes predictions on the validation dataset, computes a confusion matrix, and prints the accuracy score along with a detailed classification report.

Files
DS_Project_9Models (1) (1).ipynb: Jupyter notebook containing the entire workflow from data preparation, model creation, training, and evaluation.
Requirements
Python 3.x
TensorFlow
Keras
NumPy
Scikit-learn
Matplotlib
How to Run
Clone the repository:
git clone https://github.com/yourusername/tuberculosis-detection.git
cd tuberculosis-detection
Install the required packages:
pip install -r requirements.txt
Open the Jupyter notebook and run the cells:
jupyter notebook DS_Project_9Models\ (1)\ (1).ipynb
Results
The notebook will guide you through the entire process of training the TB detection model and evaluating its performance on the validation dataset. The final output includes accuracy, confusion matrix, and a classification report.

Model Accuracy Comparison Graph
Description of Image

Confusion Matrix of each Model
Description of Image Description of Image

Description of Image Description of Image
Description of Image Description of Image
Description of Image Description of Image
Description of Image
Contributing
Contributions are welcome! Please fork the repository and submit pull requests.

License
This project is licensed under the MIT License - see the LICENSE file for details
