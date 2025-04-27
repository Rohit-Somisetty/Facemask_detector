# Face Mask Detector 😷

## Description

This project implements a real-time face mask detection system using Python, TensorFlow/Keras, and OpenCV. It identifies whether individuals in a video stream are wearing face masks or not. The system utilises a deep learning model built upon MobileNet architecture through transfer learning, fine-tuned on a custom dataset of masked and unmasked faces.

## Key Features

* **Real-time Detection:** Analyzes live video streams from a camera to detect mask usage.
* **Deep Learning Model:** Employs a fine-tuned MobileNet model for accurate classification.
* **Transfer Learning:** Leverages pre-trained weights from MobileNet (trained on ImageNet) and adapts them for the specific task of mask detection, freezing initial layers and training the final ones.
* **Clear Visual Feedback:** Draws a bounding box (circle in this case) around detected faces and labels them as "Mask" or "No Mask".

## Technology Stack

* Python
* TensorFlow / Keras
* OpenCV (implied for real-time video processing in `Main.ipynb`)
* NumPy
* Matplotlib / Seaborn (for model evaluation visuals)
* Scikit-learn (for data splitting and evaluation metrics)

## Setup and Usage

Follow these steps to get the detector running:

1.  **Download Repository Files:**
    * Clone or download all files from the GitHub repository: `https://github.com/Tihor3393/Facemask_detector`
    * Ensure all downloaded files (`FaceMask_Train.ipynb`, `Main.ipynb`, `app.py`, `haarcascade_frontalface_default.xml`) are in the same directory.

2.  **Download Dataset:**
    * Download the image dataset from Google Drive: [Dataset Link](https://drive.google.com/drive/folders/1HgfcJ8rWpuhZIXeAOk1dyvNamqvAdLKY?usp=sharing)
    * Organize the dataset as expected by the training script (likely separate folders for 'withmask' and 'withoutmask').

3.  **Update Training Script Paths:**
    * Open `FaceMask_Train.ipynb`.
    * Update the file paths within the notebook to point to the correct locations of your downloaded dataset directories (including the main dataset folder and the specific 'withmask' and 'withoutmask' subfolders). *Check cells involving `os.listdir` and `load_img`*.

4.  **Train the Model:**
    * Run all the cells in the `FaceMask_Train.ipynb` notebook.
    * This will preprocess the images, train the MobileNet-based model, and generate the following files:
        * `data.npy` (processed image data)
        * `labels.npy` (corresponding labels)
        * `MaskNet1.hdf5` (the trained model weights)
        * `mdl.h5` (an earlier save of the model structure/weights)

5.  **Prepare for Detection:**
    * Before running `Main.ipynb`, open it and update the directory path for the Haar Cascade file: `haarcascade_frontalface_default.xml`.

6.  **Run the Detector:**
    * Execute the `Main.ipynb` script.
    * A window will open, capturing video from your default camera.
    * Position your face within the indicated area (blue circle) for detection. The system will display "Mask" or "No Mask" based on its prediction.

7.  **Exit:**
    * Press the 'q' key to close the live stream window.

## Model Details

* **Base Model:** MobileNet (pre-trained on ImageNet, `include_top=False`).
* **Transfer Learning:** The initial layers of MobileNet are frozen (`layer.trainable = False`), and custom Dense layers (`Dense(1024)`, `Dense(512)`, `Dense(2)`) with ReLU and Softmax activations are added and trained on the mask dataset.
* **Optimizer:** Adam.
* **Loss Function:** Categorical Crossentropy.
* **Training:** The model was trained for 20 epochs with a batch size of 32, using a validation split.

## Generated Files (from `FaceMask_Train.ipynb`)

* `data.npy`: NumPy array containing the preprocessed image data.
* `labels.npy`: NumPy array containing the corresponding labels (one-hot encoded).
* `MaskNet1.hdf5`: The final trained Keras model file saved after training.
* `mdl.h5`: An intermediate save of the Keras model.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request or open an Issue.
