# 😷 Real-Time Face Mask Detector 🚀

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](https://www.python.org/) [![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)](https://www.tensorflow.org/) [![Keras](https://img.shields.io/badge/Keras-%3E3.0-red.svg)](https://keras.io/) [![OpenCV](https://img.shields.io/badge/OpenCV-Required-brightgreen.svg)](https://opencv.org/)


---

## 🔥 Overview

Ever wondered if someone's *really* wearing a mask? This project leverages the power of Deep Learning to build a **real-time face mask detection system**! Using Python, TensorFlow/Keras, and OpenCV, it analyzes your camera feed to instantly identify masked and unmasked faces.

Built upon the robust **MobileNet** architecture via **transfer learning**, this detector is both efficient and accurate.

---

## ✨ Key Features

* **⚡ Real-Time Performance:** Detects mask usage live from a webcam feed.
* **🧠 Deep Learning Powered:** Utilizes a fine-tuned MobileNet model for high accuracy.
* **🎓 Transfer Learning:** Smartly adapts a powerful pre-trained model (MobileNet on ImageNet) for this specific task, saving time and resources.
* **👀 Clear Visual Feedback:** Draws a helpful circle around detected faces, clearly labeling them "Mask" or "No Mask".

---

## 💻 Tech Stack

* Python 3
* TensorFlow / Keras
* OpenCV
* NumPy
* Matplotlib / Seaborn (for visualizing model training/evaluation)
* Scikit-learn (for data splitting and performance metrics)

---

## ⚙️ Setup & Blast Off!

Get ready to run the detector:

1.  **Clone the Repo:**
    * Grab all the code: `git clone https://github.com/Tihor3393/Facemask_detector`
    * Navigate into the directory: `cd Facemask_detector`
    * Make sure all essential files (`FaceMask_Train.ipynb`, `Main.ipynb`, `app.py`, `haarcascade_frontalface_default.xml`) are present.

2.  **Snag the Dataset:**
    * Download the image dataset from: [Google Drive Dataset Link](https://drive.google.com/drive/folders/1HgfcJ8rWpuhZIXeAOk1dyvNamqvAdLKY?usp=sharing)
    * Organize it into `withmask` and `withoutmask` folders within a main dataset directory, as expected by `FaceMask_Train.ipynb`.

3.  **Configure Training Paths:**
    * Launch `FaceMask_Train.ipynb`.
    * **Crucial:** Update the file paths inside to point exactly where you saved the dataset (check cells loading images, e.g., using `os.listdir` and `load_img`).

4.  **Train Your AI Brain!**
    * Run all cells in `FaceMask_Train.ipynb`. This might take a while depending on your hardware! ⏳
    * It'll process images and train the model, creating these essential files:
        * `data.npy` & `labels.npy`
        * `MaskNet1.hdf5` (Your shiny new trained model!)
        * `mdl.h5`

5.  **Prep the Detector:**
    * Open `Main.ipynb`.
    * Find the line loading `haarcascade_frontalface_default.xml` and update its path to the correct location in your project directory.

6.  **Launch!**
    * Run the `Main.ipynb` script.
    * Your webcam feed should appear in a new window.
    * Show your face (masked or unmasked) in the blue circle and watch the magic happen!

7.  **Shutdown:**
    * Simply press the **'q'** key to close the detector window.

---

## 🤖 Model Insights

* **Foundation:** MobileNet (pre-trained on ImageNet).
* **Adaptation (Transfer Learning):** We freeze the early MobileNet layers and train newly added Dense layers (1024, 512, and 2 neurons) specifically for mask detection using your dataset.
* **Training Recipe:** Optimized with Adam, using Categorical Crossentropy loss, trained over 20 epochs.

---

## 📊 Generated Artifacts

Running `FaceMask_Train.ipynb` produces:

* `data.npy`: Preprocessed image data.
* `labels.npy`: Corresponding image labels.
* `MaskNet1.hdf5`: The final trained model file.
* `mdl.h5`: An intermediate model save.

---

## 🤝 Contributing

Got ideas to make this even better? Contributions are highly encouraged! Fork the repo, make your changes, and submit a Pull Request. You can also open an Issue for bugs or feature suggestions.

---

