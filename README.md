# 😷 Face Mask Detector 🧍➡️✅

Can a computer instantly tell if someone's wearing a face mask? This project explores that using a Deep Learning approach (MobileNet with Transfer Learning) to analyze video feeds in real-time!

---

## 🚀 Project Goal

The main goal is to build and deploy a system that uses a camera feed to automatically detect if individuals have face masks on or not, providing immediate visual feedback. It's a practical application of computer vision and deep learning.

---

## ✨ What it Does

* **Real-Time Detection:** Analyzes live video streams from your webcam.
* **Mask On/Off Identification:** Classifies faces as either "Mask" or "No Mask".
* **Visual Feedback:** Draws a circle around detected faces and displays the classification result directly on the video feed.
* **Deep Learning Powered:** Uses a fine-tuned MobileNet convolutional neural network for accurate detection.
* **Haar Cascades Face Detection:** Employs `haarcascade_frontalface_default.xml` to efficiently locate faces in the video stream first.

---

## 💡 The Core Idea: Transfer Learning

Instead of training a massive network from scratch, this project uses **Transfer Learning**:
1.  It takes the powerful **MobileNet** model, already trained on millions of general images (ImageNet).
2.  It **freezes** the early layers (which learned general features like edges and textures).
3.  It **adds** new custom layers at the end.
4.  It **trains** only these new layers (and potentially unfreezes a few later MobileNet layers) specifically on the task of recognizing masks vs. no masks, using the provided dataset. This is much faster and often more effective than starting from zero!

---

## 💻 Tech Stack

* **Python 3**: The core language.
* **TensorFlow / Keras**: For building and training the deep learning model.
* **OpenCV**: For video capture, image processing, and drawing detections.
* **NumPy**: For numerical operations, especially handling image data.
* **Scikit-learn**: Used for splitting data and evaluating the model (Confusion Matrix, Classification Report).
* **Matplotlib / Seaborn**: For plotting training history and the confusion matrix.

---

## ⚙️ Get Started: How to Run

1.  **Clone Repo:** `git clone https://github.com/Tihor3393/Facemask_detector.git && cd Facemask_detector`
2.  **Get Dataset:** Download from [Google Drive](https://drive.google.com/drive/folders/1HgfcJ8rWpuhZIXeAOk1dyvNamqvAdLKY?usp=sharing) and place in appropriate `withmask`/`withoutmask` folders.
3.  **Train Model (`FaceMask_Train.ipynb`):**
    * Update dataset paths within the notebook.
    * Run all cells. This generates `data.npy`, `labels.npy`, and `MaskNet1.hdf5`.
4.  **Run Detector (`Main.ipynb`):**
    * Update the path to `haarcascade_frontalface_default.xml`.
    * Run the script.
    * Look at your webcam feed window!
    * Press 'q' to quit.

---

## 📊 Model Performance

The trained model (`MaskNet1.hdf5`) achieves excellent accuracy on the test set (around 99-100% based on the notebook output). The classification report and confusion matrix generated at the end of `FaceMask_Train.ipynb` provide detailed precision, recall, and F1-scores for both "Mask" and "No Mask" classes.

---

## 🤝 Contributing

Got ideas or improvements? Feel free to fork the repository and submit a pull request!

---
