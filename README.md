
# Emotion Detection Using Deep Learning

## Introduction
This project aims to classify the emotion on a person's face into one of seven categories: angry, disgusted, fearful, happy, neutral, sad, and surprised. We use deep convolutional neural networks (CNNs) to achieve this. The model is trained on the FER-2013 dataset, which consists of 35,887 grayscale, 48x48 sized face images.

## Dependencies
- Python 3
- OpenCV
- TensorFlow
- Keras

To install the required packages, run:
```bash
pip install -r requirements.txt
```

## Dataset
The FER-2013 dataset can be downloaded from Kaggle. Place the dataset in the `src` folder.

## Basic Usage
1. Clone the repository and enter the folder:
    ```bash
    git clone https://github.com/yourusername/Emotion-Detection.git
    cd Emotion-Detection
    ```

2. To train the model, run:
    ```bash
    cd src
    python emotions.py --mode train
    ```

3. To view predictions using a pre-trained model, run:
    ```bash
    cd src
    python emotions.py --mode display
    ```

## Folder Structure
```
Emotion-Detection/
│
├── src/
│   ├── data/
│   ├── emotions.py
│   ├── haarcascade_frontalface_default.xml
│   ├── model.h5
│   └── dataset_prepare.py
│
├── LICENSE
├── README.md
└── requirements.txt
```

## Algorithm
1. **Face Detection**: The Haar cascade method is used to detect faces in each frame of the webcam feed.
2. **Preprocessing**: The detected face region is resized to 48x48 pixels.
3. **Emotion Classification**: The preprocessed face image is passed through the CNN, which outputs a list of softmax scores for the seven emotion classes. The emotion with the highest score is displayed.

## References
- "Challenges in Representation Learning: A report on three machine learning contests." I Goodfellow, et al. arXiv 2013.

## License
This project is licensed under the MIT License - see the LICENSE file for details.

---

