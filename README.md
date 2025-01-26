![MasterHead](docs/SLR.png)
# Sign Language Recognition

This repository contains the source code and resources for a **Sign Language Recognition System**. The goal of this project is to develop a user-friendly and efficient application that recognizes and interprets sign language gestures in real-time using **Flask** and **OpenCV**.

Although the project focuses on recognizing **Sign Language**, it can be extended to detect any custom hand gestures with ease. The system is fully customizable and designed for real-world applications.

<br><br>

## Introduction
Sign language serves as a vital communication tool for individuals with hearing impairments. This project bridges the communication gap by using computer vision and machine learning algorithms to translate sign language gestures into text in real-time.

The system is made up of:

- **Real-Time Gesture Detection:** Using MediaPipe for hand detection and keypoint extraction.
- **Customizable Models:** Trainable for new gestures beyond sign language.
- **Web Interface:** A user-friendly interface for ease of use.
- **Real-Time Prediction:** Recognition of gestures with an interactive webcam feed.

Key Features:
- Fully customizable for recognizing gestures beyond sign language.
- Add spaces automatically when the **left hand** is detected for intuitive typing.
- Accurate and fast gesture recognition leveraging machine learning.

<br><br>

## Installation
To set up the Sign Language Recognition System locally, follow these steps:

1. Clone the repository:
   ```bash
   git clone https://github.com/your-repo/sign-language-recognition.git
   cd sign-language-recognition
   ```

2. Install dependencies:
   ```bash
   # Windows
   virtualenv env
   .\env\Scripts\activate
   pip install -r requirements.txt

   # Linux/macOS
   virtualenv env
   source env/bin/activate
   pip install -r requirements.txt
   ```

3. Install Jupyter Notebook if you want to retrain the model:
   ```bash
   pip install notebook
   ```

You are now ready to use the application.

<br><br>

## Usage
1. Ensure all dependencies are installed and your webcam is accessible.

2. Run the main application:
   ```bash
   python app.py
   ```

3. Open the application in your browser (usually `http://127.0.0.1:5000` by default).

4. Use gestures in front of your webcam:
   - Right hand: Recognizes the sign and displays the corresponding letter.
   - Left hand: Adds a **space** to the generated phrase automatically.

5. Enjoy the real-time recognition system!

<br><br>

## Customization
The system can be personalized to detect new gestures or signs:

1. Run the `app.py` file to open the application.
2. Press the `1` key to switch to **Data Collection Mode**. Use the `0` key to return to **Prediction Mode**.
3. Make the gesture you want to add to the dataset.
4. Press a key corresponding to the gesture's label to save it in the dataset.
5. Collect multiple samples for better accuracy.
6. Quit the program using the `Esc` key and open `train.ipynb` to retrain the model.

### Retraining Steps:
1. Open `train.ipynb` in Jupyter Notebook.
2. Execute each cell to preprocess the dataset and train the model.
3. Save the new model to replace the existing one in the `model` folder.

Your system is now updated to recognize new gestures!

<br><br>

## System Overview
The system uses the following pipeline:
1. **Gesture Detection:** Hand keypoints are extracted using MediaPipe.
2. **Feature Extraction:** Keypoints are normalized to ensure consistency.
3. **Prediction:** The processed keypoints are classified using a trained machine learning model.
4. **Result Display:** The predicted letters are displayed on the web interface in real-time.

![System Overview](docs/flow-chart.png)

<br><br>

## Data Collection
Data collection is crucial for training an accurate model. MediaPipe is used to track and extract **21 hand landmarks**. The landmarks are normalized and stored for model training.

![Hand Landmarks](docs/hand-landmarks.png)

Steps for data collection:
1. Collect samples for each gesture with varied lighting and angles.
2. Normalize the hand landmarks to remove noise.
3. Save the preprocessed data for model training.

<br><br>

## Preprocessing
Data preprocessing involves:

1. **Landmark Extraction:** Extract hand landmarks relative to the wrist's coordinate `(0, 0)`.
   
2. **Flatten and Normalize:** Convert the landmarks to a 1D list and normalize between `-1` and `1`.
   ```python
   max_value = max(list(map(abs, temp_landmark_list)))
   landmark_list = [n / max_value for n in temp_landmark_list]
   ```

3. **Save Preprocessed Data:**
   ```python
   with open("slr/model/keypoint.csv", 'a', newline="") as f:
       writer = csv.writer(f)
       writer.writerow([index, *landmark_list])
   ```

<br><br>

## Model Training
The preprocessed data is used to train a **Convolutional Neural Network (CNN)**:

1. Use the dataset split into training and validation sets.
2. Train the model using TensorFlow or PyTorch.
3. Save the trained model for deployment.

For this project:
- Accuracy on the testing set: **90.60%**
- Real-time performance: **71.12%**

<br><br>

## Contributing
Contributions are welcome! Feel free to:
- Open issues for bugs or feature requests.
- Submit pull requests with improvements or fixes.
- Share feedback to enhance usability.

Thank you for supporting the Sign Language Recognition System!

