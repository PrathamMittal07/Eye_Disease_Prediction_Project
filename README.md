# Eye_Disease_Prediction_Project
This is a ML and DL based project that takes an input of OCT Scan from user and predicts four conditions out of CNV (Choroidal Neovascularization), DME (Diabetic Macular Edema),Drusen (Dry AMD),Normal.It will be giving output as Normal.
Eye Disease Prediction using OCT Scans
Overview
This project implements a Deep Learning and Computer Vision solution to predict common eye diseases from Optical Coherence Tomography (OCT) scans. Our goal is to develop an easy-to-use web application that can assist ophthalmologists and healthcare professionals in the early and accurate diagnosis of retinal conditions, ultimately enhancing patient outcomes through timely intervention.

Project Goals
Develop a Deep Learning Model: Design an efficient model that can classify OCT images into four groups: Normal, Choroidal Neovascularization (CNV), Diabetic Macular Edema (DME), and Drusen.

Build a Web Application: Give a simple interface for uploading OCT images and getting real-time predictions along with confidence scores.

Obtain High Accuracy: Train the model to obtain high accuracy on training and validation sets, providing consistent performance in a clinical scenario.

How it Works: The Project Workflow
The application takes a safe and streamlined workflow to make predictions.

Patient Data Input: A user (for instance, a medical practitioner) uploads an OCT scan image through the web application.

Data Preprocessing: The preprocessed image is automatically uploaded. This consists of resizing and normalizing the pixel values to fit the format needed by the deep learning model.

Model Prediction: The preprocessed OCT scan is input to our deep learning model that has been trained. The model scans the image for certain patterns and characteristics that are indicative of the four disease groups.

Prediction Output: The model gives a prediction, stating the likely condition and the confidence level. For example, it may give "Prediction: CNV, Confidence: 95%".

Generated Report: This confidence and prediction score are subsequently rendered in a clean report format on the web application's interface.

Technology Stack
Frontend: The user interface is constructed by using HTML, CSS, and JavaScript so that the application is responsive and intuitive in nature.

Backend: Server-side logic, responsible for image processing and interaction with the deep learning model, is written in Python.

Deep Learning Architecture: We employ the MobileNet,ResNet 50 and one more  deep learning architecture. These models are selected due to their effectiveness and efficiency in image classification, which makes them suitable for a web application that should be lightweight. We leverage a transfer learning strategy, whereby we trained these models for our dataset and made accuracy of each model more than 95%.

Getting Started
To have a local copy of this project running, follow these steps:

Clone the Repository:
git clone [https://github.com/PrathamMittal07/Eye_Disease_Prediction_Project/tree/main]
cd [your-project-directory]

Install Dependencies:

pip install -r requirements.txt

Run the Application:

# Command to run our Flask/FastAPI app, e.g.
python app.py

Access the Application: Open your web browser and visit the local server URL (e.g., http://127.0.0.1:5000).

Future Plans
We have a number of ideas for how to improve the capabilities of the project:

Improve Model Accuracy: Try more complex architectures (e.g., ResNet, EfficientNet) and ensemble learning methods.

Enlargen Disease Coverage: Enlarge the model to be able to diagnose more eye diseases beyond the present four types.

Integrate with EHR: Create an API to facilitate smooth integration with Electronic Health Records (EHR) in clinical settings.
