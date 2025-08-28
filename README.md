# DermaScan

DermaScan is an AI-powered web application designed to classify images of skin conditions into one of six categories: 
- Chickenpox
- Cowpox
- HFMD
- Healthy
- Measles
- Monkeypox

Users can upload an image, and the application provides a prediction with a confidence score, along with detailed information about the predicted condition, including a description, symptoms, recommended actions, and prevention tips.

## Demo Video
<p align="center">
  <a href="https://youtu.be/iE5DSq-bQPw" target="_blank">
    <img src="https://img.youtube.com/vi/iE5DSq-bQPw/0.jpg" alt="Watch the video" width="600">
  </a>
</p>

## Features

- **Image Classification**: Classifies skin condition images into six categories using a fine-tuned ResNet50 model (quantized TensorFlow Lite).
- **Dynamic UI**: Transitions to a two-column layout post-prediction:
  - Left column: Displays the uploaded image and prediction results (prediction and confidence score).
  - Right column: Shows detailed information about the predicted condition with an independent scrollbar.
- **Detailed Results**: Provides a description, common symptoms, recommended actions, and prevention tips for the predicted disease.
- **Theme Toggle**: Includes a button to switch between light and dark modes, with preferences saved via `localStorage`.
- **Responsive Design**: Built with HTML, CSS, and JavaScript for a seamless user experience across devices.

## Technologies

- **Backend**: Python, Flask framework, Gunicorn web server.
- **Machine Learning**: TensorFlow/Keras, with a Tensorflow model (`final_model.h5`) based on ResNet50.
- **Frontend**: Single `index.html` page with CSS for styling (light/dark modes) and JavaScript for dynamic, asynchronous predictions using the Fetch API.

## Prerequisites

To run DermaScan locally, ensure you have the following installed:

- Git
- Python 3.9 or higher

## Setup and Run Instructions

Follow these steps to set up and run the DermaScan application locally:

1. **Clone the Repository**
   Clone the project from GitHub:

   ```bash
   git clone https://github.com/JaganM28/derma-scan
   cd dermascan
   ```

2. **Create and Activate a Virtual Environment**
   Create a virtual environment to isolate dependencies:

   - On Windows:

     ```bash
     python -m venv venv
     .\venv\Scripts\activate
     ```

   - On macOS/Linux:

     ```bash
     python3 -m venv venv
     source venv/bin/activate
     ```

3. **Install Dependencies**
   Install the required Python packages listed in `requirements.txt`:

   ```bash
   pip install -r requirements.txt
   ```

4. **Run the Application**
   Start the Flask server:

   ```bash
   python app.py
   ```

5. **Access the Application**
   Open a web browser and navigate to:

   ```
   http://127.0.0.1:5000
   ```

## Project File Structure

- `app.py`: The main Flask application script that handles routing and prediction logic.
- `final_model.h5`: The Tensorflow model file for skin condition classification.
- `requirements.txt`: Lists all Python dependencies required for the project.
- `templates/index.html`: The single HTML file containing the frontend user interface.
- `uploads/`: A temporary folder created at runtime to store uploaded images.

## Notes

- Ensure a stable internet connection for dependency installation.
- The `uploads/` folder is automatically created when the application processes an image upload.

## Disclaimer

DermaScan is an AI-powered tool and is not intended to provide medical diagnoses. Always consult a qualified healthcare professional for any health concerns.
