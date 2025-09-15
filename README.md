# Forest-Fire-Detection-App

## Forest Fire Detection Using CNN (Week 1 & 2)

### Overview
This project aims to build a deep learning model for detecting forest fires from images. Forest fires pose a significant risk to both human lives and the environment. Detecting these fires early is crucial for mitigating the damage. The goal is to create an image classification model using Convolutional Neural Networks (CNN) to identify fire and no-fire images from a dataset of forest fire images.

### Week 1 Work
- Setup and exploration of the Wildfire Dataset  
- Organizing the dataset into training, validation, and testing directories  
- Visualizing sample images from both the "fire" and "no fire" classes  
- Basic understanding of the dataset structure

## Dataset
The dataset used is the Wildfire Dataset. It contains images in two classes:  
- **Fire:** Images of forest fires  
- **No Fire:** Images of forest areas without any fire  

Organized into three main directories:  
- **Train:** Used for training the model  
- **Validation:** Used to evaluate the model during training  
- **Test:** Used to evaluate final model performance

## Project Structure
1. **Dataset Exploration:** Inspect images and class distribution  
2. **Model Construction:** Build a CNN to classify images  
3. **Data Preprocessing:** Resize and rescale images; use data generators  
4. **Training the Model:** Train on the training set; validate on validation and test sets

## Steps Involved
1. **Install Required Libraries**  
2. **Dataset Setup** (download and organize into train/val/test)  
3. **Visualize the Data** (display sample images)  
4. **Build & Train CNN** (define architecture, compile, train)

## Technologies Used
- **TensorFlow/Keras** for deep learning  
- **Kaggle** for dataset download  
- **Google Colab** for cloud training with GPU  
- **Matplotlib** for visualization  

---

## Forest Fire Detection (Week 3)

### Overview
In Week 3, focus shifted to integration and deployment. The trained CNN model was connected to a Streamlit web interface for real-time fire detection from user-uploaded images. The deployment pipeline was finalized for hosting via GitHub and Streamlit Cloud.

### Week 3 Work
- **Model Saving and Finalization**  
  - Completed training and validation of the CNN model  
  - Saved the trained model as `FFD.keras`  
- **Streamlit App Development (`app.py`)**  
  - Created a responsive web interface using Streamlit  
  - Features:  
    - Upload and preview images  
    - Predict fire vs. no fire  
    - Display prediction confidence  
  - Integrated `FFD.keras` model with `tensorflow.keras.models.load_model()`

## Repository Structure Finalized
```
Forest-Fire-Dectection-App/
├── .gitattributes       # Git LFS config
├── FFD.keras            # Saved model (Git LFS)
├── app.py               # Streamlit application
├── requirements.txt     # Dependencies
└── README.md            # Documentation
```

## Deployment Guide
1. Push `app.py`, `FFD.keras`, `requirements.txt`, and `README.md` to GitHub  
2. Go to [streamlit.io/cloud](https://streamlit.io/cloud)  
3. Connect your GitHub repository  
4. Set `app.py` as the main file  
5. Launch the application

## requirements.txt
```
streamlit
tensorflow
pillow
numpy
```

## Technologies Used
| Tool             | Purpose                                  |
| ---------------- | ---------------------------------------- |
| Python           | Core programming language                |
| TensorFlow/Keras | Deep learning model development          |
| NumPy            | Numerical computations                   |
| Matplotlib       | Data visualization                       |
| Streamlit        | Web application interface                |
| Google Colab     | Cloud-based training                     |
| GitHub           | Version control and deployment           |

## Future Enhancements
- Add webcam-based real-time fire detection  
- Improve model performance with more data and regularization  
- Incorporate Grad-CAM visualizations for interpretability  
- Implement real-time alert systems (SMS, email)
