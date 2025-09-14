#  Deforestation Detection using U-Net on Sentinel-2 Imagery

This project focuses on detecting deforested regions using a deep learning-based semantic segmentation model (U-Net) applied to **cloud-free Sentinel-2 satellite images**. It enables efficient environmental monitoring by automatically identifying areas affected by deforestation.

---

##  Project Goal

To develop an automated solution using U-Net CNN that accurately segments deforestation areas from satellite images. The system helps minimize manual satellite inspection time and supports real-time environmental conservation efforts.

---

##  Dataset Description

- **Source**: [Kaggle – Deforestation Detection Dataset by Akhil Chibber](https://www.kaggle.com/datasets/akhilchibber/deforestation-detection-dataset)
- **Used Data**:
  - `1_CLOUD_FREE_DATASET/2_SENTINEL2/IMAGE_16_GRID`
  - `3_TRAINING_MASKS/MASK_16_GRID`
- Each `.tif` image represents a satellite tile, and each corresponding `.tif` mask represents deforestation areas in binary format.

---

##  Tools & Technologies

- Python
- TensorFlow / Keras
- NumPy, tifffile
- Scikit-learn
- OpenCV
- Streamlit (for frontend UI)

---

##  Problem Statement

Manual interpretation of satellite images for deforestation detection is time-consuming and labor-intensive. There's a need for an automated, scalable solution that can process large areas quickly and accurately to support environmental monitoring and policy-making.

---

##  Solution

The proposed solution leverages a deep learning U-Net model trained on patches of Sentinel-2 satellite images to predict pixel-level deforestation masks. The model is further integrated into a simple **Streamlit web app** to allow users to upload an image and view predicted deforestation regions instantly.

---

##  Methodology

###  Data Preprocessing
- Normalized Sentinel-2 bands (used B4, B3, B2 as RGB)
- Extracted 128×128 image patches
- Filtered patches with very few deforestation pixels to avoid class imbalance

### 2. Data Augmentation
- Applied horizontal/vertical flips, zoom, rotation, and shifts using `ImageDataGenerator`

### 3. U-Net Model
- Encoder-decoder architecture with:
  - BatchNormalization
  - Dropout layers
- Final Sigmoid layer for binary mask output
- Loss: Combined Dice Loss + Binary Crossentropy
- Metrics: Dice Coefficient, Accuracy

### 4. Training Strategy
- Trained for 100 epochs with:
  - EarlyStopping
  - ReduceLROnPlateau
  - ModelCheckpoint (best model saved as `unet_best_model.keras`)

---

##  Results

- Achieved high **Dice Coefficient** and **validation accuracy**
- Visualizations show strong overlap between ground truth and predicted masks


