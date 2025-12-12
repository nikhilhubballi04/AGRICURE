# Download Guide - Grape Plant Disease Recognition Project

## 📦 What You Need to Download

### 1. **Dataset (REQUIRED)**
The project requires the grape disease dataset to train or test the models.

**Dataset Link:** https://www.kaggle.com/datasets/rm1000/grape-disease-dataset-original

**How to Download:**
- You need a Kaggle account (free)
- Go to the dataset page and click "Download"
- Extract the dataset to your project folder

**Expected Folder Structure:**
```
Grape-Plant-Disease-Recognition-main/
├── Original Data/
│   ├── train/
│   │   ├── Black Rot/
│   │   ├── ESCA/
│   │   ├── Healthy/
│   │   └── Leaf Blight/
│   └── test/
│       ├── Black Rot/
│       ├── ESCA/
│       ├── Healthy/
│       └── Leaf Blight/
```

**After downloading, update the paths in the notebooks:**
- Change `/kaggle/input/grape-disease-dataset-original/Original Data/train` 
- To your local path, e.g., `./Original Data/train` or `C:/path/to/Original Data/train`

### 2. **Python Packages (REQUIRED)**

Install all required packages using pip:

```bash
pip install -r requirements.txt
```

**Or install individually:**
```bash
pip install numpy pandas tensorflow keras Pillow opencv-python matplotlib seaborn jupyter jupyterlab streamlit scikit-learn
```

**Key Packages:**
- **TensorFlow/Keras** - Deep learning framework
- **NumPy & Pandas** - Data processing
- **Pillow** - Image processing
- **Matplotlib** - Visualization
- **Jupyter** - Notebook environment
- **Streamlit** - Web app (optional, for GRU.ipynb)

### 3. **Trained Model Files (OPTIONAL)**

If you want to use pre-trained models without training:
- The notebooks will train models and save them as `.h5` files
- You can also check if the author has shared trained models
- Look for files like `grape_disease_model.h5` or similar

**Note:** The GRU.ipynb references model files at `/content/drive/MyDrive/grape_disease_model.h5` - you'll need to:
- Train a model first using one of the model notebooks, OR
- Download pre-trained models if available, OR
- Update the path to point to your trained model

### 4. **Additional Files (OPTIONAL)**

- **Classification Report Images** - Already included in the `images/` folder
- **Labels Mapping JSON** - May be needed for the Streamlit app (check GRU.ipynb)

## 🚀 Quick Start

1. **Install Python packages:**
   ```bash
   cd Grape-Plant-Disease-Recognition-main/Grape-Plant-Disease-Recognition-main
   pip install -r requirements.txt
   ```

2. **Download the dataset:**
   - Visit: https://www.kaggle.com/datasets/rm1000/grape-disease-dataset-original
   - Download and extract to the project folder
   - Update dataset paths in notebooks

3. **Start Jupyter:**
   ```bash
   jupyter notebook
   ```
   Or use the running instance at: http://localhost:8889

4. **Open and run notebooks:**
   - Start with `models/MobileNet(97%).ipynb` for the best model
   - Update dataset paths before running
   - Train the model or use pre-trained weights

## ⚠️ Important Notes

- The notebooks were created for **Google Colab/Kaggle**, so paths need to be updated for local use
- Dataset contains **9000+ images** - ensure you have enough disk space
- Training deep learning models requires significant computational resources (GPU recommended)
- The dataset has 4 classes: Black Rot, ESCA, Healthy, Leaf Blight

## 📊 Model Performance

- **MobileNet**: 97% accuracy (Recommended)
- **ResNet50V2**: 93% accuracy
- **VGG16**: 88% accuracy

