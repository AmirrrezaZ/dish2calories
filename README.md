
# dish2calories

![License](https://img.shields.io/badge/license-MIT-green) ![Python](https://img.shields.io/badge/python-3.10%2B-blue) ![PyTorch](https://img.shields.io/badge/PyTorch-EfficientNet--B0-red)

**dish2calories** is a reimplementation of the **Nutrition5k** dataset and model, designed to estimate the nutritional content of a dish (calories, protein, fat, carbohydrates, and mass) from a single image.  
It combines **multi-label ingredient classification** with **mass regression**, leveraging a deep learning model built on an **EfficientNet-B0 backbone**.


---

<p align="center">
  <img src="pics/ui.jpg" alt="dish2calories ui" width="300"/>
</p>

---



## 📌 Overview
The project uses a **multi-task learning** approach to:

- **Classify Ingredients**: Identify which ingredients are present in a dish image.
- **Predict Mass**: Estimate the total dish mass (in grams).
- **Calculate Nutrition**: Combine predictions with per-gram nutritional data to compute:
  - Total Calories  
  - Protein  
  - Fat  
  - Carbohydrates  

The model is served via:
- **FastAPI** (`inference.py`) → for programmatic inference.
- **Streamlit** (`app.py`) → for interactive user-friendly UI.  
- **Docker** → for portable and easy deployment.

---

## 📊 Dataset
The model is trained on the **Nutrition5k dataset**, which includes:

- **5000 unique dishes**, each with ~16 images from alternating angles.
- **Directory structure**: Images grouped by dish ID.
- **Metadata**: CSVs with nutritional info (calories, protein, fat, carbs, and mass).

**Sources**:
- [Kaggle: Nutrition5k Dataset (Side-Angle Images)](https://www.kaggle.com/search?q=nutrition5k)
- [GitHub: google-research-datasets/nutrition5k](https://github.com/google-research-datasets/nutrition5k)

---

## 🏗 Model Architecture

### 🔹 Google Baseline (Nutrition5k Original)
- **Backbone**: InceptionV2 pretrained on **JFT-300M** (~300M images, 18k classes, closed-source).  
- **Input**: 256×256 images, downsized & center-cropped.  
- **Optimization**: RMSProp (lr=1e-4, momentum=0.9, decay=0.9, epsilon=1.0).  
- **Multi-task heads**:
  - Shared 4096-dense layers.
  - Regression heads for calories, macronutrients, and mass.

### 🔹 Our Implementation (CalorieNet)
- **Backbone**: EfficientNet-B0 (ImageNet-pretrained).  
- **Input**: 256×256 images (normalized).  
- **Multi-task heads**:
  - **Classification Head**: Multi-label ingredient classification.
  - **Regression Head**: Mass prediction (grams) via 2-layer MLP (hidden_dim=64, dropout=0.3).
- **Mass Scaling**: `StandardScaler` normalization.  
- **Loss Function**:
  - BCEWithLogitsLoss → classification.  
  - MAE (L1) → regression.  
  - Weighted combination with **uncertainty weighting** ([Kendall et al., 2017](https://arxiv.org/abs/1705.07115)).

---

## 📈 Results

All MAE values are on the held-out test split. Calories are in kcal, mass in grams, and fat/carbs/protein in grams.

- **Direct**: nutrition is computed from the predicted ingredients and predicted mass (portion-dependent).
- **Portion Indep**: the per-gram prediction is scaled to the *ground-truth* mass, which isolates ingredient/macro error from mass error.

| Method                 | Calories MAE | Mass MAE | Fat MAE | Carb MAE | Protein MAE |
|------------------------|--------------|----------|---------|----------|-------------|
| Google (Portion Indep) | 24.1         | -        | 2.3     | 2.7      | 2.2         |
| Google (Direct)        | 70.6         | 40.4     | 5.0     | 6.1      | 5.5         |
| Ours (Direct)          | 67.66        | 28.56    | 4.61    | 6.41     | 4.52        |
| Ours (Portion Indep)   | 56.12        | -        | 4.04    | 4.79     | 3.94        |

Ingredient classification (multi-label, micro-averaged):

| Accuracy | Precision | Recall | F1     |
|----------|-----------|--------|--------|
| 0.8754   | 0.9614    | 0.9503 | 0.9558 |

---

## 🔍 What Does the Model Do?
The dish2calories model's methodology enhances interpretability through transparent ingredient classification, mass regression, and nutritional calculations, while being more lightweight than Google's baseline by using EfficientNet-B0 and only two task heads (classification and regression) compared to Google's four.
The classification head takes the features from EfficientNet-B0 and predicts which ingredients are present.  
This is a multi-label classification task, meaning the model can predict multiple ingredients at once (e.g., a dish might have chicken and rice).  

The regression head also uses the features from EfficientNet-B0 to predict the total mass of the dish in grams.  

The model uses a table called `norm_ingr`, which lists the nutritional content per gram for each ingredient. For example (illustrative values):
- Chicken: 0.27 g protein, 0.04 g fat, 0 g carbs per gram.
- Rice: 0.03 g protein, 0.003 g fat, 0.28 g carbs per gram.

The model calculates the per-gram nutrition of the dish by combining the predicted ingredients. There are two ways it can do this:
- **Binary Mode**: If an ingredient is predicted (e.g., chicken and rice), it takes the average of their per-gram nutrition.
- **Probabilistic Mode**: Uses the confidence scores (e.g., 80% chicken, 60% rice) to weigh the contribution of each ingredient's nutrition. Higher-confidence ingredients contribute more to the average.

For example, if the dish is predicted to have chicken and rice, the per-gram nutrition is (binary mode):
- Protein: (0.27 + 0.03) / 2 = 0.15 g per gram.
- Fat: (0.04 + 0.003) / 2 ≈ 0.02 g per gram.
- Carbs: (0 + 0.28) / 2 = 0.14 g per gram.

The total nutrition is calculated by multiplying the per-gram values by the predicted mass. If mass = 200 g:
- Total protein = 0.15 × 200 = 30 g.
- Total fat = 0.02 × 200 ≈ 4 g.
- Total carbs = 0.14 × 200 = 28 g.

Calories are calculated using the standard Atwater formula:
- Calories = (4 × protein) + (9 × fat) + (4 × carbs).
- Using the example above: (4 × 30) + (9 × 4) + (4 × 28) = 120 + 36 + 112 = **268 kcal**.

**Output Example**  
The model returns a dictionary with:
- Total Protein: e.g., 172g.  
- Total Fat: e.g., 31g.  
- Total Carbs: e.g., 28g.  
- Total Calories: e.g., 1079 calories.  
- Total Mass: e.g., 200g.  
- Labels: List of detected ingredients (e.g., ["chicken", "rice"]).  
- Confidence Scores: How sure the model is about each ingredient (e.g., [0.8, 0.6]).  
- Mean Confidence: Average confidence across detected ingredients.  

For a single image, this is returned via the FastAPI endpoint or displayed in the Streamlit UI.  

---


## 🚀 Quick Start

```bash
git clone https://github.com/AmirrrezaZ/dish2calories.git
cd dish2calories
docker compose up --build        # CPU, no GPU needed
```

Open the UI at <http://localhost:8501> and upload a dish photo. Or call the API directly:

```bash
curl -F "file=@dataset/test.jpg" http://localhost:8000/predict
```

```json
{
  "total_protein": 30.1,
  "total_fat": 4.2,
  "total_carbs": 28.0,
  "total_calories": 268.0,
  "total_mass": 200.0,
  "labels_txt": ["chicken", "rice"],
  "labels_conf": [0.8, 0.6],
  "mean_conf": 0.7
}
```

(Values above are illustrative.) Interactive API docs are at <http://localhost:8000/docs>.

---

## 🗂 Project Structure

| Path | Purpose |
|------|---------|
| `app.py` | Streamlit UI |
| `inference.py` | FastAPI server (`/`, `/health`, `/predict`) |
| `onnx_predictor.py` | Lightweight ONNX Runtime inference + nutrition calculation (used by the app and API) |
| `predict.py` | PyTorch `CalorieNetPredictor` (used for evaluation) |
| `export_onnx.py` | Converts the `.pth` checkpoint to `.onnx` |
| `models.py`, `loss.py` | CalorieNet model and multi-task loss |
| `dataset_loaders.py` | Nutrition5k data loading and splits |
| `main.py`, `train.py` | Training entry point and loop |
| `evaluate.py`, `logs_visualizer.py` | Metrics and training-curve plots |
| `weights/` | Trained checkpoint (`efficientnet_b0.pth`) and exported `efficientnet_b0.onnx` |
| `configs/dataset_info.json` | Class index and per-gram ingredient nutrition table |
| `dataset/` | Metadata CSVs (images are **not** included – download them, see Dataset) |
| `service.py` | Model loading + result formatting shared by the UI and the API |
| `requirements*.txt` | `requirements.txt` (app), `-api` (FastAPI), `-train` (PyTorch, training/export) |
| `Dockerfile`, `docker-compose.yml`, `start.sh` | Container setup |

---

## ⚙️ Installation & Usage

### 🔹 Docker Compose

```bash
docker compose up --build
```

* FastAPI: [http://localhost:8000](http://localhost:8000)
* Streamlit: [http://localhost:8501](http://localhost:8501)

**Tips**

* Run in background: `docker compose up --build -d`
* Stop & clean up: `docker compose down`
* Single container with both API and UI: `docker build -t dish2calories . && docker run -p 8501:8501 dish2calories`

Inference runs on **ONNX Runtime (CPU)**, so no GPU, CUDA or PyTorch is needed in the image.

### 🔹 Running Locally (Without Docker)

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt     # Streamlit + ONNX Runtime (small, no PyTorch)
streamlit run app.py                # runs the model in-process, no API needed
```

To also run the REST API: `pip install -r requirements-api.txt && python inference.py`.

The UI has two modes: if the `API_URL` environment variable is set it calls that FastAPI server (this is what Docker Compose does); otherwise it loads the ONNX model directly. Configuration is via environment variables – see [`.env.example`](.env.example): `API_HOST`, `API_PORT`, `API_URL`, `API_RELOAD`.

### 🔹 Live Demo

Hosted on Streamlit Community Cloud: <https://dish2calories.streamlit.app/>. It runs `app.py` with the small `requirements.txt` (no PyTorch, unpinned versions) so installs are fast on any Python version.

### 🔹 Training & Evaluation

1. Download the Nutrition5k side-angle images and place them under `dataset/` next to the metadata CSVs (see [Dataset](#-dataset)).
2. Adjust hyper-parameters in the `config` dict at the top of `main.py` (image size, batch size, epochs, learning rate, `num_workers`, …).
3. Install training dependencies and train:

   ```bash
   pip install -r requirements-train.txt   # PyTorch (CPU build; use pytorch.org for CUDA), torchmetrics, ...
   python main.py
   ```

   Checkpoints go to `weights/`, logs and loss curves to `logs/`, and class info to `configs/dataset_info.json`.
4. Export the trained checkpoint for the app (`weights/efficientnet_b0.onnx`):

   ```bash
   python export_onnx.py
   ```

5. Evaluate with `evaluate.py` (`NutritionEvaluator`) and plot curves with `logs_visualizer.py`.

The repo already ships trained weights (`.pth` for training/evaluation and `.onnx` for the app), so you do **not** need to train to run the app. Re-run `export_onnx.py` after retraining.

---

## ⚠️ Limitations

- Estimates come from a **single image**, so hidden ingredients, sauces and portion depth are hard to see. Mass error is ~29 g on the test set.
- Trained on Nutrition5k (cafeteria dishes), so accuracy drops on other cuisines, packaged foods and unusual plating.
- The ingredient vocabulary is fixed to the classes in `configs/dataset_info.json`.
- Results are estimates for general interest, **not medical or dietary advice**.

---

## 📚 References

* Nutrition5k Paper: [arXiv:2103.03375](https://arxiv.org/abs/2103.03375)
* Multi-task Loss: [arXiv:1705.07115](https://arxiv.org/abs/1705.07115)
* Blog: *Predicting Nutritional Contents of Meals Using Deep Learning*
* Dataset: [Kaggle: Nutrition5k](https://www.kaggle.com/search?q=nutrition5k) | [GitHub: Nutrition5k](https://github.com/google-research-datasets/nutrition5k)

---

## 📜 License

This project is licensed under the **MIT License**.
See the [LICENSE](LICENSE) file for details.

## 📝 Citation

```bibtex
@inproceedings{thames2021nutrition5k,
  title     = {Nutrition5k: Towards Automatic Nutritional Understanding of Generic Food},
  author    = {Thames, Quin and Karpur, Arjun and Norris, Wade and Xia, Fangting and Panait, Liviu and Weyand, Tobias and Sim, Jack},
  booktitle = {CVPR},
  year      = {2021}
}
```
