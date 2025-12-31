# 📊 Comparative Sentiment Analysis: BERT vs. RNN, LSTM, & GRU

## 📖 Overview
This project performs a comprehensive comparative analysis of multiple deep learning architectures for **Sentiment Analysis**. Implemented and evaluated four distinct models—**BERT, LSTM, GRU, and Simple RNN**—to understand their performance trade-offs in handling natural language context.

The goal is to benchmark "state-of-the-art" Transformer models against traditional Recurrent Neural Networks (RNNs) in terms of accuracy, F1-score, and computational efficiency.

## 📂 Dataset
* **Source:** [Sentiment Analysis Dataset on Kaggle](https://www.kaggle.com/datasets/abhi8923shriv/sentiment-analysis-dataset)
* **Content:** 1.6 million tweets labeled as Positive (4) or Negative (0).
* **Preprocessing:** Cleaning (URL/handle removal), tokenization, and stratified sampling to ensure balanced classes.

## 🛠️ Tech Stack
* **Language:** Python 3.10+
* **Deep Learning:** PyTorch, Hugging Face Transformers
* **Data Manipulation:** Pandas, NumPy
* **Evaluation & Viz:** Scikit-learn, Matplotlib, Seaborn

## 🧠 Models Implemented
1. **BERT (Bidirectional Encoder Representations from Transformers):** Uses pre-trained contextual embeddings (`bert-base-uncased`).
2. **GRU (Gated Recurrent Unit):** A streamlined RNN variant that solves the vanishing gradient problem with fewer parameters.
3. **LSTM (Long Short-Term Memory):** Handles long-term dependencies in text sequences.
4. **Simple RNN:** Used as a baseline to demonstrate limitations in capturing long-term context.

## 📈 Key Results
| Model | Accuracy | F1-Score | Training Loss (Final) |
| :--- | :--- | :--- | :--- |
| **BERT** | **High** | **High** | **0.14** (Best) |
| **GRU** | Medium | Medium | 0.51 |
| **LSTM** | Low | Low | ~0.69 (Did not converge) |
| **RNN** | Low | Low | ~0.69 (Baseline) |

* **BERT** achieved the highest convergence speed and accuracy, proving the superiority of pre-trained embeddings.
* **GRU** served as a viable lightweight alternative for resource-constrained environments.

## 🚀 How to Run
1.  **Clone the Repository**
    ```bash
    git clone https://github.com/chetanal23/sentiment-analysis-comparison.git
    cd sentiment-analysis-comparison
    ```

2.  **Install Dependencies**
    ```bash
    pip install pandas numpy torch transformers scikit-learn matplotlib seaborn tqdm ipywidgets
    ```

3.  **Download Data**
    * Download `training.1600000.processed.noemoticon.csv` from Kaggle.
    * Place it in the root directory.

4.  **Run the Notebook**
    * Open `NLP_Sentiment_Analysis.ipynb` in Jupyter Notebook or Google Colab.
    * Execute all cells to reproduce training and visualizations.

## 📊 Visualizations
The notebook generates the following deliverables:
* **F1-Score Comparison Bar Chart:** Visualizing the performance gap between models.
* **Confusion Matrices:** Heatmaps showing True Positives/Negatives for each model.

## 🤝 Contribution
Feel free to fork this repo and submit Pull Requests to add more models (e.g., RoBERTa, DistilBERT) or improve the preprocessing pipeline.
