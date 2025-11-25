# Text Summarization for Indonesian News Articles

A comprehensive deep learning project implementing both **abstractive** and **extractive** text summarization for Indonesian news articles using state-of-the-art transformer models. This project fine-tunes and evaluates multiple models including T5, EncoderDecoder, and BERT-based models on the Liputan6 news dataset.

You can download the PDF File of Documentation on : https://drive.google.com/file/d/1HWoqitzmqGmkaxu38fJGPTzq2H3CQo6K/view?usp=drive_link

## 🎯 Overview

This project implements comprehensive text summarization solutions for Indonesian news articles, covering both **abstractive** and **extractive** approaches:

### Abstractive Summarization
Generates new sentences that capture the essence of the original text. The project explores two model architectures:
1. **T5 (Text-to-Text Transfer Transformer)**: A unified framework that treats summarization as a text-to-text task
2. **EncoderDecoder Model**: A BERT-GPT2 hybrid architecture specifically designed for Indonesian text summarization

### Extractive Summarization
Selects and combines the most important sentences from the original text. The project implements:
1. **K-Means Clustering with BERT Embeddings**: Uses sentence embeddings to cluster and select representative sentences
2. **TextRank Algorithm**: Graph-based ranking algorithm using sentence similarity for sentence selection
3. **Multiple BERT Models**: Evaluates various BERT architectures (IndoBERT, NusaBERT, multilingual BERT)

## ✨ Features

### Abstractive Summarization
- **Dual Model Implementation**: Fine-tuned T5 and EncoderDecoder models for Indonesian text summarization
- **Efficient Training**: Optimized training with mixed precision (FP16) and batch processing
- **Model Comparison**: Side-by-side performance comparison of different architectures

### Extractive Summarization
- **K-Means Clustering**: Unsupervised learning approach using BERT sentence embeddings
- **TextRank Algorithm**: Graph-based sentence ranking using PageRank algorithm
- **Multiple BERT Models**: Evaluation of IndoBERT, NusaBERT, and multilingual BERT

### General Features
- **Comprehensive Evaluation**: ROUGE metric evaluation (ROUGE-1, ROUGE-2, ROUGE-L) for all models
- **Preprocessing Pipeline**: Text cleaning and normalization for Indonesian news articles
- **Comparative Analysis**: Performance comparison between abstractive and extractive approaches

## 📊 Dataset

This project uses the **Liputan6 News Dataset**, a large-scale Indonesian news article dataset.

### Dataset Information

- **HuggingFace Dataset Page**: [id_liputan6](https://huggingface.co/datasets/id_liputan6)
- **Google Drive Download**: [Download Dataset](https://drive.google.com/file/d/1ixaIO24XBZX-BFVyHIk0FG0kI2W3lACD/view)

### Dataset Structure

The dataset is organized in the following structure:
```
liputan6_data/
└── canonical/
    ├── train/      # Training set (78,043 files)
    ├── dev/        # Development set (10,972 files)
    └── test/       # Test set (10,972 files)
```

Each JSON file contains:
- `clean_article`: Preprocessed article text
- `clean_summary`: Preprocessed summary text
- `extractive_summary`: Extractive summary reference

### Dataset Setup

1. Download the dataset from the Google Drive link above
2. Extract the dataset to the project root directory
3. Ensure the `liputan6_data/` folder is in the correct location

## 🚀 Installation

### Prerequisites

- Python 3.8 or higher
- CUDA-capable GPU (recommended for training)
- Git

### Step-by-Step Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/GregReynaldi/abstractive-summarization-indonesian.git
   cd abstractive-summarization-indonesian
   ```

2. **Create a virtual environment** (recommended)
   ```bash
   # Using venv
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   
   # Or using conda
   conda create -n summarization python=3.9
   conda activate summarization
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Download and prepare the dataset**
   - Download from the [Google Drive link](#dataset)
   - Extract to the project root directory
   - Ensure the folder structure matches the expected format

## 💻 Usage

### Running the Notebook

1. **Start Jupyter Notebook**
   ```bash
   jupyter notebook
   ```

2. **Open the notebooks**
   - **Abstractive Summarization**: Open `AbstractiveSummarization.ipynb`
   - **Extractive Summarization**: Open `ExtractiveSummarization.ipynb`
   - **Data Exploration**: Open `ExploratoryDataAnalysis.ipynb`

3. **Execute cells sequentially**
   - **AbstractiveSummarization.ipynb** includes:
     - Data loading and preprocessing
     - T5 model training and evaluation
     - EncoderDecoder model evaluation
   - **ExtractiveSummarization.ipynb** includes:
     - K-Means clustering with BERT embeddings
     - TextRank algorithm implementation
     - Multiple BERT model comparisons

### Training the T5 Model

The notebook includes code to:
- Load and preprocess the Liputan6 dataset
- Fine-tune the T5 model (`panggi/t5-small-indonesian-summarization-cased`)
- Evaluate using ROUGE metrics
- Save the trained model

### Evaluating the EncoderDecoder Model

The notebook also includes evaluation of the pre-trained EncoderDecoder model (`cahya/bert2gpt-indonesian-summarization`) for comparison.

### Extractive Summarization

The `ExtractiveSummarization.ipynb` notebook implements two approaches:

1. **K-Means Clustering Approach**:
   - Uses BERT embeddings to represent sentences
   - Applies K-Means clustering to group similar sentences
   - Selects sentences closest to cluster centers
   - Evaluates with multiple BERT models

2. **TextRank Algorithm**:
   - Builds similarity graph between sentences
   - Uses PageRank algorithm to rank sentences
   - Selects top-ranked sentences for summary
   - Implements cosine similarity using BERT embeddings

### Key Parameters

- **Input Max Length**: 384 tokens
- **Output Max Length**: 160 tokens
- **Training Batch Size**: 64
- **Evaluation Batch Size**: 32
- **Learning Rate**: 2e-5
- **Training Epochs**: 2
- **Training Samples**: 1,000 (randomly sampled)
- **Test Samples**: 2,000 (randomly sampled)

## 📈 Results

### Abstractive Summarization Results

#### T5 Model Performance

| Metric | Score |
|--------|-------|
| ROUGE-1 | 0.379 |
| ROUGE-2 | 0.209 |
| ROUGE-L | 0.310 |

#### EncoderDecoder Model Performance

| Metric | Score |
|--------|-------|
| ROUGE-1 | 0.426 |
| ROUGE-2 | 0.249 |
| ROUGE-L | 0.359 |

### Extractive Summarization Results

#### K-Means Clustering with BERT Embeddings

| Model | ROUGE-1 | ROUGE-2 | ROUGE-L |
|-------|---------|---------|---------|
| indobert-base-p1 | 0.260 | 0.115 | 0.195 |
| all-nusabert-large-v4 | 0.279 | 0.130 | 0.210 |
| bert-base-multilingual-cased | 0.247 | 0.103 | 0.186 |

#### TextRank Algorithm (all-nusabert-large-v4)

| Metric | Score |
|--------|-------|
| ROUGE-1 | 0.146 |
| ROUGE-2 | 0.051 |
| ROUGE-L | 0.146 |

### Comparative Analysis

**Abstractive vs Extractive Performance:**
- The **EncoderDecoder model** achieves the best overall performance (ROUGE-1: 0.426), demonstrating the effectiveness of abstractive approaches for generating summaries
- Among extractive methods, **NusaBERT with K-Means** performs best (ROUGE-1: 0.279), showing that domain-specific Indonesian models outperform multilingual alternatives
- Abstractive methods generally outperform extractive methods, as they can generate more coherent and concise summaries rather than simply selecting existing sentences

**Model-Specific Insights:**
- **EncoderDecoder (BERT2GPT)**: Best overall performance, indicating the effectiveness of the hybrid architecture
- **T5 Model**: Good performance with faster training due to smaller model size
- **NusaBERT**: Best extractive performance, highlighting the importance of Indonesian-specific pre-training

## 📁 Project Structure

```
.
├── AbstractiveSummarization.ipynb    # Main notebook with model training and evaluation
├── ExploratoryDataAnalysis.ipynb      # Data exploration and analysis
├── ExtractiveSummarization.ipynb      # Extractive summarization experiments
├── collect_data.ipynb                 # Data collection utilities
├── collect_data_2.ipynb               # Additional data collection utilities
├── requirements.txt                   # Python dependencies
├── README.md                          # Project documentation
├── .gitignore                         # Git ignore patterns

## 🤖 Models

### Abstractive Summarization Models

1. **T5 Model**: `panggi/t5-small-indonesian-summarization-cased`
   - Pre-trained on Indonesian text
   - Fine-tuned on Liputan6 dataset
   - Architecture: T5-small
   - Training: Seq2SeqTrainer with FP16 mixed precision

2. **EncoderDecoder Model**: `cahya/bert2gpt-indonesian-summarization`
   - BERT encoder + GPT-2 decoder
   - Pre-trained specifically for Indonesian summarization
   - Used for evaluation and comparison

### Extractive Summarization Models

1. **IndoBERT**: `indobenchmark/indobert-base-p1`
   - Indonesian-specific BERT model
   - Used for sentence embedding extraction

2. **NusaBERT**: `LazarusNLP/all-nusabert-large-v4`
   - Large-scale Indonesian BERT model
   - Best performing extractive model

3. **Multilingual BERT**: `bert-base-multilingual-cased`
   - General multilingual BERT
   - Baseline comparison model

### Model Training Details

**T5 Fine-tuning:**
- **Framework**: Hugging Face Transformers
- **Trainer**: Seq2SeqTrainer
- **Optimization**: AdamW optimizer
- **Learning Rate**: 2e-5
- **Precision**: Mixed precision training (FP16)
- **Evaluation**: ROUGE metrics computed during training

**Extractive Methods:**
- **K-Means**: 2 clusters, k-means++ initialization
- **TextRank**: PageRank with damping factor 0.85
- **Embeddings**: Mean pooling of BERT last hidden states

## 📦 Requirements

All required packages are listed in `requirements.txt`. Key dependencies include:

**Core Libraries:**
- `torch` >= 2.0.0
- `transformers` >= 4.30.0
- `datasets` >= 2.12.0
- `pandas` >= 2.0.0
- `numpy` >= 1.24.0

**Machine Learning:**
- `scikit-learn` >= 1.3.0 (for K-Means clustering)

**Graph Algorithms:**
- `networkx` >= 3.0 (for TextRank algorithm)

**Evaluation & Visualization:**
- `rouge-score` >= 0.1.2
- `matplotlib` >= 3.7.0

For the complete list, see [requirements.txt](requirements.txt).

## 👤 Author

**Gregorius Reynaldi**

- GitHub: [@GregReynaldi](https://github.com/GregReynaldi)
- Computer Science Student at Harbin Institute of Technology
- Aspiring AI Engineer | NLP Engineer

## 📄 License

This project is open source and available under the [MIT License](LICENSE).

## 🙏 Acknowledgments

- **Liputan6 Dataset**: For providing the comprehensive Indonesian news dataset
- **Hugging Face**: For the Transformers library and model hub
- **Model Contributors**: 
  - `panggi` for the T5 Indonesian summarization model
  - `cahya` for the BERT2GPT Indonesian summarization model
  - `indobenchmark` for the IndoBERT model
  - `LazarusNLP` for the NusaBERT model

---

**Note**: This project is for educational and research purposes. The dataset should be used in accordance with its original license terms.




