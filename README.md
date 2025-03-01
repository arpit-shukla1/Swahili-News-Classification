# Swahili News Classification

## Overview
This project focuses on **Swahili News Classification** using **Natural Language Processing (NLP)** techniques. The objective is to develop a machine learning model that classifies Swahili-language news articles into predefined categories.

## Dataset
- **Source:** [Zindi Africa Swahili News Classification Competition](https://zindi.africa/competitions/swahili-news-classification)
- **Categories:**
  - **Biashara** (Business)
  - **Burudani** (Entertainment)
  - **Kimataifa** (International News)
  - **Kitaifa** (National News)
  - **Michezo** (Sports)
- **Challenges:**
  - **Severe class imbalance** (e.g., Burudani has only 2 samples)
  - **Overlapping topics** between categories
  - **Limited labeled Swahili NLP datasets**

## Preprocessing
- **Data Cleaning:** Removed special characters and redundant spaces
- **Tokenization:** Used Swahili-specific tokenizer
- **Truncation & Padding:** Standardized input length to 512 tokens
- **Stopword Removal:** Removed frequent but non-informative words
- **Label Encoding:** Converted categorical labels into numerical format

## Model Selection
- **Proposed Model:**
  - **RoBERTa-based model** fine-tuned for Swahili
  - **Pre-trained model:** [`benjamin/roberta-base-wechsel-swahili`](https://huggingface.co/benjamin/roberta-base-wechsel-swahili)
  - **Final Accuracy:** **91.4%**

## Methodology
- **Train-Validation Split:** 80% Training, 20% Validation
- **Loss Function:** Cross-Entropy Loss (weighted for class imbalance)
- **Optimizer:** AdamW
- **Hyperparameters:**
  - **Batch Size:** 8
  - **Learning Rate:** 2e-5
  - **Epochs:** 10

## Results
| Metric  | Score |
|---------|------|
| **Accuracy** | 91.40% |
| **Precision (Weighted)** | 91.55% |
| **Recall (Weighted)** | 91.40% |
| **F1 Score (Weighted)** | 91.41% |

### **Per-Class Performance**
| Class | Precision | Recall | F1 Score |
|-------|-----------|-----------|-----------|
| **Biashara** | 92.55% | 88.39% | 90.42% |
| **Burudani** | 50.00% | 100.00% | 66.67% |
| **Kimataifa** | 75.00% | 54.55% | 63.16% |
| **Kitaifa** | 87.11% | 91.71% | 89.35% |
| **Michezo** | 96.17% | 94.18% | 95.17% |

## Key Challenges
- **Severe class imbalance** affecting minority classes
- **Semantic overlap** between categories
- **Low-resource NLP tools** for Swahili


## Future Work
- **Train multilingual models** (e.g., XLM-R) for better cross-lingual performance
- **Improve pretraining on Swahili corpora** for better context understanding
- **Deploy as an API** for real-time Swahili news classification

## Acknowledgements
- **Supervisor:** Dr. Nirav Bhatt
- **Dataset Provider:** Zindi Africa
- **Pre-trained Model:** [Hugging Face](https://huggingface.co/benjamin/roberta-base-wechsel-swahili)

## License
This project is open-source and available under the **MIT License**.

---

🚀 **Contributions & Feedback** are welcome!
