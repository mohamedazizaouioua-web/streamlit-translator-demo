# 🤖 AI English-to-French Translator 🇫🇷

This project is a fully functional web application that translates English sentences into French using a deep learning Transformer model. The entire model was built and trained from scratch and is deployed as an interactive demo using Streamlit.

[![Live Demo](https://img.shields.io/badge/Live_Demo-Click_Here-brightgreen?style=for-the-badge)](https://app-translator-demo-58ybsevqcwparweazxyyku.streamlit.app/)
[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://app-translator-demo-58ybsevqcwparweazxyyku.streamlit.app/)

---

### Add a Screenshot of Your App Here!
> **Pro Tip:** Take a nice screenshot of your running application and drag-and-drop it into this README file on GitHub. A visual makes a huge difference!

<img width="1919" height="911" alt="Image" src="https://github.com/user-attachments/assets/699386cd-2f5a-48b8-9e68-560f28955b0f" />
## 🚀 Project Overview

This project demonstrates an end-to-end machine learning workflow:
1.  **Data Processing:** Cleaning and preparing a large corpus of over 175,000 English-French sentence pairs.
2.  **Model Building:** Implementing a Transformer neural network from scratch using TensorFlow and Keras.
3.  **Training:** Training the model on a Kaggle GPU to learn the patterns of translation.
4.  **Deployment:** Packaging the trained model and tokenizers into a user-friendly, interactive web application with Streamlit.
5.  **Sharing:** Deploying the app to the cloud for anyone to use.

## ✨ Features

-   **Real-time Translation:** Instantly translates English text into French.
-   **Interactive UI:** A simple and clean user interface built with Streamlit.
-   **Deep Learning Core:** Powered by a from-scratch Transformer model, the same architecture that powers modern systems like ChatGPT.

## 🛠️ Technology Stack

-   **Backend & Model:** Python, TensorFlow, Keras
-   **Frontend App:** Streamlit
-   **Data Manipulation:** Pandas, NumPy, Scikit-learn
-   **Training Environment:** Kaggle Notebooks (with GPU)
-   **Version Control & Deployment:** Git, GitHub, Streamlit Community Cloud

## 🧠 How It Works: The Transformer Model

The heart of this translator is a **Transformer** model, an architecture that relies on a concept called the **Attention Mechanism**.

-   **The Encoder:** Reads the input English sentence and creates a rich numerical representation, understanding the context and relationships between words.
-   **The Decoder:** Takes the Encoder's output and, one word at a time, generates the French translation, paying "attention" to the most relevant parts of the English sentence at each step.

This allows the model to handle long-range dependencies and complex grammatical structures far more effectively than older architectures.

## 📦 Setup and Local Installation

Want to run this app on your own machine? Follow these steps.

### Prerequisites

-   Python 3.8+
-   Git

### Installation Steps

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/DabbebiNaim/streamlit-translator-demo.git
    cd streamlit-translator-demo
    ```

2.  **Create and activate a virtual environment:**
    ```bash
    # For macOS/Linux
    python3 -m venv venv
    source venv/bin/activate

    # For Windows
    python -m venv venv
    .\venv\Scripts\activate
    ```

3.  **Install the required libraries:**
    ```bash
    pip install -r requirements.txt
    ```

## ▶️ Running the App

Once the installation is complete, run the following command in your terminal:

```bash
streamlit run app.py
