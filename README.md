# RECOMP MASTER PRO // Terminal v4.0

![Fallout Theme](https://img.shields.io/badge/UI-Fallout--Style-green?style=for-the-badge)
![Python](https://img.shields.io/badge/Python-3.10+-blue?style=for-the-badge&logo=python)
![Scikit-Learn](https://img.shields.io/badge/Machine--Learning-Scikit--Learn-orange?style=for-the-badge)

**RECOMP MASTER PRO** is a robust data analytics and machine learning application designed to track and predict body recomposition progress. Clad in a vintage "Fallout" terminal aesthetic, it provides a high-contrast, immersive environment for serious data analysis.

## 🚀 Features

- **📊 Data Management**: Import datasets from CSV, Excel, or SQLite. Preview and manage records with a high-performance terminal interface.
- **🛠️ CRUD Operations**: Manual entry, modification, and deletion of records (Weight, Calories, Protein, etc.) directly through the UI.
- **📈 Advanced Visualizations**: Dynamic generation of Scatter, Bar, Line, and Pie charts with a matching phosphor-green aesthetic.
- **🤖 ML Engine**:
    - Multiple models: Linear Regression, Logistic Regression, SVM, Random Forest, KNN.
    - Automated outlier detection via IQR.
    - Specialized Logistic Regression for classification based on user-defined body fat thresholds.
    - Integrated inference panel for manual predictions.
- **⌨️ Immersive UX**: Typewriter animations and console logs for every operation.

## 🛠️ Installation

1. **Clone the repository**:
   ```bash
   git clone <repository_url>
   cd BodyRecompAnalytics
   ```

2. **Create a virtual environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the application**:
   ```bash
   python main.py
   ```

## 📦 Dependencies

- `customtkinter`: Modern, dark-themed UI components.
- `pandas`: Powerful data manipulation and analysis.
- `scikit-learn`: Implementation of Machine Learning models.
- `matplotlib`: Industry-standard visualization library.
- `python-dotenv`: Environment variable management.

---

*Developed as part of the Talento Tech Project.*
