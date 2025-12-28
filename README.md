# 🏆 Data Science Competition & Education Kit
![License: CC BY-NC 4.0](https://img.shields.io/badge/License-CC%20BY--NC%204.0-lightgrey.svg)
![Python 3.8+](https://img.shields.io/badge/Python-3.8%2B-blue.svg)
![Status: Active](https://img.shields.io/badge/Status-Active-brightgreen.svg)

> **A complete, turnkey ecosystem for hosting Data Science Hackathons, Bootcamps, and Training Sessions.**

---

## 📖 Table of Contents
- [Overview](#overview)
- [Who Is This For?](#who-is-this-for)
- [Repository Structure](#repository-structure)
- [The Pedagogical Approach](#the-pedagogical-approach)
- [The 20 Projects](#the-20-projects)
- [Technology Stack](#technology-stack)
- [How to Use This Repository](#how-to-use-this-repository)
- [Evaluation Standards](#evaluation-standards)
- [Appendix: Project Folder Contents](#appendix-project-folder-contents)
- [Notebook Generation (Scripts Folder)](#-notebook-generation-scripts-folder)
- [License](#license)

---

## Overview

This repository contains **20 curated Data Science Projects** designed to challenge students and professionals across all levels (Beginner to Advanced). Unlike standard datasets, this kit focuses on the **full pedagogical lifecycle**:

1.  **Real-World Problem**: Each project simulates a genuine business scenario (energy forecasting, fraud detection, box-office prediction, etc.).
2.  **Dirty Data**: Datasets intentionally include missing values, outliers, and inconsistencies to teach proper data cleaning.
3.  **Dual Notebooks**: Each project provides a **Beginner** (guided) and **Intermediate** (template-based) solution notebook.
4.  **Rigorous Evaluation**: A full evaluation kit (`Evaluation_Resources/`) ensures fair, consistent grading based on Data Science best practices.

---

## Who Is This For?

| Audience | How to Use This Kit |
|:---|:---|
| 🎓 **Bootcamp Instructors** | Use the Beginner notebooks as in-class walkthroughs. Assign the Intermediate notebooks as homework. Use the Evaluation Resources for grading. |
| 🏅 **Hackathon Organizers** | Distribute only `dataset.csv` and `taches.pdf`. Use the `scoring_guide.md` for the jury. |
| 💼 **Recruiters / Technical Assessors** | Assign a single project (e.g., P19 Fraud) as a take-home test. Evaluate using the `marking_sheet.md`. |
| 👨‍💻 **Self-Learners** | Start with the Beginner notebook, then try to replicate the result using only the Intermediate template. |

---

## Repository Structure

```text
mlProjects_Mehdi/
│
├── Projet_01/                     # Example: Solar Energy Forecasting
│   ├── energie_solaire.csv        # The raw dataset
│   ├── taches.pdf                 # The task sheet (business context, requirements)
│   ├── Projet_01_Debutant.ipynb   # Beginner: Guided solution with full code
│   ├── Projet_01_Intermediaire.ipynb # Intermediate: Template with instructions only
│   └── scripts/                   # 🛠️ Notebook Generation Scripts
│       ├── generer_notebook_debutant_projet_01.py
│       └── generer_notebook_intermediaire_projet_01.py
│
├── Projet_02/ to Projet_20/       # Same structure for all 20 projects
```

---

## The Pedagogical Approach

### 📓 Beginner Notebook (`*_Debutant.ipynb`)
The Beginner notebook is a **fully guided, step-by-step tutorial**. It is designed to be executed from top to bottom, with all code provided.

**Structure (Standard 3-Session, ~45 min each):**
1.  **Session 1: EDA & Cleaning**
    *   Load data with `pandas`.
    *   Identify and impute missing values (using median for robustness).
    *   Convert data types (e.g., `Horodatage` to `datetime`).
    *   Visualize distributions and correlations.

2.  **Session 2: Feature Engineering**
    *   Extract temporal features (Hour, Month, Day of Week).
    *   Create polynomial features (`Temperature ** 2`).
    *   Create interaction features (`Temperature * (100 - Cloud_Cover)`).

3.  **Session 3: Modeling & Evaluation**
    *   Split data **before** any preprocessing that could leak information.
    *   Train a model (e.g., `RandomForestRegressor`).
    *   Evaluate with business-relevant metrics (RMSE, R², MAE).
    *   Visualize feature importances.

**Key Characteristic:** Every code cell is **pre-filled**. Students run cells and observe outputs.

---

### 📄 Intermediate Notebook (`*_Intermediaire.ipynb`)
The Intermediate notebook is a **structured template**. It contains the same session structure and step names, but the code cells are empty, replaced with `# Votre code ici` (Your code here).

**Key Characteristics:**
*   **Instructions, Not Code:** Each step provides:
    *   The **Objective** (what you need to achieve).
    *   **Recommended Approaches** (functions, parameters).
    *   **Expected Deliverables** (what the output should look like).
*   **Pedagogical Goal:** Students must translate natural language instructions into working Python code.
*   **Evaluation Focus:** This is the version used to evaluate students in competitions/assessments.

---

## The 20 Projects

Below is a detailed breakdown of every project in the portfolio.

### Foundational Projects (P01-P05)
These projects focus on core Pandas skills, data cleaning, and basic regression/classification.

| ID | Name | Domain | Key Tech | Business Objective |
|:---|:---|:---|:---|:---|
| **01** | ⚡ Solar Energy Forecasting | Energy | `RandomForestRegressor`, Feature Engineering | Predict the next 24h of solar panel output based on weather data. |
| **02** | 💧 Water Pump Failure Prediction | Industry 4.0 | `RandomForestClassifier`, Sensor Data | Identify failing pumps *before* they break using sensor readings. |
| **03** | 😷 Air Quality & Health Impact | Environment | `LinearRegression`, Correlation Analysis | Model the relationship between pollution levels and hospital admissions. |
| **04** | 🎓 School Dropout Early Alert | Education | `LogisticRegression`, Social Data | Flag at-risk students for intervention before they leave school. |
| **05** | 💸 Micro-Credit Risk Scoring | Finance | `LogisticRegression`, Risk Analysis | Predict loan default probability for unbanked populations. |

---

### NLP & Text Projects (P06, P08)
These projects introduce text preprocessing and NLP classification.

| ID | Name | Domain | Key Tech | Business Objective |
|:---|:---|:---|:---|:---|
| **06** | 📰 Fake News Classifier | Media | `TfidfVectorizer`, `LogisticRegression` | Distinguish real news from misinformation. |
| **08** | 🧠 Mental Health Sentiment | Healthcare | `CountVectorizer`, Multi-class Classification | Detect emotional distress signals in text data. |

**Critical Workflow (tested in Evaluation):**
1.  `fit_transform()` the vectorizer **only on `X_train`**.
2.  `transform()` on `X_test`.
3.  **Never** fit on the full dataset—this causes **Data Leakage**.

---

### Time Series & Forecasting (P01, P07, P13, P17, P20)
These projects deal with temporal data and demand forecasting.

| ID | Name | Domain | Key Tech | Business Objective |
|:---|:---|:---|:---|:---|
| **07** | ♻️ Food Waste Reduction | Retail | Lag Features, Interaction Effects | Optimize fresh product stock to minimize waste. |
| **13** | 👥 Visitor Arrivals Forecasting | Tourism | Trend Analysis, Event Features | Predict tourist arrivals for resource planning. |
| **17** | 📦 Perishable Stock Optimizer | Logistics | Shelf-Life Constraints, Rolling Averages | Adjust inventory to avoid spoilage. |
| **20** | 🏥 Disease Outbreak Prediction | Healthcare | Early Indicators, Trend Detection | Anticipate epidemic spikes from early signals. |

---

### Advanced Classification & Imbalance (P09, P12, P15, P19)
These projects feature heavily imbalanced datasets where **Accuracy is a trap**.

| ID | Name | Domain | Key Tech | Business Objective |
|:---|:---|:---|:---|:---|
| **09** | 🚑 Road Accident Severity | Transport | Multiclass Classification, Factor Analysis | Predict accident severity to allocate emergency resources. |
| **12** | 💳 E-commerce Fraud | E-commerce | High-Volume Transactional Data, `RandomForest` | Block fraudulent purchases in real-time. |
| **15** | 🛎️ Hotel Cancellation Optimizer | Hospitality | Lead Time Analysis, Booking Patterns | Predict cancellations to optimize overbooking strategy. |
| **19** | 💳 Credit Card Fraud | Finance | **SMOTE (imbalanced-learn)**, Cost-Benefit Analysis, Recall Optimization | Detect rare (0.17%) fraud cases while minimizing false positives. |

**Critical Workflow for P19 (tested in Evaluation):**
1.  **Split** into Train/Test **first**.
2.  Apply SMOTE **only on `X_train`, `y_train`**.
3.  The Test set must remain **untouched and imbalanced** to reflect real-world conditions.

---

### Other Domain Projects (P10, P11, P14, P16, P18)

| ID | Name | Domain | Key Tech | Business Objective |
|:---|:---|:---|:---|:---|
| **10** | 🌍 Travel Recommendation | Tourism | User Profiling, Cold Start Problem | Suggest personalized travel destinations. |
| **11** | 🏠 Real Estate Undervaluation | Real Estate | Anomaly Detection, Regression Residuals | Find underpriced properties for investment. |
| **14** | 🚗 Used Car Valuation | Automotive | Categorical Encoding, Depreciation Curves | Estimate fair market value for used vehicles. |
| **16** | 🎬 Box Office Prediction | Entertainment | Log-Transformation, Skewed Targets, Feature Engineering | Predict a movie's commercial success. |
| **18** | 🚕 Ride-Sharing Hotspots | Transport | Geospatial Analysis (Lat/Long), Clustering | Guide drivers to high-demand pickup zones. |

**Critical Workflow for P16 (tested in Evaluation):**
1.  **Split** into Train/Test **first**.
2.  Impute missing budget values **using statistics from `X_train` only**.
3.  Apply the *same* imputation value to `X_test`.

---

## Technology Stack

All projects are built using the following standard Python libraries:

| Category | Libraries |
|:---|:---|
| **Data Manipulation** | `pandas`, `numpy` |
| **Visualization** | `matplotlib` (Seaborn is intentionally excluded to force deeper matplotlib practice) |
| **Machine Learning** | `scikit-learn` (`LinearRegression`, `LogisticRegression`, `RandomForestRegressor`, `RandomForestClassifier`) |
| **Imbalanced Data** | `imbalanced-learn` (`SMOTE`) — used specifically in P19 |
| **NLP (Text Processing)** | `scikit-learn` (`CountVectorizer`, `TfidfVectorizer`) |
| **Environment** | Jupyter Notebooks / Google Colab |

---

## How to Use This Repository

### For Competition Organizers
1.  **Select a Project**: Choose based on difficulty and domain.
2.  **Distribute Materials**: Give participants **only**:
    *   `dataset.csv`
    *   `taches.pdf`
    *   (Optional) `*_Intermediaire.ipynb` as a starter template.

### For Self-Learners
1.  **Start with Beginner**: Open `*_Debutant.ipynb` and run every cell. Understand *why* each step is taken.
2.  **Attempt Intermediate**: Open `*_Intermediaire.ipynb`. Try to write the code yourself using only the instructions.
3.  **Compare**: If stuck, refer back to the Beginner notebook to see the correct implementation.

---

## Evaluation Standards

These projects are designed to catch common Data Science pitfalls. The Evaluation Resources grade students on **Methodological Rigor**, not just accuracy.

### Core Evaluation Criteria (Applied to ALL Projects):
| Criterion | What We Check |
|:---|:---|
| 🚫 **Data Leakage Prevention** | Did they impute/vectorize/scale **after** the Train/Test split, using only Train data? |
| 📊 **Proper EDA** | Did they explore the data before modeling (distributions, correlations, missing values)? |
| 📉 **Correct Metrics** | Did they use **Recall** for Fraud, **RMSE** for Regression, **F1** for imbalanced classification (not raw Accuracy)? |
| 🧹 **Data Cleaning** | Did they handle NaNs, outliers, and incorrect values (e.g., `' Class'` typos)? |
| ⚙️ **Feature Engineering** | Did they create meaningful new features (temporal, polynomial, interaction)? |

---

## Appendix: Project Folder Contents

Each project folder (`Projet_XX/`) contains the following files:

| File | Description |
|:---|:---|
| `dataset.csv` | The raw dataset (e.g., `energie_solaire.csv`, `fraude_carte_credit.csv`) |
| `taches.pdf` | The **Task Sheet** given to students with business context and requirements |
| `Projet_XX_Debutant.ipynb` | Beginner notebook: Fully guided solution with all code provided |
| `Projet_XX_Intermediaire.ipynb` | Intermediate notebook: Template with empty code cells |
| `scripts/` | 🛠️ Python scripts to regenerate the notebooks programmatically |

### What's in `taches.pdf`?

1.  **Business Context**: A short story explaining *why* this analysis is needed (e.g., "The grid operator needs to predict solar output to balance load").
2.  **Dataset Description**: Column definitions, data types, and expected target variable.
3.  **Main Task (Tâche Principale)**: The core objective (e.g., "Build a regression model to predict `Production_Energie`").
4.  **Secondary Tasks (Tâches Secondaires)**: Bonus objectives for advanced students (e.g., "Which hour of the day is most productive?").
5.  **Expected Features/Results**: Links to `FeaturesPrevus.pdf` or `ResultatPrevus.pdf` for specific targets.

**Purpose:** The `taches.pdf` is the *only document* given to participants in a competition setting. It deliberately omits code hints—forcing students to design their own solution.

---

## 🛠️ Notebook Generation (Scripts Folder)

**For Contributors & Maintainers:**

Each project folder now includes a `scripts/` directory containing Python scripts to automatically generate the `.ipynb` notebooks. This ensures consistency between the Beginner and Intermediate versions and allows for rapid iteration on content.

### Scripts Folder Structure

```text
Projet_XX/scripts/
├── generer_notebook_debutant_projet_XX.py      # Generates the Beginner notebook
├── generer_notebook_intermediaire_projet_XX.py # Generates the Intermediate notebook
└── taches_content.txt                          # (Optional) Task content for reference
```

### How the Generators Work

The generator scripts use the `nbformat` library to programmatically create Jupyter notebooks:

```python
import nbformat as nbf

def generer_notebook_debutant():
    nb = nbf.v4.new_notebook()
    
    # Add markdown cells for explanations
    nb.cells.append(nbf.v4.new_markdown_cell("""
    # 📊 Project Title
    ## Introduction and objectives...
    """))
    
    # Add code cells with pre-filled solutions
    nb.cells.append(nbf.v4.new_code_cell("""
    import pandas as pd
    df = pd.read_csv('dataset.csv')
    """))
    
    # Save the notebook
    nbf.write(nb, 'Projet_XX_Debutant.ipynb')
```

### What the Generators Include

| Component | Beginner Version | Intermediate Version |
|:---|:---|:---|
| **Theory Explanations** | ✅ Full markdown explanations | ✅ Same explanations |
| **Code Cells** | ✅ Complete, runnable code | ❌ Empty with `# Votre code ici` |
| **TODO Sections** | ✅ Practice exercises with hints | ✅ Main task (no hints) |
| **Session Structure** | ✅ 3 sessions (~45 min each) | ✅ Same structure |
| **Bonus Tasks** | ✅ Extra challenges | ✅ Same challenges |

### How to Regenerate Notebooks

To regenerate the notebooks for any project, run the corresponding script. Replace `XX` with the project number (e.g., `03`, `17`, `19`):

```bash
# Navigate to the project folder
cd Projet_XX

# Generate Beginner Notebook
python scripts/generer_notebook_debutant_projet_XX.py

# Generate Intermediate Notebook
python scripts/generer_notebook_intermediaire_projet_XX.py
```

**Or from the repository root:**

```bash
python Projet_XX/scripts/generer_notebook_debutant_projet_XX.py
python Projet_XX/scripts/generer_notebook_intermediaire_projet_XX.py
```

> **Note:** This will overwrite the existing `.ipynb` files in the project folder with the latest content defined in the scripts. Always commit your changes before regenerating.

### Why Use Generator Scripts?

1. **Consistency**: Ensures Beginner and Intermediate notebooks follow identical session structures.
2. **Version Control**: Python scripts are easier to diff and review than JSON-based `.ipynb` files.
3. **Rapid Iteration**: Update content once in the script, regenerate both notebook versions.
4. **Collaboration**: Multiple contributors can work on different sections without merge conflicts.
5. **Quality Control**: The same pedagogical structure is enforced across all 20 projects.

---

## License

This repository is licensed under **Creative Commons Attribution–NonCommercial 4.0 International (CC BY-NC 4.0)**.

You are free to use, modify, and share this material **for learning, teaching, and academic practice only**.

❌ Commercial use, paid courses, resale, or monetization of this content (or derivatives) is not permitted.
