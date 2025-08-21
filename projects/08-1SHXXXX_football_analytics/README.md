# Project Report – Football Analytics: Probabilistic Modelling of Expected Goals

This repository contains the results and materials for a Probabilistic Machine Learning project on expected goals (xG) modelling in football. The project compares Logistic Regression (LR), Bayesian Logistic Regression (BLR), and Bayesian Neural Networks (BNN) for predicting goal probabilities, with a focus on probability calibration and predictive uncertainty quantification. The complete methodology, experiments, and analyses are documented in the `notebooks/` folder, with all datasets and scripts linked for full reproducibility.

**Project Structure**  
- `notebooks/` – Experimental notebooks and scripts for preprocessing, training, and evaluation.  
- `data/` and `results/` – Due to size constraints, datasets and results for BLR and BNN are stored externally: [Google Drive – Data & Results](https://drive.google.com/drive/folders/1e4mFj6TXaXZ9wSUs3_TrCW1dcdXnWTZP?usp=sharing)  
- `report/` – Final written report.  

**Data Sources**  
Data are derived from publicly available football event datasets provided by [StatsBomb Open Data](https://github.com/statsbomb/open-data). Preprocessing steps are fully documented in the report and notebooks.

**Getting Started**  
You do not need to set up a virtual environment to explore this project. The notebooks can be opened interactively in **Google Colab** directly from the GitHub repository, allowing you to run all code without installing anything locally. Alternatively, you can clone the repository and open the notebooks in **Jupyter Notebook** or **JupyterLab** if you prefer to run them on your own machine.

**Clone the repository:**
```bash
git clone https://github.com/IvaroEkel/Probabilistic-Machine-Learning_lecture-PROJECTS.git
cd Probabilistic-Machine-Learning_lecture-PROJECTS
