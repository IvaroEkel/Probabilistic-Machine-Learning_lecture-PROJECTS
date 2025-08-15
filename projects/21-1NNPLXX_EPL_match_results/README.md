# Premier League Match Results Prediction

**Course:** Probabilistic Machine Learning (SoSe 2025)  
**Lecturer:** Dr. Alvaro Diaz-Ruelas  
**Students:** Niklas Nesseler, Lauren Pommer  
**PROJECT-ID:** 21-1NNPLXX_EPL_match_results  

## Overview
We predict English Premier League match outcomes (win/draw/loss) for the 2022/23–2023/24 seasons using features such as xG, possession, venue, and recent team form. Models compared: **SVM** and **XGBoost**.

## Data
- **Source:** [Kaggle – Premier League Matches](https://www.kaggle.com/datasets/mhmdkardosha/premier-league-matches)  
- Filtered to 1,520 recent matches  
- Engineered features: rolling form, home/away strength, goals/xG averages, possession, head-to-head stats  

## Results
| Model   | Accuracy | Notes                  |
|---------|----------|------------------------|
| SVM     | 66.45%   | Stable CV, failed draws |
| XGBoost | 67.11%   | Better draw prediction  |

Both outperform random guessing; team form is the most important feature.

## Limitations / Future Work
Improve draw prediction, add player-level & live data, test ensemble and deep learning models.
