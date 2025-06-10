# Project Report - Classification of Political Content on Instagram

## Abstract

This project investigates the classification of Instagram posts from major German political parties based on their dominant colors. A dataset of approximately 28,500 publicly available Instagram posts, collected from the inception of each party's account up to December 18, 2024, was utilized. For each image, the four most dominant colors were extracted and transformed into compact feature vectors. These features were then used to train various classifiers. The study aimed to assess the effectiveness of simple visual features like dominant colors in distinguishing between political parties and to understand how different feature representations and preprocessing strategies impact classification performance. The analysis involved evaluating different color spaces, incorporating temporal information, and applying selective data filtering to optimize model accuracy.

## How to Use This Repository

This repository contains the results and related materials for the Probabilistic Machine Learning project.

The core of this project, including a comprehensive description of the methodology, detailed results, and analyses, is presented in a Jupyter Notebook.

**To access the full project details and results, please navigate to the [`results`](https://github.com/IvaroEkel/Probabilistic-Machine-Learning_lecture-PROJECTS/tree/d2602fefa8540ad79bd793247f67058e6c129701/projects/05-1PLXXXX_political_color_posts/results) folder and open the [`project_report_lukas_pasold.ipynb`](https://github.com/IvaroEkel/Probabilistic-Machine-Learning_lecture-PROJECTS/blob/d2602fefa8540ad79bd793247f67058e6c129701/projects/05-1PLXXXX_political_color_posts/results/project_report_lukas_pasold.ipynb) notebook.**

Within this notebook, all other necessary data and code files used in the project are linked. You can easily access and review these files by clicking on the respective links within the notebook.

## Project Structure

-   [`results/`](https://github.com/IvaroEkel/Probabilistic-Machine-Learning_lecture-PROJECTS/tree/d2602fefa8540ad79bd793247f67058e6c129701/projects/05-1PLXXXX_political_color_posts/results): This directory contains the main project report notebook ([`project_report_lukas_pasold.ipynb`](https://github.com/IvaroEkel/Probabilistic-Machine-Learning_lecture-PROJECTS/blob/d2602fefa8540ad79bd793247f67058e6c129701/projects/05-1PLXXXX_political_color_posts/results/project_report_lukas_pasold.ipynb)) and any generated visualizations or output data.
    
-   [`notebooks/`](https://github.com/IvaroEkel/Probabilistic-Machine-Learning_lecture-PROJECTS/tree/d2602fefa8540ad79bd793247f67058e6c129701/projects/05-1PLXXXX_political_color_posts/notebooks): This directory contains scripts and notebooks used for data collection, processing and classification.
    
-   [`data/`](https://github.com/IvaroEkel/Probabilistic-Machine-Learning_lecture-PROJECTS/tree/d2602fefa8540ad79bd793247f67058e6c129701/projects/05-1PLXXXX_political_color_posts/data): This directory contains the processed image data as .csv files. Due to the size constraints, the full image dataset is not directly included but links to the original sources are provided.
    

## Data Sources

The full image dataset used in this analysis was collected from the publicly available Instagram pages of the following German political parties:

-   [CDU](https://www.instagram.com/cdu/ "null")
    
-   [CSU](https://www.instagram.com/csu/ "null")
    
-   [SPD](https://www.instagram.com/spd_de/ "null")
    
-   [FDP](https://www.instagram.com/fdp/ "null")
    
-   [Bündnis 90/Die Grünen](https://www.instagram.com/die_gruenen/ "null")
    
-   [Die Linke](https://www.instagram.com/dielinke/ "null")
    
-   [AfD](https://www.instagram.com/afd.bund/ "null")
    

## Getting Started

1.  Clone the Repository:
    
    ```
    git clone https://github.com/IvaroEkel/Probabilistic-Machine-Learning_lecture-PROJECTS.git cd Probabilistic-Machine-Learning_lecture-PROJECTS/projects/05-1PLXXXX_political_color_posts    
    ```
    
2.  Open the Notebook:
    
-   **GitHub:** You can view the notebook directly on GitHub, though you won't be able to run the code.
    
-   **Google Colab:** The most convenient way to run the notebook is often through Google Colab. Look for a "Open in Colab" badge or link within the notebook's GitHub preview.
    
-   **Jupyter Notebook/JupyterLab (Local):** If you prefer to work locally, you'll need to have Jupyter installed. If you don't, you can install it via pip.

## Dependencies

The project relies on standard Python libraries for data manipulation, machine learning, and visualization. Please refer to the [`requirements.txt`](https://github.com/IvaroEkel/Probabilistic-Machine-Learning_lecture-PROJECTS/blob/d2602fefa8540ad79bd793247f67058e6c129701/projects/05-1PLXXXX_political_color_posts/requirements.txt).
    

It is recommended to create a virtual environment for the project:

```
python -m venv venv
source venv/bin/activate  # On Windows: `venv\Scripts\activate`
pip install -r requirements.txt 
```