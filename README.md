# County-Level Spatial Graph Modeling of Social Vulnerability and FEMA Disaster Assistance (2018–2020)

- [County-Level Spatial Graph Modeling of Social Vulnerability and FEMA Disaster Assistance (2018–2020)](#county-level-spatial-graph-modeling-of-social-vulnerability-and-fema-disaster-assistance-20182020)
- [Overview](#overview)
- [Abstract](#abstract)
- [Project Contributions](#project-contributions)
- [Methodology](#methodology)
  - [1. Exploratory Data Analysis (01\_eda.ipynb)](#1-exploratory-data-analysis-01_edaipynb)
  - [2. Baseline Deep Learning Model (02\_mlp.ipynb)](#2-baseline-deep-learning-model-02_mlpipynb)
  - [3. Spatial Graph Modeling (03\_gcn.ipynb)](#3-spatial-graph-modeling-03_gcnipynb)
- [Data Sources](#data-sources)
  - [SVI Data Summary](#svi-data-summary)
  - [FEMA Disaster Assistance Data Summary](#fema-disaster-assistance-data-summary)
    - [Fields Retrieved](#fields-retrieved)
    - [Target Variable Construction](#target-variable-construction)
- [Implementation](#implementation)
  - [Data Download Instructions:](#data-download-instructions)
- [Meet The Team:](#meet-the-team)
- [References](#references)


# Overview

This project investigates how **county-level social vulnerability indicators relate to disaster assistance outcomes** and whether incorporating spatial relationships between counties improves predictive modeling. Rather than reproducing the Social Vulnerability Index (SVI) itself, the study evaluates how the underlying socioeconomic indicators captured by SVI explain variation in disaster recovery assistance across counties.

The analysis begins with **exploratory data analysis (EDA)** to examine the statistical distributions, feature relationships, and geographic patterns of the SVI indicators. A **Multilayer Perceptron (MLP)** is then implemented as a deep learning baseline to model nonlinear relationships between vulnerability indicators and disaster assistance outcomes using tabular data alone.

To incorporate geographic context, counties are represented as nodes in a spatial graph based on adjacency relationships, allowing neighboring counties to influence each other during model training. A **Graph Convolutional Network (GCN)** is subsequently applied to evaluate whether spatial neighborhood aggregation improves predictive performance.

Finally, the project compares **tabular deep learning and graph-based models within a unified regression framework** to assess the role of spatial connectivity in disaster vulnerability modeling and determine whether geographic structure provides predictive signal beyond the socioeconomic indicators themselves.

Full Report: https://github.com/AnhJoe/svi-fema-spatial-graph-modeling/releases/tag/v1.1

# Abstract

This study investigates the relationship between county-level social vulnerability indicators and the distribution of disaster assistance using machine learning and spatial modeling techniques. Using the **2018 CDC/ATSDR Social Vulnerability Index (SVI)** indicators as predictors and **FEMA Individual Household Program (IHP) assistance aggregations from 2018-2020** as the target outcome, the project evaluates whether incorporating spatial structure improves predictive performance. A nonlinear multilayer perceptron (MLP) baseline and a graph convolutional network (GCN) are implemented within a unified regression framework to compare tabular and spatial learning approaches.

The analysis uses log-transformed socioeconomic indicators, standardized features, and fixed train/validation/test splits to ensure consistent evaluation. Hyperparameters for both models are selected through controlled ablation experiments, and the best-performing configurations are retrained before final evaluation on a held-out test set.

Results show that the **MLP baseline outperforms the spatial GCN model** across evaluation metrics. The best MLP configuration achieves a test RMSE of approximately 2.00 and an $R^2$ of 0.57, while the GCN achieves a test RMSE of approximately 2.38 and an $R^2$ of 0.40. These findings indicate that socioeconomic vulnerability indicators explain a substantial portion of variation in disaster assistance outcomes, but that incorporating county adjacency through graph convolution does not improve predictive performance in this setting.

Overall, the results suggest that while social vulnerability indicators provide meaningful signal for predicting disaster assistance outcomes, additional spatial modeling does not necessarily yield improvements when the underlying spatial dependency in the target variable is limited. The study provides a reproducible modeling framework for integrating vulnerability indicators, disaster assistance data, and graph-based learning methods, while highlighting the importance of strong tabular baselines when evaluating spatial machine learning approaches.


# Project Contributions

This project contributes both practical insights into disaster vulnerability analysis and technical advancements in modeling spatial socioeconomic data:

1) **Empirical evaluation of vulnerability indicators and disaster assistance outcomes.**
  The study quantifies the extent to which county-level social vulnerability indicators correspond to the distribution of FEMA Individual Household Program assistance from 2018-2020, providing evidence that socioeconomic vulnerability explains a substantial portion of variation in disaster aid outcomes.

2) **Evidence-based comparison of spatial and non-spatial predictive models.**
  The results demonstrate that a well-tuned tabular neural network can outperform a spatial graph neural network for this task, offering practical guidance for analysts deciding whether spatial modeling is necessary for similar socioeconomic datasets.

3) **Reproducible analytical pipeline for vulnerability modeling.**
  The project establishes a transparent workflow for integrating CDC SVI indicators, county adjacency graphs, and disaster assistance outcomes into a consistent machine learning evaluation framework.

Together, these contributions provide a methodologically rigorous and practically relevant framework for studying how social vulnerability indicators relate to disaster outcomes, while also illustrating the strengths and limitations of spatial deep learning approaches for structured socioeconomic datasets.

# Methodology

The analysis follows a structured pipeline:

## 1. Exploratory Data Analysis (01_eda.ipynb)

Several analyses were conducted to understand the statistical and geographic structure of the dataset prior to model development:

- **Feature Distribution Analysis**  
  Examine the distributions of the SVI percentage indicators using histograms and summary statistics.

- **Log Feature Transformation**  
  Apply log transformations to reduce skewness and stabilize variance for modeling.

- **Correlation Analysis**  
  Compute Pearson correlation matrices to assess relationships among vulnerability indicators and identify potential multicollinearity.

- **Unsupervised Clustering & Principal Component Analysis (PCA)**  
  Analyze the variance structure of the indicators using PCA to evaluate whether a smaller set of latent components explains most of the variability in the dataset. Clustering techniques are also applied to explore whether counties naturally group into distinct vulnerability profiles based on their socioeconomic characteristics.

## 2. Baseline Deep Learning Model (02_mlp.ipynb)

A **Multilayer Perceptron (MLP)** was implemented as a baseline model for predicting county-level disaster assistance outcomes using tabular SVI indicators.

Reasons for selecting an MLP baseline:

- Captures nonlinear relationships between socioeconomic indicators and outcomes  
- Well-suited for tabular regression tasks  
- Provides a deep learning benchmark before introducing spatial structure

The model consists of a fully connected feedforward architecture with ReLU activations, trained using the Adam optimizer and Mean Squared Error (MSE) loss.

## 3. Spatial Graph Modeling (03_gcn.ipynb)

To incorporate geographic structure, counties are represented as nodes in a spatial graph constructed using **Queen contiguity adjacency**, where neighboring counties share an edge.

A **Graph Convolutional Network (GCN)** is applied to learn county representations by aggregating information from neighboring counties. This enables the model to capture localized spatial dependencies in vulnerability indicators that may influence disaster assistance outcomes.


# Data Sources

1. US County 2018 SVI dataset:  
https://www.atsdr.cdc.gov/place-health/php/svi/svi-data-documentation-download.html

2. US County 2018 SVI dataset documentation:  
https://svi.cdc.gov/map25/data/docs/SVI2018Documentation_01192022_1.pdf

3. US County 2018 TIGER/Line county boundaries:  
https://www2.census.gov/geo/tiger/TIGER2018/COUNTY/

4. FEMA Individuals and Households Program disaster assistance data API (2018–2020):  
https://www.fema.gov/api/open/v2/IndividualsAndHouseholdsProgramValidRegistrations

The TIGER/Line geographic data provide county boundary geometries used to construct spatial adjacency relationships for the graph-based models developed later in the study.

## SVI Data Summary

The county-level **2018 Social Vulnerability Index (SVI)** dataset is a relative vulnerability dataset designed to identify counties that may be more vulnerable before, during, and after hazardous events. The index was developed by the Centers for Disease Control and Prevention and the Agency for Toxic Substances and Disease Registry to support disaster preparedness, response planning, and public health resource allocation (Flanagan et al., 2011; CDC/ATSDR, 2024). The dataset integrates socioeconomic indicators derived primarily from the U.S. Census Bureau American Community Survey and converts them into percentile-based vulnerability rankings across U.S. counties.

In the official documentation, SVI variables are organized through a structured pipeline of derived fields. These include raw estimate fields (E_), margin-of-error fields (M_), derived percentage fields (EP_), percentage margin-of-error fields (MP_), percentile-ranked indicator fields (EPL_), theme sums (SPL_), theme percentile rankings (RPL_THEME1 through RPL_THEME4), and the overall percentile ranking (RPL_THEMES). Higher percentile values correspond to **greater relative social vulnerability compared with other counties**.

In this study, the SVI dataset is used as a **feature dataset representing underlying social and demographic conditions at the county level**. The analysis uses the underlying socioeconomic indicators as predictors to examine whether patterns of social vulnerability are associated with **disaster assistance outcomes observed during the 2018–2020 period**. This framing allows the study to evaluate whether vulnerability indicators measured in 2018 correspond to subsequent disaster impacts and recovery assistance across U.S. counties.

## FEMA Disaster Assistance Data Summary

Disaster assistance outcomes used in this study are obtained directly from the **Federal Emergency Management Agency (FEMA) OpenFEMA API**, specifically the *Individuals and Households Program (IHP) Valid Registrations* dataset. This dataset records household-level disaster assistance provided to individuals following federally declared disasters and includes information on financial assistance amounts, geographic location, and disaster identifiers.

The dataset is accessed programmatically using the OpenFEMA API endpoint. Using the API allows the study to retrieve only the fields necessary for analysis and filter records by date, avoiding the need to download the full dataset. To maintain a clear temporal ordering between predictors and outcomes, the analysis uses:

- SVI predictors from 2018
- Disaster assistance outcomes from 2018–2020

This design evaluates whether social vulnerability indicators measured in 2018 correspond to disaster recovery assistance observed during the subsequent three-year period.

### Fields Retrieved

The following fields are retrieved directly from the OpenFEMA API:

- **disasterNumber** – FEMA disaster identifier  
- **incidentBeginDate** – date when the disaster incident began  
- **fips** – county-level FIPS geographic identifier  
- **ihpAmount** – total FEMA Individuals and Households Program assistance awarded for a registration  

The ihpAmount field represents the total financial assistance provided to a household through FEMA’s Individual Assistance programs, which may include housing repair, replacement, rental assistance, personal property assistance, and other eligible recovery costs.

### Target Variable Construction

Because the FEMA dataset records assistance at the individual registration level, the data are aggregated to the **county level** using the FIPS geographic identifier. The final modeling target is constructed by summing all **ihpAmount** values within each county across the 2018–2020 period. Formally, the county-level disaster assistance target is defined as: County Assistance = Σ ihpAmount (for all registrations in a county from 2018–2020).

This produces a **county-level measure of total disaster assistance received during 2018-2022**, which serves as the regression target for the study's ML models. The FEMA dataset also includes county-level **FIPS codes**, allowing the aggregated assistance outcomes to be merged directly with the SVI dataset and county geometries. This shared geographic identifier enables the integration of socioeconomic vulnerability indicators, disaster assistance outcomes, and spatial adjacency relationships within a unified county-level dataset used for modeling.

# Implementation

1. Create your virtual environment

2. Install dependencies in requirements.txt

3. Run the ipynb notebooks in notebooks/ in order

4. Raw data is downloaded and saved into data/raw

5. Processed data is saved into data/processed

6. Artifacts/metrics are saved into data/artifacts

7. If changes are made to ipynb notebooks and you wish to render quarto reports,

- Install quarto at https://quarto.org/docs/download/index.html 
- Run: quarto `install tinytex`
- cd to notebooks/ and run: `quarto convert *.ipynb` to convert ipynb to qmd
- Run: `quarto render --to pdf` to render pdf report based on _quarto.yml (this will take some time)
- Rendered reports are saved into outputs/reports 

8. Changes to introduction section can be made in index.qmd

9. Changes to conclusion section can be made in notebooks/04_conclusion.qmd 


## Data Download Instructions:

1. Social Vulnerability Dataset

    Download from the ATSDR:

    https://www.atsdr.cdc.gov/place-health/php/svi/svi-data-documentation-download.html 

    Download Settings: 

    - Year = 2018

    - Geography = United States

    - Geography Type = Counties

    - File Type = CSV file

    Place the SVI dataset in:

    - data/raw/

2. Census TIGER/Line County Shapefile (2022)

    Download from the U.S. Census Bureau TIGER/Line portal:

    https://www2.census.gov/geo/tiger/TIGER2018/COUNTY/

    Download:

    - tl_2018_us_county.zip

    Unzip and place contents in:

    - data/raw/

3. FEMA Individuals and Households Program disaster assistance data API (2018–2020):  

    Run notebooks/01_eda.ipynb to download required fields from OpenFEMA API 

    Note: This may take some time

4. Alternatively, a zip file with all datasets can be downloaded from this release: https://github.com/AnhJoe/svi-fema-spatial-graph-modeling/releases/tag/v1.0


# Meet The Team:

Joe Nguyen

Haesung Becker

Jared Lyon

# References

1) Centers for Disease Control and Prevention/Agency for Toxic Substances and Disease Registry. (2024). A validity assessment of the Centers for Disease Control and Prevention/Agency for Toxic Substances and Disease Registry Social Vulnerability Index (CDC/ATSDR SVI). U.S. Department of Health and Human Services. 

2) Federal Emergency Management Agency. (2020). COVID-19 pandemic: Emergency declaration under the Stafford Act. https://www.fema.gov
   
3) Federal Emergency Management Agency. (n.d.). Disaster declarations summaries (v2). OpenFEMA. https://www.fema.gov/openfema-data-page/disaster-declarations-summaries-v2
   
4) Flanagan, B. E., Gregory, E. W., Hallisey, E. J., Heitgerd, J. L., & Lewis, B. (2011). A social vulnerability index for disaster management. Journal of Homeland Security and Emergency Management, 8(1), Article 3.

5) Hamilton, W. (2020). Graph Representation Learning. Morgan & Claypool. https://www.cs.mcgill.ca/~wlh/grl_book/files/GRL_Book.pdf
   
6) Kalaycioglu, O., Akhanli, S. E., Mentese, E. Y., Kalaycioglu, M., & Kalaycioglu, S. (n.d.). Using machine learning algorithms to identify predictors of social vulnerability in the event of a hazard: Istanbul case study. 

7) Kipf, T. N., & Welling, M. (2017). Semi-supervised classification with graph convolutional networks. International Conference on Learning Representations (ICLR). https://arxiv.org/pdf/1609.02907 
   
8) Tarling, H. A. (2017). Comparative analysis of social vulnerability indices: CDC’s SVI and SoVI® [Master’s thesis, Lund University]. 

9)  Robert T. Stafford Disaster Relief and Emergency Assistance Act, 42 U.S.C. §§ 5121–5207 (1988).
   
10) Russell, S. J., & Norvig, P. (2021). Artificial intelligence: A modern approach (4th ed.). Pearson. https://api.pageplace.de/preview/DT0400.9781292401171_A41586057/preview-9781292401171_A41586057.pdf 

11) Yedinak, J. L., Li, Y., Krieger, M. S., Howe, K., Daley Ndoye, C., Lee, H., Civitarese, A. M., Marak, T., Nelson, E., Samuels, E. A., Chan, P. A., Bertrand, T., & Marshall, B. D. L. (2021). Machine learning takes a village: Assessing neighbourhood-level vulnerability for an overdose and infectious disease outbreak. International Journal of Drug Policy, 96, 103395.

12) Yokoyama, H., & Takefuji, Y. (2026). Unbiased evaluation of social vulnerability: A multimethod approach using machine learning and nonparametric statistics. Cities, 168, 106519.

13) Zhao, Y., Paul, R., Reid, S., Coimbra Vieira, C., Wolfe, C., Zhang, Y., & Chunara, R. (2024). Constructing social vulnerability indexes with increased data and machine learning highlight the importance of wealth across global contexts. Social Indicators Research, 175, 639–657.