# Spatial Multi-Task Graph Neural Network for U.S. County Social Vulnerability Modeling

- [Spatial Multi-Task Graph Neural Network for U.S. County Social Vulnerability Modeling](#spatial-multi-task-graph-neural-network-for-us-county-social-vulnerability-modeling)
  - [Project Overview:](#project-overview)
  - [Problem Statement:](#problem-statement)
  - [Scope:](#scope)
  - [Dataset Description](#dataset-description)
  - [Setup Instructions:](#setup-instructions)
  - [Required Dependencies:](#required-dependencies)
  - [Data Download Instructions:](#data-download-instructions)
  - [Implementation Guide:](#implementation-guide)
  - [Limitations:](#limitations)
  - [Expected Contributions:](#expected-contributions)
  - [Meet The Team:](#meet-the-team)


## Project Overview:

This project models U.S. county-level social vulnerability using spatial deep learning. We construct a county adjacency graph from U.S. Census Bureau geographic data and implement multi-task Graph Neural Networks (GNNs) to jointly predict multiple Social Vulnerability Index (SVI) themes.

Unlike traditional tabular models that treat counties independently, our approach incorporates spatial relationships between neighboring counties to evaluate whether geographic structure improves predictive performance.


## Problem Statement:

We aim to predict multiple SVI theme scores jointly for each U.S. county using:

- County-level socioeconomic indicators

- Spatial adjacency information derived from official Census boundary data

We compare:

- Multi-Layer Perceptron (MLP) — Non-spatial baseline

- Graph Convolutional Network (GCN) — Spatial message-passing model

- GraphSAGE — Inductive graph aggregation model

We evaluate whether spatial modeling improves prediction of vulnerability themes relative to non-spatial baselines.


## Scope:

We focus on:

- Three architectures (MLP, GCN, GraphSAGE)

- Multi-task regression

- Spatial adjacency graph construction

- Rigorous evaluation and ablation


## Dataset Description

1. Social Vulnerability Data (2022): https://www.atsdr.cdc.gov/place-health/php/svi/svi-data-documentation-download.html 

Download Settings: 

- Year = 2022

- Geography = United States

- Geography Type = Counties

- File Type = CSV file

- Documentation: https://svi.cdc.gov/map25/data/docs/SVI2022Documentation_ZCTA.pdf 

County-level dataset containing approximately 3,100 counties and ~150 socioeconomic variables, including:

- Poverty

- Unemployment

- Income

- Disability

- Age distribution

- Minority status

- Housing type

- Transportation access

Targets (multi-task outputs):

- Theme 1: Socioeconomic Status

- Theme 2: Household Composition & Disability

- Theme 3: Minority Status & Language

- Theme 4: Housing Type & Transportation

Each theme score is treated as a continuous regression target.


2. U.S. Census Bureau TIGER/Line Shapefiles (2022): https://catalog.data.gov/dataset/tiger-line-shapefile-2022-nation-u-s-county-and-equivalent-entities/resource/1eb8657f-0109-4712-a714-32a569edc1ad? 

Official county boundary polygons are used to construct spatial adjacency.

Nodes: Counties
Edges: Counties that share a geographic border

Graph construction is performed using GeoPandas and spatial geometry operations.


## Setup Instructions:

- Clone Repo:
    - git clone <repo-url>
    - cd project

- Create Enviroment:
    - python -m venv venv
    - source venv/bin/activate
    - pip install -r requirements.txt


## Required Dependencies:
All dependencies are listed in requirements.txt.


## Data Download Instructions:

1. Social Vulnerability Dataset

    Download from the ATSDR:

    https://www.atsdr.cdc.gov/place-health/php/svi/svi-data-documentation-download.html 

    Download Settings: 

    - Year = 2022

    - Geography = United States

    - Geography Type = Counties

    - File Type = CSV file

    Place the SVI dataset in:

    - data/raw/

2. Census TIGER/Line County Shapefile (2022)

    Download from the U.S. Census Bureau TIGER/Line portal:

    https://www.census.gov/geographies/mapping-files/time-series/geo/tiger-line-file.html

    Download:

    - tl_2022_us_county.zip

    Unzip and place contents in:

    - data/raw/

3. Alternatively, a zip file with both datasets can be downloaded from this Google Drive: https://drive.google.com/file/d/1n2t0LzxWxXodHmJGttF33cSRit5VKht4/view?usp=sharing 


## Implementation Guide: 
All instructions are provided within each notebooks to preprocess, train, and evaluate.

1) EDA & Preprocessing (01_eda.ipynb)
   
2) Model Training:

   - Train & Evaluate MLP Baseline (02_baseline_mlp.ipynb)

   - Train & Evaluate GCN + GraphSAGE (03_gcn.ipynb)

3) Processed data are saved to data/processed
   
4) Evaluation Metrics are saved to data/artifacts/
   
5) Quarto for reporting rendering:

   - Install quarto from https://quarto.org/docs/download/index.html

   - Quarto install tinytex

   - cd to notebooks/ then run `quarto convert *.ipynb` to convert to qmd (repeat for each ipynb editted)

   - cd back to root and run `quarto render` to render from _quarto.yml

6) All quarto reports are saved to outputs/reports/


## Limitations:

- Cross-sectional (single-year) data

- Observational data only (no causal interpretation)

- County-level aggregation

- Spatial adjacency defined via border sharing only

This project is predictive and does not claim causal inference.


## Expected Contributions:

- Demonstration of spatial deep learning for public policy modeling

- Quantitative comparison of spatial vs. non-spatial methods

- Multi-task learning architecture design

- Reproducible research pipeline


## Meet The Team:

Joe Nguyen

Haesung Becker

Jared Lyon