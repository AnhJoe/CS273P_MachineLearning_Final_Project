# Spatial Multi-Task Graph Neural Network for U.S. County Social Vulnerability Modeling

- [Spatial Multi-Task Graph Neural Network for U.S. County Social Vulnerability Modeling](#spatial-multi-task-graph-neural-network-for-us-county-social-vulnerability-modeling)
  - [Project Overview](#project-overview)
  - [Problem Statement](#problem-statement)
  - [Dataset Description](#dataset-description)
  - [Repository Structure](#repository-structure)
  - [Setup Instructions](#setup-instructions)
  - [Required Dependencies](#required-dependencies)
  - [Data Download Instructions](#data-download-instructions)
  - [Preprocessing](#preprocessing)
  - [Training the Models](#training-the-models)
  - [Evaluation](#evaluation)
  - [Experimental Design](#experimental-design)
  - [Limitations](#limitations)
  - [Scope](#scope)
  - [Expected Contributions](#expected-contributions)
  - [Team Contributions](#team-contributions)
  - [Guidelines:](#guidelines)

## Project Overview

This project models U.S. county-level social vulnerability using spatial deep learning. We construct a county adjacency graph from U.S. Census Bureau geographic data and implement multi-task Graph Neural Networks (GNNs) to jointly predict multiple Social Vulnerability Index (SVI) themes.

Unlike traditional tabular models that treat counties independently, our approach incorporates spatial relationships between neighboring counties to evaluate whether geographic structure improves predictive performance.

## Problem Statement

We aim to predict multiple SVI theme scores jointly for each U.S. county using:

- County-level socioeconomic indicators

- Spatial adjacency information derived from official Census boundary data

We compare:

- Multi-Layer Perceptron (MLP) — Non-spatial baseline

- Graph Convolutional Network (GCN) — Spatial message-passing model

- GraphSAGE — Inductive graph aggregation model

We evaluate whether spatial modeling improves prediction of vulnerability themes relative to non-spatial baselines.

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

## Repository Structure
project/
│
├── data/
│   ├── raw/
│   ├── processed/
│
├── notebooks/
│   └── demo.ipynb
│
├── src/
│   ├── models/
│   │   ├── mlp.py
│   │   ├── gcn.py
│   │   └── graphsage.py
│   │
│   ├── training/
│   │   ├── train.py
│   │   └── trainer.py
│   │
│   ├── evaluation/
│   │   ├── metrics.py
│   │   └── analysis.py
│   │
│   ├── graph/
│   │   └── build_graph.py
│   │
│   └── utils/
│
├── experiments/
├── configs/
├── requirements.txt
├── README.md
└── report/

## Setup Instructions

- Clone Repo:
    - git clone <repo-url>
    - cd project

- Create Enviroment:
    - python -m venv venv
    - source venv/bin/activate
    - pip install -r requirements.txt

## Required Dependencies

- Python 3.10+

- PyTorch

- PyTorch Geometric

- GeoPandas

- Shapely

- NetworkX

- NumPy

- Pandas

- Scikit-learn

- Matplotlib

- Seaborn

All dependencies are listed in requirements.txt.

## Data Download Instructions

1. Social Vulnerability Dataset

Place the SVI dataset in:

- data/raw/

Ensure it includes:

- County FIPS codes
  
- Theme percentile scores
  
- Socioeconomic feature columns

1. Census TIGER/Line County Shapefile (2022)

Download from the U.S. Census Bureau TIGER/Line portal:

https://www.census.gov/geographies/mapping-files/time-series/geo/tiger-line-file.html

Download:

- tl_2022_us_county.zip

Unzip and place contents in:

- data/raw/tiger/

## Preprocessing

Run:

- python src/graph/build_graph.py


This will:

- Load shapefiles
  
- Construct adjacency graph
  
- Merge SVI features
  
- Save processed graph object

Output:

- data/processed/graph_data.pt

## Training the Models

Train MLP Baseline

-python src/training/train.py --model mlp

Train GCN

- python src/training/train.py --model gcn

Train GraphSAGE

- python src/training/train.py --model graphsage


Training logs and metrics are saved to:

- experiments/

## Evaluation

Models are evaluated using:

- Mean Squared Error (MSE)
  
- Mean Absolute Error (MAE)
  
- R²
  
Spatial ablation experiments include:
  
- Removing graph edges
  
- Random graph rewiring
  
- Feature group ablation

Evaluation results are saved as:

- CSV metric tables
  
- Learning curve plots
  
- Comparative performance charts


## Experimental Design

We perform:

- Baseline vs. Spatial model comparison
  
- Hyperparameter tuning
  
- Multi-task loss balancing
  
- Spatial ablation studies
  
- Regional error analysis
  
- Reproducing Results

To reproduce main experiments:

- python src/training/train.py --config configs/final_config.yaml


Expected outputs:

- Trained model weights
  
- Performance metrics
  
- Training curves
  
- Evaluation tables

Demo Notebook

notebooks/demo.ipynb provides:

- Step-by-step walkthrough
  
- Graph construction visualization
  
- Model inference example
  
- Sample evaluation results

## Limitations

- Cross-sectional (single-year) data

- Observational data only (no causal interpretation)

- County-level aggregation

- Spatial adjacency defined via border sharing only

This project is predictive and does not claim causal inference.

## Scope

We focus on:

- Three architectures (MLP, GCN, GraphSAGE)
- 
- Multi-task regression
- 
- Spatial adjacency graph construction
- 
- Rigorous evaluation and ablation

We do not include:

- Temporal modeling

- Advanced causal inference

- Large-scale hyperparameter sweeps

- National-scale simulation studies

## Expected Contributions

- Demonstration of spatial deep learning for public policy modeling
- 
- Quantitative comparison of spatial vs. non-spatial methods
- 
- Multi-task learning architecture design
- 
- Reproducible research pipeline

## Team Contributions

Joe Nguyen

Haesung Becker

Jared Lyon

## Guidelines:

This project is intended to give you hands-on experience in:
- Formulating a concrete ML problem

- Designing appropriate models

- Implementing the solution

- Training and debugging models

- Evaluating performance rigorously

- Communicating results clearly

- Expected Level of Sophistication

The project must demonstrate sufficient technical depth.

❌ Out of Scope:

- Simply running a sklearn model

- Using off-the-shelf decision trees, e.g., random forest, XGBoosting etc without meaningful modification or analysis

✅ Expected Level:

- Develop and implement deep learning models using PyTorch

- Design custom architectures, loss functions, or training strategies

- Perform thoughtful experimentation and ablation studies

- Analyze results carefully

- You are encouraged to explore modern techniques such as neural nets, CNNs, RNNs, transformers, deep generative models, graph neural nets, LLMs. 

Use of Open-Source Code:

- You do not need to build everything from scratch.

- You must clearly document what code is borrowed.

- You must demonstrate meaningful contribution beyond simply running someone else’s code.

Submission Requirements:

(A) Final Written Report: a single write-up document in PDF, approximately 4-6 pages long

- Problem definition and motivation: describe the problem you chose and the methods you used to address it

- Related work

- Dataset description

- Model design

- Training procedure

- Evaluation metrics

- Experimental results

- Analysis and discussion: 
    - Which model(s) you tried, how you trained them, how you selected any parameters they might require, and how they performed on the test data

    - Tables of performance of different approaches, or plots of performance used to perform model selection (i.e., parameters that control complexity)

- Limitations and future work

(B) GitHub Repository:

- A README.md file that clearly explains:

- Project overview

- Setup instructions

- Required dependencies

- How to train the model

- How to evaluate the model

- Expected outputs

Clear instructions on:

- Where to download the dataset

- How to preprocess the data

- How to reproduce your results

- A small sample dataset (if the real dataset is private)

- A demo notebook (e.g., Jupyter notebook) that allows us to run the code, test the model, and reproduce key results

(C) In the last paragraph of the report, please write down the names of your team members and contributions from each member.

- Try to describe to the best of your ability who was responsible for which aspects (which learners, etc.), and how the team as a whole put the ideas together.

- The report should be written like a typical conference paper. Do not simply copy/paste your code or results from your terminals.  Try to organize it and present it in a neat and coherent format in terms of tables and figures. 