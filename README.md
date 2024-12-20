## Contents 
- [Overview](#overview)
- [Data Preprocess Flow](#data-preprocess-flow)
- [Model Structure](#model-structure)
- [Use Models](#use-models)
- [Links](#links)

## Overview
Recent advancement of aviation industry has increased the research interest in the field of Air Traffic Management (ATM). 
A contemporary - open - problem in ATM field is the prediction of aircrafts trajectories hidden parameters. 
Such parameters prove to be really important for timescale and financial scheduling, as they are capable of defining flight’s Key Performance Indicators (KPIs), 
such as fuel needs, duration and distance to cover. 
This thesis aims on the prediction of Cost Index (CI) and Maximum Takeoff Weight (MTOW) hidden parameters, using simulated trajectories provided in a time serries format.

The problem is casted as a regression task and the methodology used is based on the integration of Convolutional Neural Networks and spatial graph theory, constructing the proposed GCNN. 
The problem formulation is projected to a graph based environment, where agents, represented as nodes, communicate and collaborate to provide the final outcome. 
Each one receives and processes a specific part of the flight. 
The communication between them is achieved, by using transformers and applying multiheads attention function as the convolutional kernel. 
Results show that GCNN adapts efficiently to the flight data and worthily competes the baseline models, provided by previous, similar research works

## Data Preprocess Flow

```mermaid
graph TD;

  DB[Flight Data]
  Ppr[Ppr Variables]
  Opr[Opr Variables]

  DB --> Ppr
  DB --> Opr

  Ppr --TrainTestSplit.py--> PprTrain[PprTrainSet]
  Ppr --TrainTestSplit.py--> PprTest[PprTestSet]
  Opr --TrainTestSplit.py--> OprTrain[OprTrainSet]
  Opr --TrainTestSplit.py--> OprTest[OprTestSet]

  PprTrain -- PprTrainToCSV.py --> PprTrain.csv[PprTrain.csv]
  PprTest -- PprTestToCSV.py --> PprTest.csv[PprTest.csv]
  OprTrain -- OprTrainToCSV.py --> OprTrain.csv[OprTrain.csv]
  OprTest -- OprTestToCSV.py --> OprTest.csv[OprTest.csv]

  PprTrain.csv -- DataCleanNorm.py --> PprTrainNorm.csv
  PprTest.csv -- DataCleanNorm.py --> PprTestNorm.csv
  OprTrain.csv -- DataCleanNorm.py --> OprTrainNorm.csv
  OprTest.csv -- DataCleanNorm.py --> OprTestNorm.csv

```

## Model Structure

![](modelStructure.png)

## Use Models

## Links

