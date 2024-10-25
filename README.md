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
