```mermaid
graph TD;

  DB[Flight Data]
  Ppr[Ppr Variables]
  Opr[Opr Variables]

  DB --> Ppr
  DB --> Opr

  Ppr --> PprTrain[PprTrainSet]
  Ppr --> PprTest[PprTestSet]
  Opr --> OprTrain[OprTrainSet]
  Opr --> OprTest[OprTestSet]

  PprTrain -- PprTrainToCSV.py --> PprTrain.csv[PprTrain.csv]
  PprTest -- PprTestToCSV.py --> PprTest.csv[PprTest.csv]
  OprTrain -- OprTrainToCSV.py --> OprTrain.csv[OprTrain.csv]
  OprTest -- OprTestToCSV.py --> OprTest.csv[OprTest.csv]
```
