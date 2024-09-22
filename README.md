```mermaid
graph TD;
  DB[gradguest_data] --> Ppr[PprVariables]
  DB[gradguest_data] --> Orgn[OrgnVariables]
  Ppr --> PprTrain[PprTrainSet]
  Ppr --> PprTest[PprTestSet]
  Orgn --> OrgnTrain[OrgnTrainSet]
  Orgn --> OrgnTest[OrgnTestSet]
  PprTrain -- FiftyOnePrepocess.py --> PprTrain.csv[PprTrain.csv]
  PprTest -- FiftyOnePrepocess.py --> PprTest.csv[PprTrain.csv]
  OrgnTrain -- FiftyOnePrepocess.py --> OrgnTrain.csv[OrgnTrain.csv]
  OrgnTest -- FiftyOnePrepocess.py --> OrgnTest.csv[OrgnTrain.csv]

