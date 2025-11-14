# NBA investment Tech Case - MP DATA x Nathan Naudé

## Description
Repository of candidate Nathan Naudé for tech case, part of the recruitment processus of Tilt Energy 2025.  

The goal is analyse and predict the electricity consumption in France (cf subject file "Tech Case 2025.pdf")

---

## Installation

Start by running `make --version` and `python --version` to make sure you have all the prerequisites.  

Install the right virtual environment:  
- Run `make setup`  
- Activate your virtual environment: `make show-activate` (then follow instructions)  

---

## Arborescence

To reproduce code, we advice you to follow this data arborescence (this empty arborescence is pushed):  

```                    
├── data  
│   └── nbalogreg.csv
│
├── src/                      
│   ├── ml
│   │    └── nba_predict.py
│   └── app
│        └── main.py
│
├── notebooks/
│   ├── exploration.ipynb                   
│
├── results/                  
│
├── models/                   
│
└── # rest of the repo
```

---

## CLI & API Usage

All commands are launched from the root directory.  


Train and test directly with CLI. The data must be stored in `data` directory. It performs training (with best parameters from the grid search) and test. Store the results in `results` directory and model artefact in `models.

```bash
python ./src/ml/nba_predict.py   
```

For the API usage, launch this CLI.
```bash
uvicorn src.app.main:APP --reload
```
Then go to your navigator and go to to `http://127.0.0.1:8000/docs`. Clic the `POST/predict` banner. Click `Try it out`. Fill the value of the JSON as you please then click the `execute` button to launch the prediction. You can read the result in the `Response body`.

---

## Dev summary

- data cleaning : Correct percentages
- feature engineering : add per minutes features and ratio AST to TOV and (AST+PTS) to TOV
- classification problem with optimisation (during training/valiation) with respect to positive recall.
- API developed with Fast API

---

## Code quality

Formating
```bash
make format
```

Lint check
```bash
make lint-check
```