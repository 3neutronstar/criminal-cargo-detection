# crime-cargo-detection
crime-cargo-detection

### How to use
- Train binary classification whether cargo is crime or not
``` shell script
    python run.py train_crime
```
- Train binary classification whether cargo have more priority or not
``` shell script
    python run.py train_priority
```
- Train binary classification whether cargo is crime or not
``` shell script
    python run.py [train_xgboost_crime,train_xgboost_priority]
```

- Data generation, saved in `data/custom_contest/*` (Colab is also available)
``` shell script
    python run.py gen_data
```

- In Colab(Google) `data/custom_contest/*`,
``` shell script
    python run.py [mode] --colab True
```

