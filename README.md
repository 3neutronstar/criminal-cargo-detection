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
- Train binary classification with the custom_loss (kd: [temperature, alpha], fbeta: [lambda, beta] options)
``` shell script
    python run.py [train_crime,train_priority,train_mixed] --custom_loss [kd_loss,fbeta_loss]
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

- For data recording at test.csv,
``` shell script
    python run.py record --file_name [name in 'training_data' directory]
```

