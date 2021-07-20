# crime-cargo-detection
crime-cargo-detection

### How to Reproduce
#### Check the generated log
- `training_data/07-16_21-16-00/log.txt` you could check the learning process.
#### Check the devices
- CPU: i7-8700k
- GPU: GTX-1060 6GB, CUDA 11.4
- RAM: 32GB
- (If you don't satisfiy this device setting, you could get different results.)
#### Installation(Recomment conda)
- Python 3.8 (Recommend to install when you create conda env `conda create -n [env name] python=3.8`) -> you could get another version of python
- Pytorch(1.9.0) htts://pytorch.org/
```shell script 
    pip3 install torch==1.9.0+cu111 torchvision==0.10.0+cu111 torchaudio===0.9.0 -f https://download.pytorch.org/whl/torch_stable.html
```
or
```shell script 
    conda install pytorch torchvision torchaudio cudatoolkit=11.1 -c pytorch -c conda-forge
```
- Numpy(1.20.2, we install this version by conda, but it's version does not exist anymore)
```shell script
    pip install numpy==1.20.2
```
- Pandas(1.2.5)
```shell script
    pip install pandas==1.2.5
```
- Xgboost
```shell script
    pip install xgboost
```
- scikit-learn
```shell script
    pip install scikit-learn==0.24.2
```
- tensorboard
```shell script
    pip install tensorboard==2.5.0
```
- Seaboran
```shell script
    pip install seaborn 0.11.1
```
- Warning: If you don't satisfy all the condition we have, you couldn't get the same result
#### Find the seed fixed
- `Utils\seed.py` that are referenced from `main.py`
### How to use
- To reproduce the results from our weights of model to csv
``` shell script
    python main.py record --file_name 07-16_21-16-00
```
- To reproduce the learning weight(If you didn't set the exact same environment, then you could get different result!!!)
- `Warning: if you use '--preprocess True' option, generated dataset(we provided) would be changed.(Valid dataset generation would affect to the calcuation of train dataset preprocess.`
``` shell script
    python main.py train_mixed --preprocess True --epoch 50 -sr 0.80 --custom_loss fbeta_loss --seed 11 --lr 0.001 --beta 15 --lambda 0.3
```

`python main.py train_mixed --preprocess True --epoch 50 -sr 0.80 --custom_loss fbeta_loss --seed 11 --lr 0.001 --beta 15 --lambda 0.3`</br>
명령어를 통해 학습을 진행합니다. 학습이 마무리된 모델 파라미터와 전처리에 사용된 json 파일이 training_data 폴더 내에 학습 당시의 시간을 나타내는 [월-일_시-분-초]의 이름의 폴더 및 json 파일로 생성됩니다. </br>

`python main.py record --file_name "폴더이름"`
학습이 완료된 모델 파라미터를 가져와 test.csv를 추론합니다.</br>
"폴더이름"은 학습이 완료된 모델 파라미터가 포함되어 있는 [월-일_시-분-초] 입니다.
예) 학습 완료 후에 training_data 폴더에  07-16_21-16-00 폴더가 생성된 경우 :
`python main.py record --file_name 07-16_21-16-00`

최종 제출한 모델 파라미터는 07-16_21-16-00 폴더에 있으며, 이는 training_data 폴더에 저장되어 있습니다.</br>

Reproducing을 위한 Library version은 Markdown에 명기되어있으며, seed 고정을 위한 함수는 Utils/seed.py입니다.</br>

- Train binary classification whether cargo is crime or not
``` shell script
    python main.py train_crime
```
- Train binary classification whether cargo have more priority or not
``` shell script
    python main.py train_priority
```
- Train binary classification with the custom_loss (kd: [temperature, alpha], fbeta: [lambda, beta] options)
``` shell script
    python main.py [train_crime,train_priority,train_mixed] --custom_loss [kd_loss,fbeta_loss]
```

- Train binary classification whether cargo is crime or not
``` shell script
    python main.py train_xgboost_crime
```

- In Colab(Google) save the file in`drive/MyDrive/data/custom_contest/*.csv`,
``` shell script
    python main.py [mode] --colab True
```

- For data recording at test.csv,
``` shell script
    python main.py record --file_name [name in 'training_data' directory]
```
