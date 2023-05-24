# tinyNG
This is a language model using LSTM trained for Name Generation. It was trained on a male name dataset. Built with Pytorch.

## Installation
To install requirements, simply:
```
pip install -r requirements.txt
```

## Usage
For argument helpers, use:
```
python main.py -h
```

## Examples
For name generation with a given starting letter "A", to generate 100 names, use mode 2:
```
python main.py -c config/config.json -o cache/output.txt -w cache/model_pretrained.pth -p A -l 100 -m 2
```
For random name generation, use mode 3:
```
python main.py -c config/config.json -o cache/output.txt -w cache/model_pretrained.pth -p A -l 100 -m 3
```

Some of the names generated by the model:
```
Aughaud
Rathan
Rens
Cytory
Jarrin
Oliley
Rahoe
Tengus-Picesyck
Kelf
Zayste
Bavimais
```
