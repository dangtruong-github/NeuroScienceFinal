# Create environment
To create an environment, please use thia code. A conda environment named "group_3_neuro_env" would be created:
```
conda env create -f environment.yml
```

# Extract feature(s)
In the ```./data``` folder there is already train and test indices files, namely ```train.csv``` and ```test.csv```.

To extract the features, first, put all the data samples in the ```./data/raw``` folder. Then, run this code:
```
python extract_final.py
```

This would create 8 csv files corresponding to 8 different types of feature extraction (fill bad segment/no fill, fft/no fft - time domain, channel-wise/total)

# Model evaluation
After creating 8 csv files of features, run this code to get the best MAE of all models:
```
python main.py
```

The result can be seen in the file ```./logging/out_model.txt```. In fact, all the code outputs could be seen inside ```./logging``` folder.
