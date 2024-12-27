# Create and access environment
To create an environment, please use thia code. A conda environment named "group_3_neuro_env" would be created:
```
conda env create -f environment.yml
```

After finishing creating the conda environment, run this code to access the environment
```
conda activate group_3_neuro_env
```

# Extract feature(s)
You could skip this part and move to the next part as the result of this stage (8 csv files) are saved in the ```./data/csv``` folder.
 
In the ```./data``` folder there is already train and test indices files, namely ```train.csv``` and ```test.csv```. We split train and test using script inside ```split_data.ipynb```.

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
