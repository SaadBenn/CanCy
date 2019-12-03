# Cancer-Stats-Miner

The [dataset](http://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+%28Original%29) used for modelling comes from University of Wisconsin.


### Data Visualizing
---
in order to use visualization
run the script 

```python3.6 src/visualization.py --dataset_type type_of_dataset```

Replace type_of_dataset with either one of these options: 
- breast-cancer-wisconsin.data
- wdbc.data
- wpbc.data

The script will create a folder called visualization. The dataset folder would be created accordingly 
in the visualization folder. In order to run the visualization we can run this script in terminal:

```tensorboard --logdir=visualization/wdbc.data --host=localhost ```



# TODOS
- [X] Data labelling/cleaning

- [ ] Data Modelling

- [ ] Data Prediction
