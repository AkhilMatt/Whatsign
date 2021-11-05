# Standard
import os

# Third-party
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelBinarizer
import joblib

# Local

labels = []
paths = []
for root, dirs, files in os.walk('data\\training_images'):
    for file in files:
        label = os.path.join(root, file).split(os.sep)[-2]
        path = os.path.join(root, file)
        labels.append(label)
        paths.append(path)
data = pd.DataFrame({'Paths':paths, 'Labels':labels})

# Label binarize the labels
lb = LabelBinarizer()
labels = lb.fit_transform(labels)
print(f"The first label binarized labels: {labels[0]}")
print(f"Mapping the first label binarized label to its category: {lb.classes_[0]}")
print(f"Total instances: {len(labels)}")
for i in range(len(labels)):
    index = np.argmax(labels[i])
    data.loc[i, 'Labels'] = int(index)
# shuffling the dataset
data = data.sample(frac=1).reset_index(drop=True)
data.to_csv('data\\data.csv', index=False)
# pickle the binarized labels
print('Saving the binarized labels as a pickled file')
joblib.dump(lb, 'outputs\\lb.pkl')
