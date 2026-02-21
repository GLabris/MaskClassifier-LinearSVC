import os
import numpy as np
from tqdm import tqdm
import cv2
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score
import joblib




#prepare data
input_dir = os.path.join('.','data','data')
categories = ['with_mask','without_mask']

#define arrays where data and labels will be stored
data = []
labels = []

#iteration through files to preprocess and load images
for category_idx, category in enumerate(tqdm(categories)):
    for file in tqdm(os.listdir(os.path.join(input_dir, category))):
        img_path = os.path.join(input_dir, category, file)
        img = cv2.imread(img_path)
        if img is None:
            continue
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (32, 32))
        data.append(img.flatten())
        labels.append(category_idx)

data = np.asarray(data)
labels = np.asarray(labels)



# train / test split


x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2,shuffle=True, stratify=labels,random_state=42)


#train classsifier

classifier = LinearSVC()

parameters = [{'C': [0.1, 1, 10, 100]}]

grid_search = GridSearchCV(classifier,parameters,verbose=3)

grid_search.fit(x_train,y_train)

#test performance

best_estimator = grid_search.best_estimator_

y_prediction = best_estimator.predict(x_test)

score = accuracy_score(y_prediction, y_test)

print('{}% of samples were correctly classified'.format(str(score*100)))


#save model
joblib.dump(best_estimator, "mask_detector_LinearSVC.pkl")
