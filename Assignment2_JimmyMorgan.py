import numpy as np
import pandas as pd
import skimage.io as io
import skimage.transform as form
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn import preprocessing
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

def plot_image(img, title):
    # Plot image
    fig, ax1 = plt.subplots(ncols=1, figsize=(18, 6), sharex=True,
                                   sharey=True)
    ax1.imshow(img, cmap='gray')
    ax1.set_title(title)
    ax1.axis('off')


litter = io.imread_collection('./KNN/animals/dogs/*.jpg')
formated_images = []
labels = []

print("Getting Puppers")
# resize and reshape images
for pup in litter:
    resized = form.resize(pup, (32, 32), anti_aliasing=True)
    formated_images.append( resized.reshape(3072) )
    labels.append('dog')


# Repeat for other animals
litter = io.imread_collection('./KNN/animals/cats/*.jpg')

print("Getting Kittens")
for kitten in litter:
    resized = form.resize(kitten, (32, 32), anti_aliasing=True)
    formated_images.append( resized.reshape(3072) )
    labels.append('cat')

len(labels)
len(formated_images)

litter = io.imread_collection('./KNN/animals/panda/*.jpg')

print("Getting Pandas")
for panda in litter:
    resized = form.resize(panda, (32, 32), anti_aliasing=True)
    # Some Panda pictures are black and white with only 1 layer so they don't match the rest of the data.
    if(resized.size == 3072):
        formated_images.append( resized.reshape(3072) )
        labels.append('panda')

# Encode labels and convert list to numpy array.
le = preprocessing.LabelEncoder()
animal_label= le.fit_transform(labels)
image_array = np.array(formated_images)

X_train, X_test, y_train, y_test = train_test_split(image_array, animal_label, test_size=0.3, random_state=21)
X_test, X_validate, y_test, y_validate = train_test_split(X_test, y_test, test_size = (1/3), random_state = 21)

# hold scores for value of k neighbors 
#   (note these lists are parallel)
# p_val 1= Manhattan, 2= Euclidean
p_val = 1
neighbors = list(range(1,30))
cv_train_scores = []
cv_validate_scores = []

for k in neighbors:
    knn = KNeighborsClassifier(n_neighbors=k, p=p_val)
    print("Trying ", k," Validation")
    
    # fitting the model
    knn.fit(X_train, y_train)
    
    # get predictions on train
    y_train_pred = knn.predict(X_train)
    cv_train_scores.append(accuracy_score(y_train, 
                                          y_train_pred))
    
    # get predictions on validation
    y_validate_pred = knn.predict(X_validate)
    cv_validate_scores.append(accuracy_score(y_validate, 
                                         y_validate_pred))

# Get Optimal K
test_min = cv_validate_scores.index(max(cv_validate_scores))
opt_k = neighbors[test_min]
print("Optimal K: ",opt_k)

# Display Eval
axes = plt.gca()
axes.set_xlim([0,len(neighbors)])
axes.set_ylim([min(cv_validate_scores)-.2,1.1])

red_patch = mpatches.Patch(color='orange', label='Train')
blue_patch = mpatches.Patch(color='blue', label='Test')
plt.legend(handles=[red_patch, blue_patch])

title="Accuracy CrossVal, optimal k: {}".format(opt_k)

axes.plot(neighbors,cv_validate_scores, label='test scores')
axes.plot(neighbors,cv_train_scores, label='train scores')
plt.title(title)

# Create Model
model = KNeighborsClassifier(n_neighbors=opt_k, p=p_val)

# Train the model using the training sets
model.fit(X_train,y_train)

# predict the response
y_test_pred = knn.predict(X_test)


print('Accuracy: {}'.format(round(accuracy_score(y_test, 
                                                 y_test_pred),3)))