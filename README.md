# Digit_Recognizer_CNN_Model
Identify digits from a dataset of tens of thousands of handwritten images with CNN

# Importing Libraries and Dataset


```python
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout, BatchNormalization
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
import matplotlib.pyplot as plt
```


```python
df_train = pd.read_csv(r"C:\Users\ACER\Downloads\digit-recognizer\train.csv")
df_train
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>label</th>
      <th>pixel0</th>
      <th>pixel1</th>
      <th>pixel2</th>
      <th>pixel3</th>
      <th>pixel4</th>
      <th>pixel5</th>
      <th>pixel6</th>
      <th>pixel7</th>
      <th>pixel8</th>
      <th>...</th>
      <th>pixel774</th>
      <th>pixel775</th>
      <th>pixel776</th>
      <th>pixel777</th>
      <th>pixel778</th>
      <th>pixel779</th>
      <th>pixel780</th>
      <th>pixel781</th>
      <th>pixel782</th>
      <th>pixel783</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>41995</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>41996</th>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>41997</th>
      <td>7</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>41998</th>
      <td>6</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>41999</th>
      <td>9</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>42000 rows × 785 columns</p>
</div>



# Dataset Splitting and Normalization


```python
X = df_train.drop(columns='label').values / 255.0  # Normalizing the data
y = df_train['label']
```


```python
# Reshape the data to fit the model (28x28x1 for grayscale images)
X = X.reshape(-1, 28, 28, 1)

# One-hot encode the labels
y = to_categorical(y, num_classes=10)
```


```python
# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
```

# Modelling


```python
# Data Augmentation
datagen = ImageDataGenerator(
    rotation_range=10,
    zoom_range=0.1,
    width_shift_range=0.1,
    height_shift_range=0.1
)

# Fit the data generator on the training data
datagen.fit(X_train)
```


```python
# Build the CNN model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    BatchNormalization(),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D((2, 2)),
    Conv2D(128, (3, 3), activation='relu'),
    BatchNormalization(),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(10, activation='softmax')
])
```

    D:\anaconda3\Lib\site-packages\keras\src\layers\convolutional\base_conv.py:107: UserWarning: Do not pass an `input_shape`/`input_dim` argument to a layer. When using Sequential models, prefer using an `Input(shape)` object as the first layer in the model instead.
      super().__init__(activity_regularizer=activity_regularizer, **kwargs)
    


```python
# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model using the augmented data
model.fit(datagen.flow(X_train, y_train, batch_size=32), epochs=10, validation_data=(X_test, y_test))

```

    Epoch 1/10
    

    D:\anaconda3\Lib\site-packages\keras\src\trainers\data_adapters\py_dataset_adapter.py:121: UserWarning: Your `PyDataset` class should call `super().__init__(**kwargs)` in its constructor. `**kwargs` can include `workers`, `use_multiprocessing`, `max_queue_size`. Do not pass these arguments to `fit()`, as they will be ignored.
      self._warn_if_super_not_called()
    

    [1m1050/1050[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m33s[0m 26ms/step - accuracy: 0.7903 - loss: 0.6674 - val_accuracy: 0.9811 - val_loss: 0.0721
    Epoch 2/10
    [1m1050/1050[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m25s[0m 24ms/step - accuracy: 0.9501 - loss: 0.1704 - val_accuracy: 0.9740 - val_loss: 0.0820
    Epoch 3/10
    [1m1050/1050[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m26s[0m 24ms/step - accuracy: 0.9663 - loss: 0.1202 - val_accuracy: 0.9756 - val_loss: 0.0953
    Epoch 4/10
    [1m1050/1050[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m25s[0m 24ms/step - accuracy: 0.9681 - loss: 0.1129 - val_accuracy: 0.9756 - val_loss: 0.0793
    Epoch 5/10
    [1m1050/1050[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m25s[0m 24ms/step - accuracy: 0.9740 - loss: 0.0928 - val_accuracy: 0.9911 - val_loss: 0.0389
    Epoch 6/10
    [1m1050/1050[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m25s[0m 24ms/step - accuracy: 0.9741 - loss: 0.0889 - val_accuracy: 0.9868 - val_loss: 0.0480
    Epoch 7/10
    [1m1050/1050[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m25s[0m 24ms/step - accuracy: 0.9781 - loss: 0.0776 - val_accuracy: 0.9905 - val_loss: 0.0350
    Epoch 8/10
    [1m1050/1050[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m25s[0m 24ms/step - accuracy: 0.9781 - loss: 0.0810 - val_accuracy: 0.9892 - val_loss: 0.0390
    Epoch 9/10
    [1m1050/1050[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m25s[0m 24ms/step - accuracy: 0.9781 - loss: 0.0785 - val_accuracy: 0.9910 - val_loss: 0.0377
    Epoch 10/10
    [1m1050/1050[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m25s[0m 24ms/step - accuracy: 0.9823 - loss: 0.0621 - val_accuracy: 0.9885 - val_loss: 0.0459
    




    <keras.src.callbacks.history.History at 0x262dfff6e10>



# Evaluation


```python
# Evaluate the model
train_loss, train_accuracy = model.evaluate(X_train, y_train)
test_loss, test_accuracy = model.evaluate(X_test, y_test)

print("Menggunakan Model CNN dengan Augmentasi Data dan Batch Normalization:")
print(f"Akurasi pada data training: {train_accuracy:.4f}")
print(f"Akurasi pada data testing: {test_accuracy:.4f}")
```

    [1m1050/1050[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m7s[0m 6ms/step - accuracy: 0.9930 - loss: 0.0259
    [1m263/263[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m2s[0m 6ms/step - accuracy: 0.9883 - loss: 0.0531
    Menggunakan Model CNN dengan Augmentasi Data dan Batch Normalization:
    Akurasi pada data training: 0.9919
    Akurasi pada data testing: 0.9885
    


```python
# Visualization: Displays predicted and actual results on testing data
idx = np.random.choice(len(X_test), size=36, replace=False)
images, labels = X_test[idx], y_test[idx]
preds = np.argmax(model.predict(images), axis=1)  # Prediksi dari model
true_labels = np.argmax(labels, axis=1)  # Label asli

fig, axes = plt.subplots(6, 6, figsize=(15, 15))
for img, true_label, pred, ax in zip(images, true_labels, preds, axes.flatten()):
    font = {'color': 'g'} if true_label == pred else {'color': 'r'}
    ax.imshow(img.reshape(28, 28), cmap='gray')
    ax.set_title(f"Label: {true_label} | Pred: {pred}", fontdict=font)
    ax.axis('off')
plt.show()
```

    [1m2/2[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 199ms/step
    


    
![png](output_13_1.png)
    


# Apply to New Dataset


```python
df_test = pd.read_csv(r"C:\Users\ACER\Downloads\digit-recognizer\test.csv")

X_new_test = df_test.values / 255.0
X_new_test = X_new_test.reshape(-1, 28, 28, 1)
```


```python
results = model.predict(X_new_test)
results = np.argmax(results, axis=1)
```

    [1m875/875[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m6s[0m 6ms/step
    


```python
# Create DataFrame for submission
df_results = pd.DataFrame({
    'ImageId': np.arange(1, len(results) + 1),
    'Label': results
})
```


```python
df_results
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>ImageId</th>
      <th>Label</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>2</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>9</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>3</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>27995</th>
      <td>27996</td>
      <td>9</td>
    </tr>
    <tr>
      <th>27996</th>
      <td>27997</td>
      <td>7</td>
    </tr>
    <tr>
      <th>27997</th>
      <td>27998</td>
      <td>3</td>
    </tr>
    <tr>
      <th>27998</th>
      <td>27999</td>
      <td>9</td>
    </tr>
    <tr>
      <th>27999</th>
      <td>28000</td>
      <td>2</td>
    </tr>
  </tbody>
</table>
<p>28000 rows × 2 columns</p>
</div>




```python
# Save the predictions to a CSV file
df_results.to_csv(r"C:\Users\ACER\Downloads\Digit Recognizer CNN Model III.csv", index=False)

print("Predictions saved to 'predictions.csv'.")
```

    Predictions saved to 'predictions.csv'.
    


```python

```

