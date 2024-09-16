# **TCC classificação de patologias cardíacas**

"""importando bibliotecas"""

import pandas as pd
import wfdb
import numpy as np
import ast
from matplotlib import pyplot as plt
from tensorflow.keras.layers import Input, Conv1D, BatchNormalization, ReLU, GlobalAveragePooling1D, GlobalAveragePooling2D, Flatten, Dense, Dropout
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.callbacks import ModelCheckpoint#, ReduceLROnPlateau, EarlyStopping
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn import metrics
import seaborn as sns
from imblearn.under_sampling import RandomUnderSampler

# Load only archives with this sampling rate
sampling_rate = 100
def load_raw_data(df, sampling_rate, path):
    if sampling_rate == 100:
        data = [wfdb.rdsamp(path+f) for f in df.filename_lr]
    else:
        data = [wfdb.rdsamp(path+f) for f in df.filename_hr]
    data = np.array([signal for signal, meta in data])
    return data

path = 'PTBXL/'

# Load and convert annotation data
Y = pd.read_csv(path+'ptbxl_database.csv', index_col='ecg_id')
Y.scp_codes = Y.scp_codes.apply(lambda x: ast.literal_eval(x))

Y.replace(np.nan, 0, inplace=True)

# Load scp_statements.csv for diagnostic aggregation
agg_df = pd.read_csv(path+'scp_statements.csv', index_col=0)
agg_df = agg_df[agg_df.diagnostic == 1]

def aggregate_diagnostic(y_dic):
    tmp = []
    for key in y_dic.keys():
        if key in agg_df.index:
            tmp.append(agg_df.loc[key].diagnostic_class)
    return list(set(tmp))

# Apply diagnostic superclass
Y['diagnostic_superclass'] = Y.scp_codes.apply(aggregate_diagnostic)

# Filter DataFrame to remove rows with more than one label in 'classification' column
Y = Y[Y['diagnostic_superclass'].apply(len) == 1]

Y.diagnostic_superclass = [item for sublist in Y.diagnostic_superclass for item in sublist]

# Convert the string labels into integers
label_to_index = {label: index for index, label in enumerate(np.unique(Y.diagnostic_superclass))}
Y.loc[:,'diagnostic_superclass'] = [label_to_index[label] for label in Y.diagnostic_superclass]

"""### separando os datasets de treino e teste e separando os labels de cada sinal"""

# Load raw signal data
X = load_raw_data(Y, sampling_rate, path)

# Reshape the data to contain the 12 Lead in each signal
X = X.reshape((X.shape[0], X.shape[1], 12))

# Convert to a NumPy array
y = np.array(Y.diagnostic_superclass)
X = np.array(X)

# Transpose the matrix to organize the values
X = np.transpose(X, (0, 2, 1))

#%%
# Balance the data (reducing the samples in the umbalanced class) (made the average results worse)<-------
#n_amostras, ondas, pontos_por_onda = X.shape
#X_reshaped = X.reshape((n_amostras, pontos_por_onda * ondas))

# Create RandomUnderSampler instance
#rus = RandomUnderSampler(sampling_strategy={0: 1708, 1: 535, 2: 2532, 3: 2500, 4: 2400}, random_state=42)

# Applyundersampling
#X, y = rus.fit_resample(X_reshaped, y)

#X = X.reshape((X.shape[0], ondas, pontos_por_onda))

#%%
# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, stratify=y, random_state=1)

# Number of categories to classify
num_classes = len(np.unique(y_train))

#%%
# CNN model
def make_model(input_shape):
    input_layer = Input(input_shape)

    conv1 = Conv1D(filters=8, kernel_size=3, padding="same")(input_layer)
    conv1 = BatchNormalization()(conv1)
    conv1 = ReLU()(conv1)

    conv2 = Conv1D(filters=64, kernel_size=3, padding="same")(conv1)
    conv2 = BatchNormalization()(conv2)
    conv2 = ReLU()(conv2)

    conv3 = Conv1D(filters=64, kernel_size=3, padding="same")(conv2)
    conv3 = BatchNormalization()(conv3)
    conv3 = ReLU()(conv3)

    gap = GlobalAveragePooling1D()(conv3)

    flatten = Flatten()(gap)
    
    out1 = Dense(256, activation="relu")(flatten)
    out2 = Dense(128, activation="relu")(out1)
    out3 = Dense(64, activation="relu")(out2)
    out4 = Dense(32, activation="relu")(out3)
    out5 = Dense(16, activation="relu")(out4)
    
    dropout = Dropout(0.1)(out5)

    output_layer = Dense(num_classes, activation="softmax")(dropout)

    return Model(inputs=input_layer, outputs=output_layer)


model = make_model(input_shape=X_train.shape[1:])

# Callbacks

epochs = 50
batch_size = 32

callbacks = [
    ModelCheckpoint("best_model.keras", save_best_only=True, monitor="val_loss"),
#    ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=20, min_lr=0.001),
#    EarlyStopping(monitor="val_loss", patience=50, verbose=1),
]

# Compile the model
model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss="sparse_categorical_crossentropy",
    metrics=["sparse_categorical_accuracy"],
)

# Fit the model
 
history = model.fit(
    X_train,
    y_train,
    batch_size=batch_size,
    epochs=epochs,
    callbacks=callbacks,
    validation_split=0.2,
    verbose=1,
)

# Print the results of the best model

model = load_model("best_model.keras")

test_loss, test_acc = model.evaluate(X_test, y_test)

print("Test accuracy", test_acc)
print("Test loss", test_loss)

#%%
# Plot the loss graphic

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper right')
plt.show()

#%%
# Plot the accuracy graphic
plt.plot(history.history['sparse_categorical_accuracy'])
plt.plot(history.history['val_sparse_categorical_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper right')
plt.show()

#%%
# Plot the confusion matrix
y_pred = model.predict(X_test)
y_pred = np.argmax(y_pred, axis=1)
confusion_mtx = metrics.confusion_matrix(y_test, y_pred)
plt.figure(figsize=(10, 10))
sns.heatmap(confusion_mtx, annot=True, fmt="d", linewidths=0.5, square=True, cmap="inferno")
plt.ylabel("Actual label")
plt.xlabel("Predicted label")
plt.title("Confusion Matrix")
plt.show()

print(metrics.classification_report(y_test, y_pred))