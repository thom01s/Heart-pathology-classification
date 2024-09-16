#%%**TCC classificação de patologias cardíacas**
#%%importando bibliotecas e lendo o sinal

import pandas as pd
import wfdb
import numpy as np
import ast
import seaborn as sns
from matplotlib import pyplot as plt
from tensorflow.keras.layers import Input, Conv1D, BatchNormalization, ReLU, GlobalAveragePooling1D, Flatten, Dense, Dropout
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

#lendo os dados de anotação da pasta e o dataset em CSV
path = 'PTBXL/'

Y = pd.read_csv(path+'ptbxl_database.csv', index_col='ecg_id')
Y.scp_codes = Y.scp_codes.apply(lambda x: ast.literal_eval(x))

#carrega scp_statements.csv para agregação do diagnostico
agg_df = pd.read_csv(path+'scp_statements.csv', index_col=0)
#retirar dados com múltiplos labels
agg_df = agg_df[agg_df.diagnostic == 1]

def aggregate_diagnostic(y_dic):
    tmp = []
    for key in y_dic.keys():
        if key in agg_df.index:
            tmp.append(agg_df.loc[key].diagnostic_class)
    return list(set(tmp))

#aplicando as superclasses
Y['diagnostic_superclass'] = Y.scp_codes.apply(aggregate_diagnostic)

##retirar dados com múltiplos labels da superclasse
Y = Y[Y['diagnostic_superclass'].apply(len) == 1]
Y['diagnostic_superclass'] = [item for sublist in Y.diagnostic_superclass for item in sublist]

#converte os labels para numeros
label_to_index = {label: index for index, label in enumerate(np.unique(Y.diagnostic_superclass))}
Y['diagnostic_superclass'] = [label_to_index[label] for label in Y.diagnostic_superclass]

#carregando os sinais 100Hz
def load_raw_data(df, sampling_rate, path):
    data = [wfdb.rdsamp(path+f) for f in df.filename_lr]
    data = np.array([signal for signal, meta in data])
    return data

X = load_raw_data(Y, 100, path)

#%%normalização
#scaler = MinMaxScaler()
#X = scaler.fit_transform(X)

#%%valanceamento de classes
num_classes = len(np.unique(Y.diagnostic_superclass))

class_weights = compute_class_weight('balanced', classes=np.unique(Y.diagnostic_superclass), y=Y.diagnostic_superclass)
class_weights_dict = dict(enumerate(class_weights))

#%%separando os datasets de treino e teste e separando os labels de cada sinal
"""
val_fold = 9
test_fold = 10

# Train
X_train = X[(Y.strat_fold != test_fold) & (Y.strat_fold != val_fold)]
y_train = Y[(Y.strat_fold != test_fold) & (Y.strat_fold != val_fold)].diagnostic_superclass
# Validation
X_val = X[Y.strat_fold == val_fold]
y_val = Y[Y.strat_fold == val_fold].diagnostic_superclass
# Test
X_test = X[Y.strat_fold == test_fold]
y_test = Y[Y.strat_fold == test_fold].diagnostic_superclass

"""
X_train_lead = []
X_test_lead = []

for i in range(12):
    leads = X[:, :, i].reshape(-1, 1000, 1)
    X_train, X_test, y_train, y_test = train_test_split(leads, Y.diagnostic_superclass, test_size=0.15, random_state=42, stratify=Y.diagnostic_superclass)
    
    X_train_lead.append(X_train)
    X_test_lead.append(X_test)
    
#converte pra array
y_train = np.array(y_train)
y_test = np.array(y_test)
X_train_lead = np.array(X_train_lead)
X_test_lead = np.array(X_test_lead)

#%%Modelo de classificação CNN-1D

def make_model(input_shape):
    input_layer = Input(input_shape)

    conv = Conv1D(filters=8, kernel_size=3, padding="same")(input_layer)
    conv = BatchNormalization()(conv)
    conv = ReLU()(conv)
    
    conv1 = Conv1D(filters=16, kernel_size=3, padding="same")(conv)
    conv1 = BatchNormalization()(conv1)
    conv1 = ReLU()(conv1)
        
    conv2 = Conv1D(filters=32, kernel_size=3, padding="same")(conv1)
    conv2 = BatchNormalization()(conv2)
    conv2 = ReLU()(conv2)
    
    conv3 = Conv1D(filters=64, kernel_size=3, padding="same")(conv2)
    conv3 = BatchNormalization()(conv3)
    conv3 = ReLU()(conv3)
    
    conv4 = Conv1D(filters=128, kernel_size=3, padding="same")(conv3)
    conv4 = BatchNormalization()(conv4)
    conv4 = ReLU()(conv4)

    gap = GlobalAveragePooling1D()(conv4)

    flatten = Flatten()(gap)
    
    out1 = Dense(256, activation="relu")(flatten)
    out2 = Dense(128, activation="relu")(out1)
    out3 = Dense(64, activation="relu")(out2)
    out4 = Dense(32, activation="relu")(out3)
    out5 = Dense(16, activation="relu")(out4)
    dropout = Dropout(0.1)(out5)

    output_layer = Dense(num_classes, activation="softmax")(dropout)

    return Model(inputs=input_layer, outputs=output_layer)


#%%Compilar o modelo

# Lista para armazenar os modelos treinados
lead_models = []

for i in range (12):
    print(f"Treinando o modelo para o lead {i + 1}")
    
    model = make_model(input_shape=(1000, 1))

    model.compile(optimizer= Adam(learning_rate=0.001),
                  loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])
    
    # Definir callbacks
    model_callbacks = [
        ModelCheckpoint(f"best_model_ensenble_lead_{i+1}.keras", save_best_only=True, monitor="val_loss"),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=5),
        EarlyStopping(monitor="val_loss", patience=20, verbose=1)
    ]
    
    # Treinar o modelo 
    history = model.fit(X_train_lead[i],
                        y_train,
                        batch_size=32,
                        epochs=200,
                        callbacks=model_callbacks,
                        class_weight=class_weights_dict,
                        validation_split=0.2,
                        verbose=1)
    
    lead_models.append(model)

#%%plotando resultados
from sklearn.metrics import accuracy_score
from scipy.stats import mode

accuracies = []
for i, model in enumerate(lead_models):
    # Obter as predições do modelo
    y_pred = np.argmax(model.predict(X_test_lead[i]), axis=1)
    
    # Calcular a acurácia do modelo
    acc = accuracy_score(y_test, y_pred)
    accuracies.append((i, acc))  # Armazena o índice do modelo e sua acurácia
    
top_models = sorted(accuracies, key=lambda x: x[1], reverse=True)[:9]
top_indices = [idx for idx, acc in top_models]



predictions = [np.argmax(lead_models[i].predict(X_test_lead[i]), axis=1) for i in top_indices]

# Converte a lista de previsões para um array de forma (n_models, n_samples)
predictions = np.array(predictions)

# Realiza a votação majoritária ao longo do eixo 0 (modelos) para cada amostra
# Mode retorna a moda (valor mais comum) ao longo do eixo especificado
ensemble_predictions, _ = mode(predictions, axis=0)

# Converte o resultado para um vetor
ensemble_predictions = ensemble_predictions.flatten()

plt.figure(figsize=(13, 4))

plt.subplot(131)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper right')

plt.subplot(132)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper right')


plt.subplot(133)
y_pred = model.predict(X_test)
y_pred = np.argmax(y_pred, axis=1)
confusion_mtx = metrics.confusion_matrix(y_test, ensemble_predictions)

sns.heatmap(confusion_mtx, annot=True, fmt="d", linewidths=0.5, square=True, cmap="inferno")
plt.ylabel("Actual label")
plt.xlabel("Predicted label")
plt.title("Confusion Matrix test")
plt.show()

print(metrics.classification_report(y_test, ensemble_predictions))