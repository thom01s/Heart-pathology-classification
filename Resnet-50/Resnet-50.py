# **TCC classificação de patologias cardíacas**

"""importando bibliotecas"""

import pandas as pd
import numpy as np
import ast
from matplotlib import pyplot as plt
from tensorflow.keras.layers import BatchNormalization, GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping#, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from sklearn.model_selection import train_test_split
from sklearn import metrics
import seaborn as sns
#from imblearn.under_sampling import RandomUnderSampler

"""lendo o sinal da pasta e o dataset em CSV"""

lead_para_treino = 1

path = 'PTBXL/'

# load and convert annotation data
Y = pd.read_csv(path+'ptbxl_database.csv', index_col='ecg_id')
Y.scp_codes = Y.scp_codes.apply(lambda x: ast.literal_eval(x))

Y.replace(np.nan, 0, inplace=True)

"""Limpar dados com mais de uma classificação e converter os labels de string para número"""

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

Y['diagnostic_superclass'] = [item for sublist in Y.diagnostic_superclass for item in sublist]

# Convert the string labels into integers
label_to_index = {label: index for index, label in enumerate(np.unique(Y.diagnostic_superclass))}
Y['diagnostic_superclass'] = [label_to_index[label] for label in Y.diagnostic_superclass]

#%%
"""criando o dataframe das imagens do atrator"""

lead= []

for i in range(0, len(Y)):
    lead.append('PTBXL/atratores/' + str(i) + ' ' + lead_para_treino +'.png')

Y['lead'] = lead

"""pré-processamento das imagens para resnet"""

def load_and_preprocess_image(image_path):
    img = load_img(image_path, target_size=(224, 224, 3))  # ResNet-50 expects 224x224 images
    img_array = img_to_array(img).astype('float32')
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array = preprocess_input(img_array)  # Preprocess image
    return img_array

X = Y['lead'].apply(load_and_preprocess_image)

X = np.vstack(X.values).astype('float32')
y = np.array(Y['diagnostic_superclass'].values).astype('int32')

#%%
"""fazendo o undersampling da maior classe para imagens""" #ruim

#n_amostras, h, w, cn = X.shape
#X_reshaped = X.reshape((n_amostras, h * w * cn))

# Crie uma instância do RandomUnderSampler
#rus = RandomUnderSampler(sampling_strategy={0: 1708, 1: 535, 2: 2532, 3: 2500, 4: 2400}, random_state=42)

# Aplique o undersampling
#X, y = rus.fit_resample(X_reshaped, y)

#X = X.reshape((X.shape[0], h, w, cn))

#%%
"""equilibrio de classes por peso"""

qtd_amostra0 = 0
qtd_amostra1 = 0
qtd_amostra2 = 0
qtd_amostra3 = 0
qtd_amostra4 = 0

for i in range(0, len(y)):
    if y[i] == 0:
        qtd_amostra0 = qtd_amostra0 + 1
    elif y[i] == 1:
        qtd_amostra1 = qtd_amostra1 + 1
    elif y[i] == 2:
        qtd_amostra2 = qtd_amostra2 + 1
    elif y[i] == 3:
        qtd_amostra3 = qtd_amostra3 + 1
    elif y[i] == 4:
        qtd_amostra4 = qtd_amostra4 + 1
        
total = qtd_amostra0 + qtd_amostra1 + qtd_amostra2 + qtd_amostra3 + qtd_amostra4

weight_for_0 = (1 / qtd_amostra0) * (total / 5.0)
weight_for_1 = (1 / qtd_amostra1) * (total / 5.0)
weight_for_2 = (1 / qtd_amostra2) * (total / 5.0)
weight_for_3 = (1 / qtd_amostra3) * (total / 5.0)
weight_for_4 = (1 / qtd_amostra4) * (total / 5.0)

class_weight = {0: weight_for_0, 1: weight_for_1 ,2: weight_for_2, 3: weight_for_3, 4: weight_for_4}

#%%
"""separando os datasets de treino e teste e separando os labels de cada sinal"""

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, stratify=y, random_state=1)

X_validation = X_test[-5:]
y_validation = y_test[-5:]

X_test = X_test[:-5]
y_test = y_test[:-5]

num_classes = len(np.unique(y_train))

#%%
"""Modelo Resnet-50"""

# Carregar o modelo ResNet-50 pré-treinado, sem incluir a primeira camada
base_model = ResNet50(include_top=False, weights='imagenet', input_tensor=None, input_shape=(224, 224, 3))
    
train_layer = base_model.layers[-1]
train_index = base_model.layers.index(train_layer)

# Percorre as camadas a seguir e aplica a operação no novo input
for layer in base_model.layers[1:train_index+1]:
    # não desliga treinamento para BatchNorm
    if isinstance(layer, BatchNormalization): 
        layer.trainable=True
             
gap = GlobalAveragePooling2D()(layer.output)
    
dropout = Dropout(0.1)(gap)

dense = Dense(256, activation='relu')(dropout)
dense2 = Dense(128, activation='relu')(dense)
dense3 = Dense(64, activation='relu')(dense2)
dense4 = Dense(32, activation='relu')(dense3)
dense5 = Dense(16, activation='relu')(dense4)
dense6 = Dense(8, activation='relu')(dense5)
    
output_layer = Dense(num_classes, activation='softmax')(dense6)

model = Model(inputs=base_model.input, outputs=output_layer)

#%%

# Compilar o modelo
model.compile(optimizer= Adam(learning_rate=0.001),
              loss="sparse_categorical_crossentropy",
              metrics=["accuracy"])

# Definir callbacks
model_callbacks = [
    ModelCheckpoint('best_model_resnet_lead' + lead_para_treino + '.keras', save_best_only=True, monitor="val_loss"),
    #ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=5, min_lr=0.0001),
    EarlyStopping(monitor="val_loss", patience=5, verbose=1)
]

# Treinar o modelo
history = model.fit(X_train,
                    y_train,
                    batch_size=32,
                    epochs=50,
                    callbacks=model_callbacks,
                    validation_data=(X_test, y_test),
                    verbose=1)
 
# Carregar o melhor modelo salvo
model = load_model('best_model_resnet_lead' + lead_para_treino + '.keras')

# Avaliar o modelo no conjunto de teste
test_loss, test_acc = model.evaluate(X_validation, y_validation)

print("Test accuracy", test_acc)
print("Test loss", test_loss)

#%%
"""plotando resultados"""

plt.figure(figsize=(13, 4))

plt.subplot(131)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper right')

plt.subplot(132)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper right')


plt.subplot(133)
y_pred = model.predict(X_test)
y_pred = np.argmax(y_pred, axis=1)
confusion_mtx = metrics.confusion_matrix(y_test, y_pred)

sns.heatmap(confusion_mtx, annot=True, fmt="d", linewidths=0.5, square=True, cmap="inferno")
plt.ylabel("Actual label")
plt.xlabel("Predicted label")
plt.title("Confusion Matrix")
plt.show()

print(metrics.classification_report(y_test, y_pred))