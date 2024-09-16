"""importando bibliotecas"""

import pandas as pd
import wfdb
import numpy as np
import ast
from matplotlib import pyplot as plt

"""lendo o sinal da pasta e o dataset em CSV"""

#carregando somente os arquivos com 100hz
sampling_rate = 100
def load_raw_data(df, sampling_rate, path):
    if sampling_rate == 100:
        data = [wfdb.rdsamp(path+f) for f in df.filename_lr]
    else:
        data = [wfdb.rdsamp(path+f) for f in df.filename_hr]
    data = np.array([signal for signal, meta in data])
    return data

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

Y.loc[:,'diagnostic_superclass'] = [item for sublist in Y.diagnostic_superclass for item in sublist]

# Convert the string labels into integers
label_to_index = {label: index for index, label in enumerate(np.unique(Y.diagnostic_superclass))}
Y.loc[:,'diagnostic_superclass'] = [label_to_index[label] for label in Y.diagnostic_superclass]

"""lendo os sinais do dataframe"""

# Load raw signal data
X = load_raw_data(Y, sampling_rate, path)

#normalizar dados do dataset
X = X.reshape((X.shape[0], X.shape[1], 12))

# Convert y_train to a NumPy array
y = np.array(Y.diagnostic_superclass)

# Convert X_train to a NumPy array
X = np.array(X)

X = np.transpose(X, (0, 2, 1))

#%%
"""construindo os atratores de cada lead"""

from scipy.signal import find_peaks
def plot_and_save(v_t, w_t, patient, lead):
    plt.figure()
    plt.hist2d(v_t.flatten(), w_t.flatten(), bins=(100, 100), cmap=plt.cm.jet)
    #plt.plot(v_t.flatten(), w_t.flatten())
    plt.axis('off')
    c = plt.colorbar()
    plt.clim(10, 0)
    c.remove()

    plt.savefig('C:\\Users\\Usuario\\Desktop\\UNISINOS\\6° semestre\\VISAO COMPUTACIONAL\\proc imagens\\ECG\\PTBXL\\atratores\\'+ str(patient) + ' ' + str(lead) + '.png' , bbox_inches='tight')
    plt.close()
    
patient = 0
for exame in range(len(X)):
    lead = 1
    for sinal in X[exame]:
        inner = []
        integral = []
      
        peaks = find_peaks(sinal, distance=70)[0]
        
        #plt.plot(sinal)
        #plt.plot(peaks, sinal[peaks], "x")
            
        #faz a média de distância entre os picos para calcular um tau médio
        soma = 0
        tau = 0
        for j in range(1, len(peaks)):
            soma = soma + (peaks[j] - peaks[j-1])
                    
            tau = int((soma/(len(peaks)-1))/3)
                
            x_t = sinal[2*tau:]
            y_t = sinal[tau:]
            z_t = sinal
                
            max_x = np.array(find_peaks(x_t)[0])
            max_y = np.array(find_peaks(y_t)[0])
            max_z = np.array(find_peaks(z_t)[0])
                
            num_pontos = len(x_t)
                
                
            v_t = (x_t[:num_pontos] + y_t[:num_pontos] - 2*z_t[:num_pontos])/(np.sqrt(6))
            w_t = (x_t[:num_pontos] - y_t[:num_pontos])/(np.sqrt(2))
                
            inner.append(np.inner(v_t, w_t))
            integral.append(sum(w_t))
        
        plot_and_save(v_t, w_t, patient, lead)
        
        lead+=1
    patient+=1