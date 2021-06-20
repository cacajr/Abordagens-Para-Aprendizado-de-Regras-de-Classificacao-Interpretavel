from Solvers.RC2.bin.pysat.examples.rc2 import RC2
from Solvers.RC2.bin.pysat.formula import WCNF

wcnf = WCNF(from_file='./model.wcnf')

with RC2(wcnf) as rc2:
    for m in rc2.enumerate():
        print('model {0} has cost {1}'.format(m, rc2.cost))

'''
import pandas as pd
import csv
import os

from Algoritmos.IMLI.imli import imli
from Algoritmos.An_existing_SAT_model.an_existing_model import an_existing_model as IKKRR

def save_files_test_and_training(arq, frac_test, address):
    try:
        Xy = pd.read_csv(arq)

        Xy_test = Xy.sample(frac=frac_test)
        print(Xy_test)
        Xy_training = Xy.drop(Xy_test.index.tolist())
        print(Xy_test)
        Xy_test.to_csv(address+"/test.csv", index=False)
        Xy_training.to_csv(address+"/training.csv", index=False)
    except:
        print("Read file error!")

arq = r"D:\Área de Trabalho (D)\TCC\Datasets\transfusion.csv"
save_files_test_and_training(arq, 0.1, "./test_and_training_files")
'''
'''
def average(matrix):
    average_each_column = []

    for j in range(len(matrix[0])):
        average_each_column.append(0)

        for i in range(len(matrix)):
            average_each_column[j] += matrix[i][j]
        
        average_each_column[j] = average_each_column[j]/len(matrix)
    
    return average_each_column

averages = average([[1,2,3], [3,2,1], [7,8,2]])

print(averages)
'''
'''
# WRITING INFORMATIONS IN CSV
a = 'testing'
f = open('./Tests_informations/'+a+'_informations.csv', 'w', newline='', encoding='utf-8')
w = csv.writer(f)

w.writerow(['IMLI', '', '', '', ''])

f.close()

def get_test_data(X, y, frac, lines=[]):
    # convert to dataframe
    df_X = pd.DataFrame(X)
    df_y = pd.DataFrame(y, columns=['P'])

    # concat dataframes X + y
    df_Xy = pd.concat([df_X, df_y], axis=1)

    df_Xy_test = pd.DataFrame()
    df_Xy_training = pd.DataFrame()
    if(lines == []):
        # get % test
        df_Xy_test = df_Xy.sample(frac=frac)

        # create dataframe without samples (training)
        df_Xy_training = df_Xy.drop(df_Xy_test.index.tolist())

        print('Entrou!!!')
    else:
        # get % test
        df_Xy_test = df_Xy.loc[lines]

        # create dataframe without samples (training)
        df_Xy_training = df_Xy.drop(lines)
        print('Não entrou!!!')

    # desconcat dataframe X - y and convert to matrix and vector respectively
    # test
    X_test = df_Xy_test.drop('P', axis=1).as_matrix()
    y_test = df_Xy_test['P'].values.ravel()

    # training
    X_training = df_Xy_training.drop('P', axis=1).as_matrix()
    y_training = df_Xy_training['P'].values.ravel()

    return X_test, y_test, X_training, y_training, df_Xy_test.index.tolist()
'''
'''
# DATASET ADDRESS
arq = r"D:\Área de Trabalho (D)\TCC\Tabela_de_testes\teste2.csv"
# DATASET DISCRETIZATION
model1 = imli()
X1, y1 = model1.discretize(arq) #, categoricalColumnIndex=[1, 3, 5, 6, 7, 8, 9, 13]
model2 = IKKRR()
X2, y2 = model2.discretize(arq)

X_test_imli, y_test_imli, X_training_imli, y_training_imli, lines_imli = get_test_data(X1, y1, 0.5)
X_test, y_test, X_training, y_training, lines = get_test_data(X2, y2, 0.5, lines_imli)

print('Test IMLI:')
print(X_test_imli)
print('Test IKKRR:')
print(X_test)

print()
print('Lines Test IMLI')
print(lines_imli)
print()

print('Training IMLI:')
print(X_training_imli)
print('Training IKKRR:')
print(X_training)

print()
print('Lines Test IKKRR')
print(lines)
print()
'''