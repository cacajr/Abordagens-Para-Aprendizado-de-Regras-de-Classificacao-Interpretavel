import pandas as pd
import time

from Algoritmos.IMLI.imli import imli
from Algoritmos.An_existing_SAT_model.an_existing_model import an_existing_model as IKKRR
from Algoritmos.An_alternative_model.an_alternative_model import an_alternative_model as MinDS

def get_test_data(X, y, frac):
    # convert to dataframe
    df_X = pd.DataFrame(X)
    df_y = pd.DataFrame(y, columns=['P'])

    # concat dataframes X + y
    df_Xy = pd.concat([df_X, df_y], axis=1)

    # get % test
    df_Xy_test = df_Xy.sample(frac=frac)

    # create dataframe without samples (training)
    df_Xy_training = df_Xy.drop(df_Xy_test.index.tolist())

    # desconcat dataframe X - y and convert to matrix and vector respectively
    # test
    X_test = df_Xy_test.drop('P', axis=1).as_matrix()
    y_test = df_Xy_test['P'].values.ravel()

    # training
    X_training = df_Xy_training.drop('P', axis=1).as_matrix()
    y_training = df_Xy_training['P'].values.ravel()

    return X_test, y_test, X_training, y_training

arq = r"D:\Área de Trabalho (D)\TCC\Datasets\parkinsons.csv"

num_lines_per_partition = [8, 16, 32]
num_clauses = [1, 2, 3]
lambda_params = [5, 10]

model = IKKRR(solver="mifumax-win-mfc_static", numLinesPerPartition=16, numClause=2, dataFidelity=10)
X, y = model.discretize(arq) #, categoricalColumnIndex=[1, 3, 5, 6, 7, 8, 9, 13]

X_test, y_test, X_training, y_training = get_test_data(X, y, 0.2)

startTime = time.time()
model.fit(X_training, y_training)
endTime = time.time()

rule = model.getRule()

score = model.score(X_test, y_test)

print('============== RULE ==============')
print(rule)
print('==================================')
print('SET OF RULE SIZE: ', model.getRuleSize())
print('BIGGEST RULE SIZE: ', model.getBiggestRuleSize())
print('SCORE: ', score)
print('TIME DURATION (training): ', (endTime - startTime),'s')