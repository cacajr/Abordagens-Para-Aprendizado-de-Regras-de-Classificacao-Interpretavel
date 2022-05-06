import pandas as pd

from Algoritmos.IMLI.imli import imli
from Algoritmos.An_existing_SAT_model.an_existing_model import an_existing_model as IKKRR
from Algoritmos.An_alternative_model.an_alternative_model import an_alternative_model as IMinDS


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
    else:
        # get % test
        df_Xy_test = df_Xy.loc[lines]

        # create dataframe without samples (training)
        df_Xy_training = df_Xy.drop(lines)

    # desconcat dataframe X - y and convert to matrix and vector respectively
    # test
    X_test = df_Xy_test.drop('P', axis=1).values
    y_test = df_Xy_test['P'].values.ravel()

    # training
    X_training = df_Xy_training.drop('P', axis=1).values
    y_training = df_Xy_training['P'].values.ravel()

    return X_test, y_test, X_training, y_training, df_Xy_test.index.tolist()

# DATASET ADDRESS, NAME AND CATEGORICAL COLUMNS
arq = './Datasets/lung_cancer.csv'
columns = []

# MODELS CONFIGURATIONS
num_lines_per_partition = 8 # [8, 16]
num_clauses = 2 # [1, 2, 3]
lambda_params =  10 # [5, 10]

# MODELS INSTANCE WITH CONFIGURATION
model_imli = imli(solver="./Solvers/wmifumax-0.9/wmifumax", numLinesPerPartition=num_lines_per_partition, numClause=num_clauses, dataFidelity=lambda_params)
X1, y1, column_info_imli, columns_imli = model_imli.discretize(arq, categoricalColumnIndex=columns)
model_ikkrr = IKKRR(solver="./Solvers/wmifumax-0.9/wmifumax", numLinesPerPartition=num_lines_per_partition, numClause=num_clauses, dataFidelity=lambda_params)
X2, y2, column_info_ikkrr, columns_ikkrr = model_ikkrr.discretize(arq, categoricalColumnIndex=columns)
model_iminds = IMinDS(solver="./Solvers/wmifumax-0.9/wmifumax", numLinesPerPartition=num_lines_per_partition, numClause=num_clauses, dataFidelity=lambda_params)
X3, y3, column_info_iminds, columns_iminds = model_iminds.discretize(arq, categoricalColumnIndex=columns)

# DATA SET SEPARATION TRAINING AND TEST
X_test_imli, y_test_imli, X_training_imli, y_training_imli, lines_imli = get_test_data(X1, y1, 0.2)
X_test_ikkrr, y_test_ikkrr, X_training_ikkrr, y_training_ikkrr, _ = get_test_data(X2, y2, 0.2, lines_imli)
X_test_iminds, y_test_iminds, X_training_iminds, y_training_iminds, _ = get_test_data(X3, y3, 0.2, lines_imli)

# TRAINING MODELS WITH DATASET TRAINING's TRAINING
model_imli.fit(X_training_imli, y_training_imli)
model_ikkrr.fit(X_training_ikkrr, y_training_ikkrr)
model_iminds.fit(X_training_iminds, y_training_iminds)

# TEST ACCURACY WITH TEST DATA
print(model_imli.score(X_test_imli, y_test_imli))
print(model_ikkrr.score(X_test_ikkrr, y_test_ikkrr))
print(model_iminds.score(X_test_iminds, y_test_iminds))

# RULES GENERATE TO EXPLAIN PREDICTIONS
print(model_imli.getRule())
print(model_ikkrr.getRule())
print(model_iminds.getRule())
