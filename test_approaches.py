import pandas as pd
import re
import time
import csv

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
    X_test = df_Xy_test.drop('P', axis=1).as_matrix()
    y_test = df_Xy_test['P'].values.ravel()

    # training
    X_training = df_Xy_training.drop('P', axis=1).as_matrix()
    y_training = df_Xy_training['P'].values.ravel()

    return X_test, y_test, X_training, y_training, df_Xy_test.index.tolist()

# DATASET ADDRESS, NAME AND CATEGORICAL COLUMNS
arq = r"D:\Área de Trabalho (D)\TCC\Datasets\toms.csv"
name = 'toms(3)'
categorical_columns = []

# DATASET DISCRETIZATION
model1 = imli()
X1, y1, column_info_imli, columns_imli = model1.discretize(arq, categoricalColumnIndex=categorical_columns)
model2 = IKKRR()
X2, y2, column_info_ikkrr, columns_ikkrr = model2.discretize(arq, categoricalColumnIndex=categorical_columns)
model3 = IMinDS()
X3, y3, column_info_iminds, columns_iminds = model3.discretize(arq, categoricalColumnIndex=categorical_columns)

# MODELS CONFIGURATIONS
num_lines_per_partition = [8, 16]
num_clauses = [1, 2, 3]
lambda_params = [5, 10]

# PERFORMANCE INFORMATION FOR EACH MODEL FOR EACH ROUND. EX: [['Rule set size', 'Biggest rule', 'Accuracy', 'Training time'],[6, 3, 0.87, 0.8000]]
performance_imli = [['Configuration', 'Rule set size', 'Biggest rule', 'Accuracy', 'Training time']]
performance_ikkrr = [['Configuration', 'Rule set size', 'Biggest rule', 'Accuracy', 'Training time']]
performance_iminds = [['Configuration', 'Rule set size', 'Biggest rule', 'Accuracy', 'Training time']]

rounds = 10
for i in range(rounds):
    # DATASET SEPARATION (TRAINING and TEST)
    X_test_imli, y_test_imli, X_training_imli, y_training_imli, lines_imli = get_test_data(X1, y1, 0.1)
    X_test_ikkrr, y_test_ikkrr, X_training_ikkrr, y_training_ikkrr, lines = get_test_data(X2, y2, 0.1, lines_imli)
    X_test_iminds, y_test_iminds, X_training_iminds, y_training_iminds, lines = get_test_data(X3, y3, 0.1, lines_imli)

    # DATASET TRAINING SEPARATION (TRAINING and TEST)
    X_training_test_imli, y_training_test_imli, X_training_training_imli, y_training_training_imli, lines_imli = get_test_data(X_training_imli, y_training_imli, 0.1)
    X_training_test_ikkrr, y_training_test_ikkrr, X_training_training_ikkrr, y_training_training_ikkrr, lines = get_test_data(X_training_ikkrr, y_training_ikkrr, 0.1, lines_imli)
    X_training_test_iminds, y_training_test_iminds, X_training_training_iminds, y_training_training_iminds, lines = get_test_data(X_training_iminds, y_training_iminds, 0.1, lines_imli)

    # ACCURACY INFORMATION FOR EACH MODEL. EX: [['config1', ..., 'config18'], [0.70, ..., 0.83]]
    configuration_and_accuracy_imli = [[],[]]
    configuration_and_accuracy_ikkrr = [[],[]]
    configuration_and_accuracy_iminds = [[],[]]

    number_config = 1
    for lpp in num_lines_per_partition:
        for nc in num_clauses:
            for lp in lambda_params:
                # MODELS INSTANCE WITH CONFIGURATION
                model_imli = imli(solver="mifumax-win-mfc_static", numLinesPerPartition=lpp, numClause=nc, dataFidelity=lp, columnInfo=column_info_imli, columns=columns_imli)
                model_ikkrr = IKKRR(solver="mifumax-win-mfc_static", numLinesPerPartition=lpp, numClause=nc, dataFidelity=lp, columnInfo=column_info_ikkrr, columns=columns_ikkrr)
                model_iminds = IMinDS(solver="mifumax-win-mfc_static", numLinesPerPartition=lpp, numClause=nc, dataFidelity=lp, columnInfo=column_info_iminds, columns=columns_iminds)

                # TRAINING MODELS WITH DATASET TRAINING's TRAINING
                model_imli.fit(X_training_training_imli, y_training_training_imli)
                model_ikkrr.fit(X_training_training_ikkrr, y_training_training_ikkrr)
                model_iminds.fit(X_training_training_iminds, y_training_training_iminds)

                # GETTING ACCURACY'S MODELS AND SAVE INFORMATIONS
                configuration_and_accuracy_imli[0].append('lpp: '+str(lpp)+' | nc: '+str(nc)+' | lp: '+str(lp))
                configuration_and_accuracy_imli[1].append(model_imli.score(X_training_test_imli, y_training_test_imli))

                configuration_and_accuracy_ikkrr[0].append('lpp: '+str(lpp)+' | nc: '+str(nc)+' | lp: '+str(lp))
                configuration_and_accuracy_ikkrr[1].append(model_ikkrr.score(X_training_test_ikkrr, y_training_test_ikkrr))

                configuration_and_accuracy_iminds[0].append('lpp: '+str(lpp)+' | nc: '+str(nc)+' | lp: '+str(lp))
                configuration_and_accuracy_iminds[1].append(model_iminds.score(X_training_test_iminds, y_training_test_iminds))

                f = open('./logs/'+name+'_test_informations.csv', 'w', newline='', encoding='utf-8')
                w = csv.writer(f)
                w.writerow(["ROUND: "+str(i+1)+"/10 | CONFIG: (lpp: "+str(lpp)+" | nc: "+str(nc)+" | lp: "+str(lp)+") "+str(number_config)+"/12"])
                f.close()
                number_config += 1

    # TAKE THE BEST CONFIGURATION TO EACH MODEL EX: 'lpp: 8 | nc: 1 | lp: 5'
    index_biggest_accuracy_imli = configuration_and_accuracy_imli[1].index(max(configuration_and_accuracy_imli[1]))
    best_configuration_imli = configuration_and_accuracy_imli[0][index_biggest_accuracy_imli]

    index_biggest_accuracy_ikkrr = configuration_and_accuracy_ikkrr[1].index(max(configuration_and_accuracy_ikkrr[1]))
    best_configuration_ikkrr = configuration_and_accuracy_ikkrr[0][index_biggest_accuracy_ikkrr]

    index_biggest_accuracy_iminds = configuration_and_accuracy_iminds[1].index(max(configuration_and_accuracy_iminds[1]))
    best_configuration_iminds = configuration_and_accuracy_iminds[0][index_biggest_accuracy_iminds]

    # TAKE THE BEST CONFIGURATION TO EACHE MODEL IN INT VECTOR. EX: [8, 1, 5]
    configuration_imli = [int(s) for s in re.findall(r'\b\d+\b', best_configuration_imli)]

    configuration_ikkrr = [int(s) for s in re.findall(r'\b\d+\b', best_configuration_ikkrr)]

    configuration_iminds = [int(s) for s in re.findall(r'\b\d+\b', best_configuration_iminds)]

    # MODELS INSTANCE WITH THE BEST CONFIGURATION
    model_imli = imli(solver="mifumax-win-mfc_static", numLinesPerPartition=configuration_imli[0], numClause=configuration_imli[1], dataFidelity=configuration_imli[2], columnInfo=column_info_imli, columns=columns_imli)
    model_ikkrr = IKKRR(solver="mifumax-win-mfc_static", numLinesPerPartition=configuration_ikkrr[0], numClause=configuration_ikkrr[1], dataFidelity=configuration_ikkrr[2], columnInfo=column_info_ikkrr, columns=columns_ikkrr)
    model_iminds = IMinDS(solver="mifumax-win-mfc_static", numLinesPerPartition=configuration_iminds[0], numClause=configuration_iminds[1], dataFidelity=configuration_iminds[2], columnInfo=column_info_iminds, columns=columns_iminds)

    # TRAINING MODELS WITH DATASET TRAINING AND TAKING THE TIME
    start_time_imli = time.time()
    model_imli.fit(X_training_imli, y_training_imli)
    end_time_imli = time.time()

    start_time_ikkrr = time.time()
    model_ikkrr.fit(X_training_ikkrr, y_training_ikkrr)
    end_time_ikkrr = time.time()

    start_time_iminds = time.time()
    model_iminds.fit(X_training_iminds, y_training_iminds)
    end_time_iminds = time.time()

    # GETTING SIZE'S SET OF RULES (|R|), BIGGEST RULE (max(R)), ACCURACY AND TIME TRAINING TO EACH MODEL AND SAVE INFORMATIONS
    size_set_of_rules_imli = model_imli.getRuleSize()
    size_biggest_rule_imli = model_imli.getBiggestRuleSize()
    accuracy_imli = model_imli.score(X_test_imli, y_test_imli)
    time_training_imli = end_time_imli - start_time_imli
    performance_imli.append([best_configuration_imli, size_set_of_rules_imli, size_biggest_rule_imli, accuracy_imli, time_training_imli])

    size_set_of_rules_ikkrr = model_ikkrr.getRuleSize()
    size_biggest_rule_ikkrr = model_ikkrr.getBiggestRuleSize()
    accuracy_ikkrr = model_ikkrr.score(X_test_ikkrr, y_test_ikkrr)
    time_training_ikkrr = end_time_ikkrr - start_time_ikkrr
    performance_ikkrr.append([best_configuration_ikkrr, size_set_of_rules_ikkrr, size_biggest_rule_ikkrr, accuracy_ikkrr, time_training_ikkrr])

    size_set_of_rules_iminds = model_iminds.getRuleSize()
    size_biggest_rule_iminds = model_iminds.getBiggestRuleSize()
    accuracy_iminds = model_iminds.score(X_test_iminds, y_test_iminds)
    time_training_iminds = end_time_iminds - start_time_iminds
    performance_iminds.append([best_configuration_iminds, size_set_of_rules_iminds, size_biggest_rule_iminds, accuracy_iminds, time_training_iminds])

# WRITING INFORMATIONS IN CSV
def average(matrix):
    average_each_column = []

    for j in range(1, len(matrix[0])):
        average_each_column.append(0)

        for i in range(1, len(matrix)):
            average_each_column[j-1] += matrix[i][j]
        
        average_each_column[j-1] = average_each_column[j-1]/(len(matrix)-1)
    
    return average_each_column

f = open('./Tests_informations/'+name+'_test_informations.csv', 'w', newline='', encoding='utf-8')
w = csv.writer(f)

w.writerow(['IMLI', '', '', '', ''])
for i in performance_imli:
    w.writerow(i)
w.writerow(['Averages'] + average(performance_imli))

w.writerow(['IKKRR', '', '', '', ''])
for i in performance_ikkrr:
    w.writerow(i)
w.writerow(['Averages'] + average(performance_ikkrr))

w.writerow(['IMinDS', '', '', '', ''])
for i in performance_iminds:
    w.writerow(i)
w.writerow(['Averages'] + average(performance_iminds))

f.close()