# • Funciones de evaluación multiclase. 

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split,LeaveOneOut,KFold
from sklearn.metrics import confusion_matrix

# • Funciones de separación de entrenamiento, validación y prueba. 
def train_validation_test(x, y, porcentaje_entrenamiento, porcentaje_validacion, porcentaje_prueba):
    """
     Función de separación de entrenamiento, validación y prueba. 
    
    argumentos:
    ----------

    x: 

    y:

    porcentaje_entrenamiento:

    porcentaje_validacion:

    porcentaje_prueba:

    Returno:
    ----------

    ejemplo:
    ----------
  
        >>> from sklearn import datasets
        >>> diabetes = datasets.load_diabetes()
        >>> x = diabetes.data[:150]
        >>> y = diabetes.target[:150]
        >>> [x_train, x_val, x_test, y_train, y_val, y_test] = train_validation_test(x,y,0.60,0.10,0.30)

    """
    temp_size = porcentaje_validacion + porcentaje_prueba
    
    x_train, x_temp, y_train, y_temp = train_test_split(x, y, test_size =temp_size)
    if(porcentaje_validacion > 0):
        test_size = porcentaje_prueba/temp_size
        x_val, x_test, y_val, y_test = train_test_split(x_temp, y_temp, test_size = test_size)
    else:
        return [x_train, None, x_temp, y_train, None, y_temp]
    return [x_train, x_val, x_test, y_train, y_val, y_test]


# • Funciones de separación de datasets con K-Fold (el usuario debe poner el K, si K = 1 debe generar un Leave-One-Out Cross Validation).
def separation_k_fold(data,K,aleatorio=True):
    """
    ejemplo:
    -----------
    >>> train, test = ML.separation_k_fold(data,4,True)
    """
    if K == 1:
        loo = LeaveOneOut()
        for train, test in loo.split(data):
            print("%s %s" % (train, test))
            return train, test
    else:
        kfold = KFold(K, aleatorio)
        ciclo = 1
        train = []
        test = []
        for indices_train, indices_test in kfold.split(data):
            print("Ciclo: "+str(ciclo))
            print("\t datos para entrenamiento:"+str(data[indices_train]))
            print("\t datos para prueba:"+str(data[indices_test]))
            train.append(data[indices_train])
            test.append(data[indices_test])
            ciclo+=1
        return train, test


# • Funciones de evaluación con matriz de confusión.
def matrix_confusion(y_esperada,y_predicha):
    result = confusion_matrix(y_esperada,y_predicha)
    [TN, FP, FN, TP] = result.ravel()
    return [TN, FP, FN, TP]

# • Funciones de obtención de Precisión (Accuracy), Sensibilidad y Especificidad.
def results_matrix_confusion(TN, FP, FN, TP):
    accuracy = (TP / (TP + FP) ) * 100
    sensitivity = (TP / (TP + FN) ) * 100
    specificity = (TN / (TN + FP) ) * 100
    return accuracy, sensitivity, specificity


# • Funciones que comparen dos clasificadores: 
#     ◦ Obtengas precisión, sensibilidad y especificidad del clasificador 1
#     ◦ Obtengas precisión, sensibilidad y especificidad del clasificador 2
#     ◦ Digas cual es mejor en terminos de precisión
#     ◦ Digas cual es mejor en términos de sensibilidad
#     ◦ Digas cual es mejor en términos de especificidad.

def clasification_comparative(y_esperada1,y_predicha1,y_esperada2,y_predicha2):
    accuracy = []
    sensitivity = []
    specificity = []
    [TN, FP, FN, TP] = matrix_confusion(y_esperada1,y_predicha1)
    accuracy[0], sensitivity[0], specificity[0] = results_matrix_confusion(TN, FP, FN, TP)
    [TN, FP, FN, TP] = matrix_confusion(y_esperada2,y_predicha2)
    accuracy[1], sensitivity[1], specificity[1] = results_matrix_confusion(TN, FP, FN, TP)
    if accuracy[0] > accuracy[1]:
        print('Clasficador 1 es mejor que clasificador 2 en cuanto a precisión: {accuracy[0]} > {accuracy[1]}')
    else:
        print('Clasficador 2 es mejor que clasificador 1 en cuanto a precisión: {accuracy[1]} < {accuracy[0]}')

    if sensitivity[0] > sensitivity[1]:
        print('Clasficador 1 es mejor que clasificador 2 en cuanto a sensibilidad: {sensitivity[0]} > {sensitivity[1]}')
    else:
        print('Clasficador 2 es mejor que clasificador 1 en cuanto a sensibilidad: {sensitivity[1]} < {sensitivity[0]}')

    if specificity[0] > specificity[1]:
        print('Clasficador 1 es mejor que clasificador 2 en cuanto a specificidad: {specificity[0]} > {specificity[1]}')
    else:
        print('Clasficador 2 es mejor que clasificador 1 en cuanto a specificidad: {specificity[1]} < {specificity[0]}')