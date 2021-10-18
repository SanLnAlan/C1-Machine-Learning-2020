import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split,LeaveOneOut,KFold
from sklearn.metrics import confusion_matrix
from sklearn import datasets
from sklearn.multiclass import OneVsRestClassifier,OneVsOneClassifier,OutputCodeClassifier
from sklearn.svm import LinearSVC


# • Funciones de separación de entrenamiento, validación y prueba. 
def train_validation_test(x, y, porcentaje_entrenamiento, porcentaje_validacion, porcentaje_prueba):
    """
     Función de separación de entrenamiento, validación y prueba. 
    
    argumentos:
    ----------

    x: datos de entrada

    y: datos de salida

    porcentaje_entrenamiento: porcentaje asignado para entrenamiento

    porcentaje_validacion: porcentaje asignado para validación

    porcentaje_prueba: porcentaje asignado para prueba

    Returno:
    ----------
    
    [x_entrenamiento, x_validacion, x_prueba, y_entrenamiento, y_validacion, y_prueba]


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
    Función de separación de datasets utlizando la validación cruzada de K-Fold

    argumentos:
    ----------

    data: dataset a trabajar

    K: número de ciclos o divisiones del dataset

    aleatorio: si es True, afecta el orden de los indices, el cual controla la aleatoridad de cada separación.

    Returno:
    -----------
    train: segmento de entrenamiento del dataset
    
    test: segmento de prueba del dataset

    ejemplo:
    -----------
    >>> train, test = ML.separation_k_fold(data,4,True)
    """
    train = []
    test = []
    if K == 1:
        loo = LeaveOneOut()
        loo.get_n_splits(data)
        for trained, tested in loo.split(data):
            # print("%s %s" % (train, test))
            print(f'{trained} {tested}')
            train.append(trained)
            test.append(tested)
        return train, test
    else:
        kfold = KFold(K, aleatorio)
        ciclo = 1
        
        for indices_train, indices_test in kfold.split(data):
            print("Ciclo: "+str(ciclo))
            print("\t datos para entrenamiento:"+str(data[indices_train]))
            print("\t datos para prueba:"+str(data[indices_test]))
            train.append(data[indices_train])
            test.append(data[indices_test])
            ciclo+=1
        return train, test


# • Funciones de evaluación con matriz de confusión.
def matrix_confusion(y_esperada,y_predicha, printed=False):
    """
    Función que obtiene la matriz de confusión mediante los datos binarios de los datos de salida predichos y esperados.

    argumentos:
    ----------

    y_esperada: salida de datos esperados

    y_predicha: salida de datos predichos

    printed: valor booleano que si es True imprime la matriz de confusión

    Returno:
    -----------
    TN: valores acertados negativos

    FP: valores no acertados positivos

    FN: valores no acertados negativos

    TP: valores acertados positivos

    ejemplo:
    -----------
    >>> TN, FP, FN, TP = ML.matrix_confusion(y_predichos,y_esperados)
    """
    result = confusion_matrix(y_esperada,y_predicha)
    TN, FP, FN, TP = result.ravel()
    if printed:
        print(f"matriz de confusión \n{result}")
    return TN, FP, FN, TP

# • Funciones de obtención de Precisión (Accuracy), Sensibilidad y Especificidad.
def results_matrix_confusion(TN, FP, FN, TP, printed=False):
    """
    Función que obtiene los valores de precisión, sensibilidad y especificidad mediante los valores obtenidos de la función matrix_confusion()

    argumentos:
    ----------

    TN: valores acertados negativos

    FP: valores no acertados positivos

    FN: valores no acertados negativos

    TP: valores acertados positivos

    printed: valor booleano que si es True imprime los resultados

    Returno:
    -----------
    accuracy: Precisión de la matriz de confusión

    sensitivity: Sensibilidad de la matriz de confusión

    specificity: Especificidad de la matriz de confusión

    ejemplo:
    -----------
    >>> accuracy, sensitivity, specificity = ML.results_matrix_confusion(TN, FP, FN, TP)
    """
    accuracy = (TP + TN) / (TP + TN + FP + FN) * 100
    sensitivity = (TP / (TP + FN) ) * 100
    specificity = (TN / (TN + FP) ) * 100
    if printed:
        print(f'precisión: {accuracy:.3f}')
        print(f'sensibilidad: {sensitivity:.3f}')
        print(f'specificidad: {specificity:.3f}')
    return accuracy, sensitivity, specificity


# • Funciones que comparen dos clasificadores: 
#     ◦ Obtengas precisión, sensibilidad y especificidad del clasificador 1
#     ◦ Obtengas precisión, sensibilidad y especificidad del clasificador 2
#     ◦ Digas cual es mejor en terminos de precisión
#     ◦ Digas cual es mejor en términos de sensibilidad
#     ◦ Digas cual es mejor en términos de especificidad.

def clasification_comparative(y_esperada,y_predicha,y_esperada2,y_predicha2):
    """
    Función que compara dos clasificadores para determinar cual es mejor en cuanto a precisión, sensibilidad y especificidad.

    argumentos:
    ----------

    y_esperada: salida de datos esperados del primer clasificador

    y_predicha: salida de datos predichos del primer clasificador

    y_esperada2: salida de datos esperados del segundo clasificador

    y_predicha2: salida de datos predichos del segundo clasificador


    Returno:
    -----------
    Imprime los resultados de las comparaciones realizadas

    ejemplo:
    -----------
    >>> ML.clasification_comparative(y_esperados,y_predichos,y_esperados2,y_predichos2)
    """
    accuracy = []
    sensitivity = []
    specificity = []
    y_esperada = [y_esperada,y_esperada2]
    y_predicha = [y_predicha,y_predicha2]
    for i in range(2):
        [TN, FP, FN, TP] = matrix_confusion(y_esperada[i],y_predicha[i])
        a, se, sp = results_matrix_confusion(TN, FP, FN, TP)
        accuracy.append(a)
        sensitivity.append(se)
        specificity.append(sp)
    if accuracy[0] > accuracy[1]:
        print(f'Clasificador 1 es mejor que clasificador 2 en cuanto a precisión: {accuracy[0]:.3f} > {accuracy[1]:.3f}')
    else:
        print(f'Clasificador 2 es mejor que clasificador 1 en cuanto a precisión: {accuracy[1]:.3f} > {accuracy[0]:.3f}')

    if sensitivity[0] > sensitivity[1]:
        print(f'Clasificador 1 es mejor que clasificador 2 en cuanto a sensibilidad: {sensitivity[0]:.3f} > {sensitivity[1]:.3f}')
    else:
        print(f'Clasificador 2 es mejor que clasificador 1 en cuanto a sensibilidad: {sensitivity[1]:.3f} > {sensitivity[0]:.3f}')

    if specificity[0] > specificity[1]:
        print(f'Clasificador 1 es mejor que clasificador 2 en cuanto a especificidad: {specificity[0]} > {specificity[1]:.3f}')
    else:
        print(f'Clasificador 2 es mejor que clasificador 1 en cuanto a especificidad: {specificity[1]} > {specificity[0]:.3f}')


# • Funciones de evaluación multiclase. 
def multiclass_test(X,y, multiclass_type="OneVsRest"):
    """
    Función que realiza una evaluación multiclase

    argumentos:
    ----------

    X: datos de entrada

    y: datos de salida

    multiclass_type: "OneVsRest" o "OneVsOne" o "OutputCode"


    Returno:
    -----------
    resultado: resultado obtenido de la evaluación multiclase

    ejemplo:
    -----------
    >>> from sklearn import datasets
    >>> X, y = datasets.load_iris(return_X_y=True)
    >>> r = ML.multiclass_test(X,y,"OneVsOne")

    """
    if multiclass_type == "OneVsRest":
        result = OneVsRestClassifier(LinearSVC(random_state=0)).fit(X, y).predict(X)
        return result
    if multiclass_type == "OneVsOne":
        result = OneVsOneClassifier(LinearSVC(random_state=0)).fit(X, y).predict(X)
        return result
    if multiclass_type == "OutputCode":
        result =  OutputCodeClassifier(LinearSVC(random_state=0), code_size=2, random_state=0).fit(X,y).predict(X)
        return result 
