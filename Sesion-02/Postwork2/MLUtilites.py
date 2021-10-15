# • Funciones de evaluación con matriz de confusión.
# • Funciones de obtención de Precisión (Accuracy), Sensibilidad y Especificidad.
# • Funciones que comparen dos clasificadores: 
#     ◦ Obtengas precisión, sensibilidad y especificidad del clasificador 1
#     ◦ Obtengas precisión, sensibilidad y especificidad del clasificador 2
#     ◦ Digas cual es mejor en terminos de precisión
#     ◦ Digas cual es mejor en términos de sensibilidad
#     ◦ Digas cual es mejor en términos de especificidad.
# • Funciones de evaluación multiclase. 

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

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

    ejemplo
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
def separation_k_fold(data,K,):
    
    data = np.array([1,2,3,4,5,6,7,8,9,10,11,12])

    K = 4
    # random_seed = 48
    aleatorio = True
    
    kfold = KFold(K, aleatorio, random_seed)
    ciclo = 1
    for indices_train, indices_test in kfold.split(data):
        print("Ciclo: "+str(ciclo))
        print("\t datos para entrenamiento:"+str(data[indices_train]))
        print("\t datos para prueba:"+str(data[indices_test]))
        ciclo+=1