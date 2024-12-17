from typing import List, Callable
import numpy as np

def _inicializar_poblacion(num_genes: int, tamano_poblacion: int, target_gene_index: int) -> List[np.ndarray]:
    """Inicializa población con al menos un predictor por individuo."""
    poblacion = []
    for _ in range(tamano_poblacion):
        individuo = np.zeros(num_genes)
        # Asegurar al menos un predictor
        num_predictores = np.random.randint(1, min(4, num_genes-1))
        predictores_disponibles = list(set(range(num_genes)) - {target_gene_index})
        predictores_seleccionados = np.random.choice(predictores_disponibles, num_predictores, replace=False)
        individuo[predictores_seleccionados] = 1
        poblacion.append(individuo)
    return poblacion

def _evaluar_fitness(poblacion: List[np.ndarray], matriz_expresion: np.ndarray, target_gene_index: int, funcion_criterio: Callable) -> np.ndarray:
    """Evalúa el fitness de cada individuo."""
    fitness = np.zeros(len(poblacion))
    for i, individuo in enumerate(poblacion):
        # Obtener índices de predictores (donde hay 1s)
        predictores = np.where(individuo == 1)[0]
        if len(predictores) == 0:
            fitness[i] = float('-inf')
        else:
            # Evaluar con función de criterio
            fitness[i] = funcion_criterio(matriz_expresion, target_gene_index, predictores.tolist())
    return fitness

def _seleccionar_padres(poblacion: List[np.ndarray], fitness_valores: np.ndarray) -> List[np.ndarray]:
    """Selecciona padres usando selección por torneo."""
    padres = []
    tamano_torneo = 3
    
    for _ in range(len(poblacion)):
        # Seleccionar participantes para el torneo
        indices_torneo = np.random.choice(len(poblacion), tamano_torneo, replace=False)
        fitness_torneo = fitness_valores[indices_torneo]
        
        # Seleccionar el mejor del torneo
        ganador_idx = indices_torneo[np.argmax(fitness_torneo)]
        padres.append(poblacion[ganador_idx].copy())
    
    return padres

def _cruzar_poblacion(padres: List[np.ndarray], prob_cruce: float) -> List[np.ndarray]:
    """Realiza el cruce entre pares de padres."""
    descendencia = []
    for i in range(0, len(padres), 2):
       if i+1 < len(padres):
            if np.random.rand() < prob_cruce:
                #Cruza en un punto al azar
                punto_cruce = np.random.randint(1, len(padres[0]))
                descendiente1 = np.concatenate((padres[i][:punto_cruce], padres[i+1][punto_cruce:]))
                descendiente2 = np.concatenate((padres[i+1][:punto_cruce], padres[i][punto_cruce:]))
                descendencia.append(descendiente1)
                descendencia.append(descendiente2)
            else:
                descendencia.append(padres[i])
                descendencia.append(padres[i+1])
    return descendencia
def _mutar_poblacion(descendencia: List[np.ndarray], prob_mutacion: float, target_gene_index: int) -> List[np.ndarray]:
    """Realiza mutaciones aleatorias en la descendencia."""
    descendencia_mutada = []
    for individuo in descendencia:
        for i in range(len(individuo)):
          if i!= target_gene_index: # no hay mutación en el gen objetivo
            if np.random.rand() < prob_mutacion:
               individuo[i] = 1 - individuo[i]
        descendencia_mutada.append(individuo)
    return descendencia_mutada