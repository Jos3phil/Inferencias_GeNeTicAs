from typing import List, Callable
import numpy as np


def _inicializar_poblacion(num_genes: int, tamano_poblacion: int, target_gene_index: int) -> List[np.ndarray]:
    """Inicializa la población con al menos un predictor por individuo."""
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