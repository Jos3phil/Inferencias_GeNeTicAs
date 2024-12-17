import random
import numpy as np
from pyeasyga import pyeasyga
# --- Clase problemas de Algoritmos Geneticos que resuelve MiniSudokus 4x4
class structureAG:
  # -- Constructor
  def __init__(self, problema, nroIndividuos = 500):
    self.nombre = 'Problema de Selección de Predictores'
    self.problema = problema
    self.individuo = list(range(len(problema)))
    self.nroIndividuos = nroIndividuos
    self.poblacion = []
    self.mejor_solucion = None
    self.mejor_fitness = float('-inf')

  # -- Nombre
  def __str__(self):
    return self.nombre

  # -- Crear individuo
  def FnIndividuo(self, individuo):
      individuoNuevo = individuo[:]
      random.shuffle(individuoNuevo)
      return individuoNuevo

  # -- Función de selección
  def FnSeleccion(self, poblacion):
      return random.choice(poblacion)

  # -- Función de cruce
  def FnCruce(self, padre1, padre2):
    # -- Determinar aleatoriamente el índice para el cruce
    indiceCruce = np.random.randint(1, len(padre1))
    hijo1 = np.concatenate((padre1[:indiceCruce], padre2[indiceCruce:]))
    hijo2 = np.concatenate((padre2[:indiceCruce], padre1[indiceCruce:]))
    
    return hijo1, hijo2
  # -- Función de mutación
  def FnMutacion(self, individuo):
    # -- Se intercambia los genes de las posiciones dadas por los índices
      for i in range(len(individuo)):
        if np.random.rand() < 0.2:
            j = np.random.randint(0,len(individuo))
            individuo[i], individuo[j] = individuo[j], individuo[i]

  # -- Definir la función de aptitud del algoritmo genético
  def FnAptitud(self, individuo, matriz_expresion, target_gene_index, funcion_criterio):
      # -- La función de aptitud no se debe aplicar directamente a los individuos
      #    de la población, sino, a las soluciones del problema.
      #    En consecuencia con los individuos y el problema se debe generar
      #    una nueva estructura que represente al individuo solución
      predictores = [self.problema[i] for i in individuo]
      return funcion_criterio(matriz_expresion, target_gene_index, predictores)

  # -- Ejecutar algoritmo genético
  def Ejecutar(self, matriz_expresion, target_gene_index, funcion_criterio):
        """Ejecuta el algoritmo genético."""
        # Inicializar población
        self.poblacion = [self.FnIndividuo(self.individuo) for _ in range(self.nroIndividuos)]
        
        # Evolución por generaciones
        for gen in range(100):  # 100 generaciones
            # Evaluar fitness
            fitness_valores = [self.FnAptitud(ind, matriz_expresion, target_gene_index, funcion_criterio) 
                             for ind in self.poblacion]
            
            # Actualizar mejor solución
            max_idx = np.argmax(fitness_valores)
            if fitness_valores[max_idx] > self.mejor_fitness:
                self.mejor_fitness = fitness_valores[max_idx]
                self.mejor_solucion = (self.poblacion[max_idx], 
                                     [self.problema[i] for i in self.poblacion[max_idx]])
                
            # Selección
            nueva_poblacion = []
            for _ in range(self.nroIndividuos):
                padre1 = self.FnSeleccion(self.poblacion)
                padre2 = self.FnSeleccion(self.poblacion)
                
                # Cruce
                if random.random() < 0.8:  # probabilidad de cruce
                    hijo1, hijo2 = self.FnCruce(padre1, padre2)
                else:
                    hijo1, hijo2 = padre1[:], padre2[:]
                
                # Mutación
                self.FnMutacion(hijo1)
                self.FnMutacion(hijo2)
                
                nueva_poblacion.extend([hijo1, hijo2])
            
            # Actualizar población
            self.poblacion = nueva_poblacion[:self.nroIndividuos]
            
            if gen % 10 == 0:
                print(f"Generación {gen}: Mejor fitness = {self.mejor_fitness}")
        
        return self.mejor_solucion