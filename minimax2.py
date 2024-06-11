import numpy as np
import random
import time
import pygame
import sys

# Clase EstadoJuego
class EstadoJuego:
    def __init__(self, gatopos, ratonpos, tabtamaño, raton_pos_anterior=None):
        self.gatopos = np.array(gatopos)  # Posición del gato como un array numpy (x, y)
        self.ratonpos = np.array(ratonpos)  # Posición del ratón como un array numpy (x, y)
        self.tabtamaño = tabtamaño  # Tamaño del tablero, asumiendo que es un tablero cuadrado de tabtamaño x tabtamaño
        self.raton_pos_anterior = raton_pos_anterior  # Posición anterior del ratón

    def termino(self):
        # El juego termina si el gato atrapa al ratón
        return np.array_equal(self.gatopos, self.ratonpos)

    def distancia_gato_raton(self):
        return np.linalg.norm(self.gatopos - self.ratonpos)

    def posibles_movimientos(self, pos):
        movimientos = []
        x, y = pos
        direcciones = np.array([[-1, 0], [1, 0], [0, -1], [0, 1], [-1, -1], [-1, 1], [1, -1], [1, 1]])  # Agregar movimientos diagonales

        for dx, dy in direcciones:
            nuevo_pos = np.array([x + dx, y + dy]) #Calcula la nueva posición sumando la dirección actual (dx, dy) a la posición (x, y).
            if np.all(nuevo_pos >= 0) and np.all(nuevo_pos < self.tabtamaño): #Verifica si la nueva posición está dentro de los límites del tablero
                movimientos.append(tuple(nuevo_pos))

        return movimientos

    def movimiento_gato(self, nueva_pos): #Esta función crea un nuevo estado del juego después de que el gato se mueve a una nueva posición
        if self.termino():
            return None
        return EstadoJuego(nueva_pos, self.ratonpos, self.tabtamaño, self.raton_pos_anterior) #Si el juego no ha terminado, crea y devuelve un nuevo estado del juego

    def movimiento_raton(self, nueva_pos):
        if self.termino():
            return None
        return EstadoJuego(self.gatopos, nueva_pos, self.tabtamaño, self.ratonpos)

    def evaluacion(self): #Esta función evalúa el estado actual del juego para determinar qué tan favorable es para el gato
        if np.array_equal(self.gatopos, self.ratonpos): #Verifica si el gato ha atrapado al ratón usando
            return np.inf  # Si el gato ha atrapado al ratón, devuelve np.inf (infinito positivo), indicando que este es el mejor resultado posible para el gato
        return -np.linalg.norm(self.gatopos - self.ratonpos) #Si el gato no ha atrapado al ratón, calcula la distancia euclidiana entre el gato y el ratón 

# Algoritmo minimax con poda alpha-beta
def minimax(estado, profundidad, alpha, beta, maximizar_jugador):
    if profundidad == 0 or estado.termino():
        return estado.evaluacion()

    if maximizar_jugador:  # Turno del gato
        max_eval = -np.inf
        for movimiento in estado.posibles_movimientos(estado.gatopos):
            nuevo_estado = estado.movimiento_gato(movimiento)
            if nuevo_estado is None: #Si nuevo_estado es None (el juego ha terminado), se salta a la siguiente iteración.
                continue
            eval = minimax(nuevo_estado, profundidad - 1, alpha, beta, False)
            max_eval = max(max_eval, eval) #Se actualiza max_eval con el valor máximo entre max_eval y el valor retornado por la llamada recursiva.
            alpha = max(alpha, eval)
            if beta <= alpha:
                break
        return max_eval #Devuelve max_eval, que es el mejor valor encontrado para el gato en este nivel de la búsqueda.
    else:  # Turno del ratón
        min_eval = np.inf #min_eval se inicializa con np.inf (peor valor posible para el ratón).
        for movimiento in estado.posibles_movimientos(estado.ratonpos):
            nuevo_estado = estado.movimiento_raton(movimiento)
            if nuevo_estado is None: 
                continue
            eval = minimax(nuevo_estado, profundidad - 1, alpha, beta, True) #Se actualiza min_eval con el valor mínimo entre min_eval y el valor retornado por la llamada recursiva.
            min_eval = min(min_eval, eval) 
            beta = min(beta, eval)
            if beta <= alpha:
                break
        return min_eval

# Función para encontrar el mejor movimiento del gato o del ratón
def mejor_movimiento(estado_inicial, profundidad_inicial, maximizar_jugador):
    mejor_mov = None
    mejor_valor = -np.inf if maximizar_jugador else np.inf

    posibles_movimientos = estado_inicial.posibles_movimientos(estado_inicial.gatopos if maximizar_jugador else estado_inicial.ratonpos)

    for movimiento in posibles_movimientos:
        nuevo_estado = estado_inicial.movimiento_gato(movimiento) if maximizar_jugador else estado_inicial.movimiento_raton(movimiento)
        if nuevo_estado is None:
            continue
        valor = minimax(nuevo_estado, profundidad_inicial - 1, -np.inf, np.inf, not maximizar_jugador)
        
        if maximizar_jugador:
            if valor > mejor_valor:
                mejor_valor = valor
                mejor_mov = movimiento
        else:
            if valor < mejor_valor:
                mejor_valor = valor
                mejor_mov = movimiento

    return mejor_mov

# Inicializar Pygame
pygame.init()

# Tamaño del tablero y de la ventana
tabtamaño = 7
ancho_celda = 100
altura_celda = 100
ancho_ventana = tabtamaño * ancho_celda
altura_ventana = tabtamaño * altura_celda
ventana = pygame.display.set_mode((ancho_ventana, altura_ventana))
pygame.display.set_caption("Gato y Ratón")  # Corrección aquí

# Colores
COLOR_FONDO = (255, 255, 255)
COLOR_CELDA = (200, 200, 200)

# Cargar imágenes
imagen_gato = pygame.image.load('gato.png')
imagen_gato = pygame.transform.scale(imagen_gato, (ancho_celda, altura_celda))
imagen_raton = pygame.image.load('raton.png')
imagen_raton = pygame.transform.scale(imagen_raton, (ancho_celda, altura_celda))
imagen_final_gato = pygame.image.load('gato_atrapando_raton.png')
imagen_final_gato = pygame.transform.scale(imagen_final_gato, (ancho_ventana, altura_ventana))
imagen_final_raton = pygame.image.load('raton_escapado.png')
imagen_final_raton = pygame.transform.scale(imagen_final_raton, (ancho_ventana, altura_ventana))

# Función para generar posiciones iniciales no adyacentes
def generar_posiciones_iniciales(tabtamaño):
    while True:
        gatopos = (random.randint(0, tabtamaño - 1), random.randint(0, tabtamaño - 1))
        ratonpos = (random.randint(0, tabtamaño - 1), random.randint(0, tabtamaño - 1))
        if np.linalg.norm(np.array(gatopos) - np.array(ratonpos)) > 1:
            return gatopos, ratonpos

# Estado inicial con posiciones aleatorias no adyacentes para el gato y el ratón
gatopos, ratonpos = generar_posiciones_iniciales(tabtamaño)
estado_inicial = EstadoJuego(gatopos, ratonpos, tabtamaño)
profundidad_inicial = 3  # Incrementamos la profundidad para mejor capacidad de decisión

# Función para dibujar el tablero
def dibujar_tablero(estado):
    ventana.fill(COLOR_FONDO)
    for fila in range(tabtamaño):
        for columna in range(tabtamaño):
            rect = pygame.Rect(columna * ancho_celda, fila * altura_celda, ancho_celda, altura_celda)
            pygame.draw.rect(ventana, COLOR_CELDA, rect, 1)

    # Dibujar el gato
    gx, gy = estado.gatopos
    ventana.blit(imagen_gato, (gy * ancho_celda, gx * altura_celda))

    # Dibujar el ratón
    rx, ry = estado.ratonpos
    ventana.blit(imagen_raton, (ry * ancho_celda, rx * altura_celda))

    pygame.display.flip()

# Función para mover el ratón priorizando alejarse del gato y si no puede, moverse en diagonal
def mover_raton_priorizando_alejarse(estado):
    posibles_mov_raton = estado.posibles_movimientos(estado.ratonpos)
    raton_pos = estado.ratonpos
    gato_pos = estado.gatopos
    distancia_actual = np.linalg.norm(raton_pos - gato_pos)
    
    # Eliminar la posición anterior del ratón de los posibles movimientos
    if estado.raton_pos_anterior is not None:
        posibles_mov_raton = [mov for mov in posibles_mov_raton if mov != tuple(estado.raton_pos_anterior)]

    # Filtrar movimientos que alejen al ratón del gato
    movimientos_alejarse = [mov for mov in posibles_mov_raton if np.linalg.norm(np.array(mov) - gato_pos) > distancia_actual]

    if movimientos_alejarse:
        return random.choice(movimientos_alejarse)

    # Si no puede alejarse, priorizar movimientos diagonales
    movimientos_diagonales = [mov for mov in posibles_mov_raton if abs(mov[0] - gato_pos[0]) == 1 and abs(mov[1] - gato_pos[1]) == 1]
    if movimientos_diagonales:
        return random.choice(movimientos_diagonales)

    # Si no hay movimientos diagonales disponibles, devolver cualquier movimiento posible
    return random.choice(posibles_mov_raton)

# Bucle del juego
turnos = 0
while not estado_inicial.termino() and turnos < 10:
    for evento in pygame.event.get():
        if evento.type == pygame.QUIT:
            pygame.quit()
            sys.exit()

    # Movimiento del ratón
    mejor_mov_raton = mover_raton_priorizando_alejarse(estado_inicial)
    if mejor_mov_raton is None or mejor_mov_raton == tuple(estado_inicial.gatopos):
        posibles_mov_raton = estado_inicial.posibles_movimientos(estado_inicial.ratonpos)
        posibles_mov_raton = [mov for mov in posibles_mov_raton if mov != tuple(estado_inicial.gatopos)]
        if estado_inicial.raton_pos_anterior is not None:
            posibles_mov_raton = [mov for mov in posibles_mov_raton if mov != tuple(estado_inicial.raton_pos_anterior)]
        mejor_mov_raton = random.choice(posibles_mov_raton) if posibles_mov_raton else None
    if mejor_mov_raton is not None:
        estado_inicial = estado_inicial.movimiento_raton(mejor_mov_raton)
    if estado_inicial is None or estado_inicial.termino():
        break

    dibujar_tablero(estado_inicial)
    time.sleep(1)

    # Movimiento del gato
    mejor_mov_gato = mejor_movimiento(estado_inicial, profundidad_inicial, True)
    if mejor_mov_gato is not None:
        estado_inicial = estado_inicial.movimiento_gato(mejor_mov_gato)
    if estado_inicial is None or estado_inicial.termino():
        break

    dibujar_tablero(estado_inicial)
    time.sleep(1)
    turnos += 1

# Mostrar el resultado final
if estado_inicial and estado_inicial.termino():
    ventana.blit(imagen_final_gato, (0, 0))
    pygame.display.flip()
    print("El juego ha terminado. El gato atrapó al ratón.")
elif turnos >= 7:
    ventana.blit(imagen_final_raton, (0, 0))
    pygame.display.flip()
    print("El juego ha terminado. El ratón escapó.")
else:
    print("El ratón escapó.")
time.sleep(3)
pygame.quit()
