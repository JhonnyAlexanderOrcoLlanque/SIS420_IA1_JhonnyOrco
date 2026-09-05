
import random

# ======================================================================================
# 1. GENERACIÓN DEL DATASET SINTÉTICO (1010 elementos)
# ======================================================================================
# Se define una ecuación de referencia (la "verdad" que el modelo deberá descubrir
# solo observando los datos, SIN conocerla de antemano):
#
#        y = B_REAL + W_REAL * x  +  ruido
#
# El ruido se genera con una distribución normal (random.gauss) de media 0 y una
# desviación estándar pequeña en relación a la escala de "y", para que los puntos
# se dispersen de forma realista alrededor de la recta, PERO sin exagerar
# (si el ruido fuera muy grande, la nube de puntos dejaría de comportarse como una
# línea recta y el modelo lineal ya no tendría sentido).

random.seed(42)                 # semilla fija -> resultados reproducibles

B_REAL = 3.5                    # intercepto "real" (desconocido para el modelo)
W_REAL = 2.8                    # pendiente "real"  (desconocida para el modelo)
N_TOTAL = 1010                  # tamaño total del dataset
N_TRAIN = 1000                  # elementos para entrenamiento
N_TEST = 10                     # elementos para prueba
X_MIN, X_MAX = 0.0, 50.0        # rango de la variable independiente x
RUIDO_STD = 3.0                 # desviación estándar del ruido (dispersión moderada)


def generar_dataset(n, b_real, w_real, x_min, x_max, ruido_std):
    """
    Genera 'n' puntos (x, y) siguiendo la ecuación y = b_real + w_real*x + ruido.
    """
    datos = []
    contador = 0

    # ---- Estructura REPETITIVA (while) ----
    while contador < n:                     # <-- también podría ser un for
        x = random.uniform(x_min, x_max)
        ruido = random.gauss(0, ruido_std)  # ruido gaussiano moderado
        y = b_real + w_real * x + ruido
        datos.append((x, y))
        contador += 1

    return datos


dataset_completo = generar_dataset(N_TOTAL, B_REAL, W_REAL, X_MIN, X_MAX, RUIDO_STD)

# Se mezcla el orden para que el conjunto de prueba no quede sesgado
random.shuffle(dataset_completo)

# ---- División DECISIVA usando slicing (equivalente a un if de partición) ----
dataset_entrenamiento = dataset_completo[:N_TRAIN]     # primeros 1000
dataset_prueba = dataset_completo[N_TRAIN:N_TRAIN + N_TEST]   # últimos 10

print("=" * 78)
print("DATASET SINTÉTICO GENERADO")
print("=" * 78)
print(f"Ecuación real usada para generar los datos : y = {B_REAL} + {W_REAL}*x + ruido")
print(f"Ruido gaussiano                             : media=0, desviación={RUIDO_STD}")
print(f"Total de elementos generados                : {len(dataset_completo)}")
print(f"Elementos de ENTRENAMIENTO                   : {len(dataset_entrenamiento)}")
print(f"Elementos de PRUEBA                          : {len(dataset_prueba)}")


# ======================================================================================
# 2. FUNCIÓN DE ERROR (MSE) - solo para monitorear el entrenamiento
# ======================================================================================
def calcular_mse(b, w, datos):
    suma_error = 0.0
    n = len(datos)
    for x, y_real in datos:                 # <-- estructura REPETITIVA (for)
        y_pred = b + w * x
        error = y_real - y_pred
        suma_error += error ** 2
    return suma_error / n


# ======================================================================================
# 3. ALGORITMO DE DESCENSO DE GRADIENTE
# ======================================================================================
def entrenar_descenso_gradiente(datos, tasa_aprendizaje=0.001,
                                 max_epocas=40000, tolerancia=1e-9):
    b = 0.0
    w = 0.0
    n = len(datos)

    error_anterior = None
    historial = []
    epoca = 0
    convergio = False

    # ---- Estructura REPETITIVA principal (while) ----
    while epoca < max_epocas and not convergio:

        suma_error_b = 0.0
        suma_error_w = 0.0

        # ---- Estructura REPETITIVA (for) recorriendo los 1000 datos ----
        for x, y_real in datos:
            y_pred = b + w * x
            error = y_real - y_pred
            suma_error_b += error
            suma_error_w += error * x

        gradiente_b = -(2 / n) * suma_error_b
        gradiente_w = -(2 / n) * suma_error_w

        b = b - tasa_aprendizaje * gradiente_b
        w = w - tasa_aprendizaje * gradiente_w

        error_actual = calcular_mse(b, w, datos)
        historial.append(error_actual)

        # ---- Estructura DECISIVA (if / else) para verificar convergencia ----
        if error_anterior is not None:
            cambio = abs(error_anterior - error_actual)
            if cambio < tolerancia:
                convergio = True
            else:
                convergio = False
        else:
            convergio = False

        error_anterior = error_actual
        epoca += 1

        # ---- Estructura DECISIVA: reportar avance cada 4000 épocas ----
        if epoca % 4000 == 0 or epoca == 1:
            print(f"  Época {epoca:5d}  ->  MSE = {error_actual:.5f}   "
                  f"b = {b:.5f}   w = {w:.5f}")

    return b, w, epoca, error_anterior, historial


# ======================================================================================
# 4. ENTRENAMIENTO (usando SOLO los 1000 datos de entrenamiento)
# ======================================================================================
print("\n" + "=" * 78)
print("ENTRENANDO EL MODELO CON LOS 1000 DATOS DE ENTRENAMIENTO")
print("=" * 78)

b_final, w_final, epocas_usadas, mse_train, historial = entrenar_descenso_gradiente(
    dataset_entrenamiento,
    tasa_aprendizaje=0.001,
    max_epocas=40000,
    tolerancia=1e-9
)

print("-" * 78)
print("RESULTADO DEL ENTRENAMIENTO")
print("-" * 78)
print(f"Épocas utilizadas          : {epocas_usadas}")
print(f"MSE final en entrenamiento : {mse_train:.5f}")
print(f"b encontrado (modelo)      : {b_final:.5f}   (b real = {B_REAL})")
print(f"w encontrado (modelo)      : {w_final:.5f}   (w real = {W_REAL})")
print(f"\nEcuación aprendida:  y = {b_final:.4f} + {w_final:.4f} * x")
print(f"Ecuación real usada :  y = {B_REAL} + {W_REAL} * x")


# ======================================================================================
# 5. PRUEBA / INFERENCIA CON LOS 10 DATOS RESERVADOS (nunca vistos en el entrenamiento)
# ======================================================================================
print("\n" + "=" * 78)
print("PRUEBA DEL MODELO CON LOS 10 DATOS RESERVADOS (NO usados para entrenar)")
print("=" * 78)

print(f"{'x':>8} | {'y real':>10} | {'y predicho':>12} | {'error abs.':>10}")
print("-" * 50)

suma_error_abs = 0.0
suma_error_cuadrado = 0.0
suma_y_real = 0.0

for x_test, y_real in dataset_prueba:        # <-- estructura REPETITIVA (for)
    y_pred = b_final + w_final * x_test
    error_abs = abs(y_real - y_pred)

    suma_error_abs += error_abs
    suma_error_cuadrado += (y_real - y_pred) ** 2
    suma_y_real += y_real

    print(f"{x_test:8.3f} | {y_real:10.3f} | {y_pred:12.3f} | {error_abs:10.3f}")

# ---- Métricas de desempeño sobre el conjunto de prueba ----
n_test = len(dataset_prueba)
mae_test = suma_error_abs / n_test
mse_test = suma_error_cuadrado / n_test
rmse_test = mse_test ** 0.5
y_promedio = suma_y_real / n_test

# R^2 (coeficiente de determinación) calculado manualmente
suma_total = 0.0
for x_test, y_real in dataset_prueba:        # <-- estructura REPETITIVA (for)
    suma_total += (y_real - y_promedio) ** 2

if suma_total > 0:
    r2 = 1 - (suma_error_cuadrado / suma_total)
else:
    r2 = 0.0

print("-" * 50)
print(f"MAE  (Error Absoluto Medio)         : {mae_test:.4f}")
print(f"MSE  (Error Cuadrático Medio)        : {mse_test:.4f}")
print(f"RMSE (Raíz del Error Cuadrático Medio): {rmse_test:.4f}")
print(f"R²   (Coeficiente de determinación)  : {r2:.4f}")
print("=" * 78)