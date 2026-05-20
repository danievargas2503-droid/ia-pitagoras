import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# Datos de entrenamiento
# [cateto1, cateto2]
catetos = np.array([
  
    [3,4],
    [5,12],
    [8,15],
    [7,24],
    [9,40],
    [12,35],
    [20,21],
    [6,8],
    [10,24],
    [15,20],
    [9,12],
    [16,30],
    [18,24],
    [21,28],
    [24,32],
    [12,16],
    [30,40],
    [14,48],
    [11,60],
    [28,45],
    [33,56],
    [40,42],
    [48,55],
    [65,72],
    [7,9],
    [13,14],
    [17,19],
    [22,27],
    [31,33],
    [45,60],
    [50,70],
    [80,90],
    [100,120],
    [25,35],
    [44,55],
    [60,80],
    [75,100],
    [90,120],
    [8,6],
    [4,3],
    [24,7],
    [40,9],
     
 

], dtype=float)

# Hipotenusas correspondientes
hipotenusas = np.array([
  5,
    13,
    17,
    25,
    41,
    37,
    29,
    10,
    26,
    25,
    15,
    34,
    30,
    35,
    40,
    20,
    50,
    50,
    61,
    53,
    65,
    58,
    73,
    97,
    11.40,
    19.10,
    25.50,
    34.83,
    45.28,
    75,
    86.02,
    120.41,
    156.20,
    43.01,
    70.43,
    100,
    125,
    150,
    10,
    5,
    25,
    41,
  
], dtype=float)

# Crear modelo
modelo = tf.keras.Sequential([
    tf.keras.layers.Dense(units=32, input_shape=[2], activation='relu'),
    tf.keras.layers.Dense(units=32, activation='relu'),
    tf.keras.layers.Dense(units=1)
])

# Compilar modelo
modelo.compile(
    optimizer=tf.keras.optimizers.Adam(0.1),
    loss='mean_squared_error'
)

# Entrenamiento
print("Entrenando el modelo...")
historial = modelo.fit(catetos, hipotenusas, epochs=500, verbose=False)
print("Modelo entrenado!")


modelo.save("modelo.h5")

print("Modelo guardado!")