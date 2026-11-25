
import numpy as np

from resolucao.fase2 import executarFaseDois

matriz = np.array([
    [0, 0, 2.5, 1],
    [1, 0, 1.5, 0],
    [0, 1, -0.5, 0]
])

termosIndependentes = [11, 9, 1]
objetivoOriginal = [3, 5, 0, 0]
baseInicial = [3, 0, 1]

resultado = executarFaseDois(
    matriz,
    termosIndependentes,
    objetivoOriginal,
    baseInicial,
    sentido="max",
    politica="largest"
)

print("Status:", resultado["status"])
print("Solução:", resultado["solucao"])
print("Valor ótimo:", resultado["valorOtimo"])
print("Iterações:", resultado["iteracoes"])
