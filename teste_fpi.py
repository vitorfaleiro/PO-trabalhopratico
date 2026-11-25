
import numpy as np
from transformacao.fpi import normalizarRestricoes


from transformacao.fpi import adicionarVariaveisFolga


matriz = np.array([
    [1, 2],
    [-3, 4],
    [2, 1]
])

operadores = ["<=", ">=", "<="]
termos = [10, -12, 5]

(
    novaMatriz,
    novosOperadores,
    novosTermos,
    sinais
) = normalizarRestricoes(
    matriz,
    operadores,
    termos,
    retornarSinais=True
)

print("Matriz:")
print(novaMatriz)

print("Operadores:", novosOperadores)
print("Termos:", novosTermos)
print("Sinais:", sinais)
