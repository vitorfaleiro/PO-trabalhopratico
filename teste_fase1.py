import numpy as np

from resolucao.fase1 import executarSimplex, construirFuncaoAuxiliar, ajustarFuncaoObjetivo

from resolucao.fase1 import removerArtificiaisDaBase


from resolucao.fase1 import escolherVariavelEntrada

coeficientes = [-3, -4, 0, 1, 0, 0]

print(
    "Minimização:",
    escolherVariavelEntrada(coeficientes, "largest", "min")
)

coeficientes = [0, 0, -2, 0]

print(
    "Maximização:",
    escolherVariavelEntrada(coeficientes, "largest", "max")
)
