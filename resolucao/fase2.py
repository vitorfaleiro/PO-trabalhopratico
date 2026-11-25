
import numpy as np

from resolucao.fase1 import (
    ajustarFuncaoObjetivo,
    executarSimplex
)


def calcularSolucaoDual(
    matrizOriginal,
    objetivoOriginal,
    baseInicial,
    sinaisNormalizacao
):
    matrizOriginal = np.array(matrizOriginal, dtype=float)
    objetivoOriginal = np.array(objetivoOriginal, dtype=float)

    matrizBasica = matrizOriginal[:, baseInicial]
    coeficientesBasicos = objetivoOriginal[baseInicial]

    solucaoDual = np.linalg.solve(
        matrizBasica.T,
        coeficientesBasicos
    )

    sinaisNormalizacao = np.array(sinaisNormalizacao, dtype=float)

    solucaoDual *= sinaisNormalizacao

    return solucaoDual


def executarFaseDois(
    matriz,
    termosIndependentes,
    objetivoOriginal,
    baseInicial,
    sentido,
    politica="largest",
    maxIteracoes=1000,
    matrizBaseOriginal=None,
    sinaisNormalizacao=None
):
    matriz = np.array(matriz, dtype=float)
    termosIndependentes = np.array(termosIndependentes, dtype=float)
    objetivoOriginal = np.array(objetivoOriginal, dtype=float)

    funcaoAjustada, termoIndependente = ajustarFuncaoObjetivo(
        matriz,
        termosIndependentes,
        objetivoOriginal,
        baseInicial
    )

    resultado = executarSimplex(
        matriz,
        termosIndependentes,
        funcaoAjustada,
        termoIndependente,
        baseInicial,
        politica,
        maxIteracoes,
        sentido=sentido
    )

    resultado["funcaoObjetivoAjustada"] = funcaoAjustada

    
    if resultado["status"] == "otimo":
        resultado["solucao"] = np.zeros(matriz.shape[1])

        for indice, variavel in enumerate(resultado["base"]):
            resultado["solucao"][variavel] = (
                resultado["termosIndependentes"][indice]
            )

        resultado["valorOtimo"] = resultado["termoObjetivo"]

        if matrizBaseOriginal is not None:
            resultado["solucaoDual"] = calcularSolucaoDual(
                matrizBaseOriginal,
                objetivoOriginal,
                resultado["base"],
                sinaisNormalizacao
            )

    return resultado

