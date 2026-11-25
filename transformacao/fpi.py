
import numpy as np


def transformarVariaveis(problema):
    matrizOriginal = problema["matrizCoeficientes"]
    objetivoOriginal = problema["coeficientesObjetivo"]
    tiposVariaveis = problema["tiposVariaveis"]

    novasColunas = []
    novoObjetivo = []
    mapeamentoVariaveis = []

    for indice, tipo in enumerate(tiposVariaveis):
        coeficientesColuna = matrizOriginal[:, indice]
        coeficienteObjetivo = objetivoOriginal[indice]

        if tipo == 1:
            novasColunas.append(coeficientesColuna)
            novoObjetivo.append(coeficienteObjetivo)
            mapeamentoVariaveis.append([(len(novoObjetivo) - 1, 1)])

        elif tipo == -1:
            novasColunas.append(-coeficientesColuna)
            novoObjetivo.append(-coeficienteObjetivo)
            mapeamentoVariaveis.append([(len(novoObjetivo) - 1, -1)])

        elif tipo == 0:
            indicePositivo = len(novoObjetivo)

            novasColunas.append(coeficientesColuna)
            novoObjetivo.append(coeficienteObjetivo)

            indiceNegativo = len(novoObjetivo)

            novasColunas.append(-coeficientesColuna)
            novoObjetivo.append(-coeficienteObjetivo)

            mapeamentoVariaveis.append([
                (indicePositivo, 1),
                (indiceNegativo, -1)
            ])

    novaMatriz = np.column_stack(novasColunas)

    return novaMatriz, novoObjetivo, mapeamentoVariaveis





def normalizarRestricoes(
    matriz,
    operadores,
    termosIndependentes,
    retornarSinais=False
):
    matriz = np.array(matriz, dtype=float)
    operadores = operadores.copy()
    termosIndependentes = termosIndependentes.copy()

    sinaisNormalizacao = []

    for indice in range(len(termosIndependentes)):
        sinal = 1

        if termosIndependentes[indice] < 0:
            matriz[indice] *= -1
            termosIndependentes[indice] *= -1
            sinal = -1

            if operadores[indice] == "<=":
                operadores[indice] = ">="

            elif operadores[indice] == ">=":
                operadores[indice] = "<="

        sinaisNormalizacao.append(sinal)

    if retornarSinais:
        return matriz, operadores, termosIndependentes, sinaisNormalizacao

    return matriz, operadores, termosIndependentes



def adicionarVariaveisFolga(matriz, operadores, termosIndependentes, objetivo):
    matriz = np.array(matriz, dtype=float)
    novaObjetivo = list(objetivo)

    novasColunas = []
    baseInicial = []
    indicesArtificiais = []
    nomesVariaveis = []

    numeroRestricoes = len(operadores)
    numeroVariaveis = matriz.shape[1]

    for indice, operador in enumerate(operadores):
        coluna = np.zeros(numeroRestricoes)

        if operador == "<=":
            coluna[indice] = 1

            novasColunas.append(coluna)
            novaObjetivo.append(0)

            baseInicial.append(numeroVariaveis + len(novasColunas) - 1)
            nomesVariaveis.append(f"s{indice + 1}")

        elif operador == ">=":
            coluna[indice] = -1

            novasColunas.append(coluna)
            novaObjetivo.append(0)
            nomesVariaveis.append(f"e{indice + 1}")

            colunaArtificial = np.zeros(numeroRestricoes)
            colunaArtificial[indice] = 1

            novasColunas.append(colunaArtificial)
            novaObjetivo.append(0)

            indiceArtificial = numeroVariaveis + len(novasColunas) - 1

            baseInicial.append(indiceArtificial)
            indicesArtificiais.append(indiceArtificial)
            nomesVariaveis.append(f"a{indice + 1}")

        elif operador == "==":
            colunaArtificial = np.zeros(numeroRestricoes)
            colunaArtificial[indice] = 1

            novasColunas.append(colunaArtificial)
            novaObjetivo.append(0)

            indiceArtificial = numeroVariaveis + len(novasColunas) - 1

            baseInicial.append(indiceArtificial)
            indicesArtificiais.append(indiceArtificial)
            nomesVariaveis.append(f"a{indice + 1}")

    if novasColunas:
        matriz = np.column_stack([matriz] + novasColunas)

    return (
        matriz,
        novaObjetivo,
        baseInicial,
        indicesArtificiais,
        nomesVariaveis
    )
