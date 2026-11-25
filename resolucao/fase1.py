
def construirFuncaoAuxiliar(numeroVariaveis, indicesArtificiais):
    funcaoAuxiliar = [0] * numeroVariaveis

    for indice in indicesArtificiais:
        funcaoAuxiliar[indice] = 1

    return funcaoAuxiliar


import numpy as np


def ajustarFuncaoObjetivo(matriz, termosIndependentes, funcaoAuxiliar, baseInicial):
    matriz = np.array(matriz, dtype=float)
    termosIndependentes = np.array(termosIndependentes, dtype=float)
    funcaoAuxiliar = np.array(funcaoAuxiliar, dtype=float)

    coeficientesBasicos = funcaoAuxiliar[baseInicial]

    funcaoAjustada = funcaoAuxiliar - coeficientesBasicos @ matriz
    termoIndependente = coeficientesBasicos @ termosIndependentes

    return funcaoAjustada, termoIndependente


def escolherVariavelSaida(
    matriz,
    termosIndependentes,
    indiceEntrada,
    baseInicial,
    epsilon=1e-10
):
    razoes = []

    for indice in range(len(termosIndependentes)):
        coeficiente = matriz[indice, indiceEntrada]

        if coeficiente > epsilon:
            razao = termosIndependentes[indice] / coeficiente

            if razao >= -epsilon:
                razoes.append((razao, baseInicial[indice], indice))

    if not razoes:
        return None

    menorRazao = min(razao[0] for razao in razoes)

    empatados = [
        razao for razao in razoes
        if abs(razao[0] - menorRazao) <= epsilon
    ]

    return min(empatados, key=lambda razao: razao[1])[2]


def realizarPivoteamento(
    matriz,
    termosIndependentes,
    funcaoObjetivo,
    termoObjetivo,
    linhaPivo,
    colunaPivo
):
    matriz = np.array(matriz, dtype=float)
    termosIndependentes = np.array(termosIndependentes, dtype=float)
    funcaoObjetivo = np.array(funcaoObjetivo, dtype=float)

    pivo = matriz[linhaPivo, colunaPivo]

    if abs(pivo) < 1e-10:
        raise ValueError("O elemento pivô não pode ser zero.")

    matriz[linhaPivo] /= pivo
    termosIndependentes[linhaPivo] /= pivo

    for indice in range(len(termosIndependentes)):
        if indice != linhaPivo:
            fator = matriz[indice, colunaPivo]

            matriz[indice] -= fator * matriz[linhaPivo]
            termosIndependentes[indice] -= fator * termosIndependentes[linhaPivo]

    fatorObjetivo = funcaoObjetivo[colunaPivo]

    funcaoObjetivo -= fatorObjetivo * matriz[linhaPivo]
    termoObjetivo += fatorObjetivo * termosIndependentes[linhaPivo]

    return (
        matriz,
        termosIndependentes,
        funcaoObjetivo,
        termoObjetivo
    )



def atualizarBase(baseInicial, linhaPivo, colunaPivo):
    novaBase = baseInicial.copy()
    novaBase[linhaPivo] = colunaPivo

    return novaBase



def escolherVariavelEntrada(
    funcaoObjetivo,
    politica,
    sentido="min",
    epsilon=1e-10
):
    if sentido == "min":
        candidatos = [
            indice
            for indice, coeficiente in enumerate(funcaoObjetivo)
            if coeficiente < -epsilon
        ]

    elif sentido == "max":
        candidatos = [
            indice
            for indice, coeficiente in enumerate(funcaoObjetivo)
            if coeficiente > epsilon
        ]

    else:
        raise ValueError("Sentido de otimização inválido.")

    if not candidatos:
        return None

    if politica == "bland":
        return min(candidatos)

    if politica == "largest":
        return max(
            candidatos,
            key=lambda indice: (funcaoObjetivo[indice], -indice)
        )

    if politica == "smallest":
        return min(
            candidatos,
            key=lambda indice: (funcaoObjetivo[indice], indice)
        )

    raise ValueError("Política de seleção inválida.")



def executarSimplex(
    matriz,
    termosIndependentes,
    funcaoObjetivo,
    termoObjetivo,
    baseInicial,
    politica="largest",
    maxIteracoes=1000,
    epsilon=1e-10,
    sentido="min"
):
    matriz = np.array(matriz, dtype=float)
    termosIndependentes = np.array(termosIndependentes, dtype=float)
    funcaoObjetivo = np.array(funcaoObjetivo, dtype=float)
    baseInicial = baseInicial.copy()

    iteracoes = []

    if politica not in ["largest", "smallest", "bland"]:
        raise ValueError("Política de seleção inválida.")

    for numeroIteracao in range(maxIteracoes):

        indiceEntrada = escolherVariavelEntrada(
            funcaoObjetivo,
            politica,
            sentido,
            epsilon
        )

        if indiceEntrada is None:
            return {
                "status": "otimo",
                "matriz": matriz,
                "termosIndependentes": termosIndependentes,
                "funcaoObjetivo": funcaoObjetivo,
                "termoObjetivo": termoObjetivo,
                "base": baseInicial,
                "iteracoes": iteracoes
            }

        indiceSaida = escolherVariavelSaida(
            matriz,
            termosIndependentes,
            indiceEntrada,
            baseInicial,
            epsilon
        )

        if indiceSaida is None:
            return {
                "status": "ilimitado",
                "matriz": matriz,
                "termosIndependentes": termosIndependentes,
                "funcaoObjetivo": funcaoObjetivo,
                "termoObjetivo": termoObjetivo,
                "base": baseInicial,
                "iteracoes": iteracoes
            }

        indiceVariavelSaida = baseInicial[indiceSaida]
        elementoPivo = matriz[indiceSaida, indiceEntrada]

        (
            matriz,
            termosIndependentes,
            funcaoObjetivo,
            termoObjetivo
        ) = realizarPivoteamento(
            matriz,
            termosIndependentes,
            funcaoObjetivo,
            termoObjetivo,
            indiceSaida,
            indiceEntrada
        )

        baseInicial = atualizarBase(
            baseInicial,
            indiceSaida,
            indiceEntrada
        )

        iteracoes.append({
            "numero": numeroIteracao + 1,
            "variavelEntrada": indiceEntrada,
            "variavelSaida": indiceVariavelSaida,
            "linhaSaida": indiceSaida,
            "elementoPivo": elementoPivo,
            "valorObjetivo": termoObjetivo
        })

    return {
        "status": "limite_iteracoes",
        "matriz": matriz,
        "termosIndependentes": termosIndependentes,
        "funcaoObjetivo": funcaoObjetivo,
        "termoObjetivo": termoObjetivo,
        "base": baseInicial,
        "iteracoes": iteracoes
    }



def removerArtificiaisDaBase(
    matriz,
    termosIndependentes,
    baseInicial,
    indicesArtificiais,
    epsilon=1e-10
):
    matriz = np.array(matriz, dtype=float)
    termosIndependentes = np.array(termosIndependentes, dtype=float)
    baseInicial = baseInicial.copy()

    artificiais = set(indicesArtificiais)
    linhasRemovidas = []

    indiceLinha = 0

    while indiceLinha < len(baseInicial):

        if baseInicial[indiceLinha] not in artificiais:
            indiceLinha += 1
            continue

        if abs(termosIndependentes[indiceLinha]) > epsilon:
            raise ValueError(
                "Variável artificial básica com valor diferente de zero."
            )

        variaveisBasicas = set(baseInicial)

        candidatos = [
            indice
            for indice in range(matriz.shape[1])
            if indice not in artificiais
            and indice not in variaveisBasicas
            and abs(matriz[indiceLinha, indice]) > epsilon
        ]

        if candidatos:
            indiceEntrada = min(candidatos)
            pivo = matriz[indiceLinha, indiceEntrada]

            matriz[indiceLinha] /= pivo
            termosIndependentes[indiceLinha] /= pivo

            for indice in range(len(termosIndependentes)):
                if indice != indiceLinha:
                    fator = matriz[indice, indiceEntrada]
                    matriz[indice] -= fator * matriz[indiceLinha]
                    termosIndependentes[indice] -= (
                        fator * termosIndependentes[indiceLinha]
                    )

            baseInicial[indiceLinha] = indiceEntrada
            indiceLinha += 1

        else:
            matriz = np.delete(matriz, indiceLinha, axis=0)
            termosIndependentes = np.delete(
                termosIndependentes, indiceLinha
            )
            baseInicial.pop(indiceLinha)
            linhasRemovidas.append(indiceLinha)

    matriz = np.delete(
        matriz, sorted(artificiais), axis=1
    )

    novaBase = [
        indice - sum(artificial < indice for artificial in artificiais)
        for indice in baseInicial
    ]

    return matriz, termosIndependentes, novaBase, linhasRemovidas



def executarFaseUm(
    matriz,
    termosIndependentes,
    objetivoOriginal,
    baseInicial,
    indicesArtificiais,
    politica="largest",
    maxIteracoes=1000,
    epsilon=1e-10
):
    matriz = np.array(matriz, dtype=float)
    termosIndependentes = np.array(termosIndependentes, dtype=float)
    objetivoOriginal = np.array(objetivoOriginal, dtype=float)

    numeroVariaveis = matriz.shape[1]

    funcaoAuxiliar = construirFuncaoAuxiliar(
        numeroVariaveis,
        indicesArtificiais
    )

    funcaoAjustada, termoIndependente = ajustarFuncaoObjetivo(
        matriz,
        termosIndependentes,
        funcaoAuxiliar,
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
        epsilon
    )

    if resultado["status"] != "otimo":
        resultado["objetivoOriginal"] = objetivoOriginal
        return resultado

    valorW = resultado["termoObjetivo"]

    if valorW > epsilon:
        resultado["status"] = "inviavel"
        resultado["objetivoOriginal"] = objetivoOriginal
        return resultado

    if valorW < -epsilon:
        raise ValueError("Valor negativo inesperado na função auxiliar.")

    (
        matrizFinal,
        termosFinais,
        baseFinal,
        linhasRemovidas
    ) = removerArtificiaisDaBase(
        resultado["matriz"],
        resultado["termosIndependentes"],
        resultado["base"],
        indicesArtificiais,
        epsilon
    )

    objetivoFinal = np.delete(
        objetivoOriginal,
        sorted(indicesArtificiais)
    )

    resultado["status"] = "viavel"
    resultado["matriz"] = matrizFinal
    resultado["termosIndependentes"] = termosFinais
    resultado["base"] = baseFinal
    resultado["objetivoOriginal"] = objetivoFinal
    resultado["linhasRemovidas"] = linhasRemovidas
    resultado["valorW"] = 0.0

    return resultado
