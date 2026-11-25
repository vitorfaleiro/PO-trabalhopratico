
import argparse
import numpy as np

from transformacao.fpi import (
    transformarVariaveis,
    normalizarRestricoes,
    adicionarVariaveisFolga
)

from resolucao.fase1 import executarFaseUm
from resolucao.fase2 import executarFaseDois


def lerArquivo(nomeArquivo):
    with open(nomeArquivo, "r", encoding="utf-8") as arquivo:
        linhas = [linha.strip() for linha in arquivo if linha.strip()]

    return linhas


def lerCabecalho(linhas):
    numeroVariaveis = int(linhas[0])
    numeroRestricoes = int(linhas[1])

    return numeroVariaveis, numeroRestricoes


def lerTiposVariaveis(linhas):
    tiposVariaveis = [int(valor) for valor in linhas[2].split()]

    return tiposVariaveis


def lerFuncaoObjetivo(linhas):
    dados = linhas[3].split()

    if dados[0].lower() in ["max", "min"]:
        sentidoOtimizacao = dados[0].lower()
        coeficientesObjetivo = [float(valor) for valor in dados[1:]]
    else:
        sentidoOtimizacao = None
        coeficientesObjetivo = [float(valor) for valor in dados]

    return sentidoOtimizacao, coeficientesObjetivo


def lerRestricoes(linhas, numeroVariaveis, numeroRestricoes):
    matrizCoeficientes = []
    operadores = []
    termosIndependentes = []

    for linha in linhas[4:4 + numeroRestricoes]:
        dados = linha.split()

        coeficientes = [float(valor) for valor in dados[:numeroVariaveis]]
        operador = dados[numeroVariaveis]
        termoIndependente = float(dados[numeroVariaveis + 1])

        matrizCoeficientes.append(coeficientes)
        operadores.append(operador)
        termosIndependentes.append(termoIndependente)

    matrizCoeficientes = np.array(matrizCoeficientes).reshape(
        numeroRestricoes, numeroVariaveis
    )

    return matrizCoeficientes, operadores, termosIndependentes


def interpretarProblema(linhas):
    numeroVariaveis, numeroRestricoes = lerCabecalho(linhas)

    if len(linhas) != 4 + numeroRestricoes:
        raise ValueError(
            "Quantidade de linhas do arquivo incompatível com o cabeçalho."
        )

    tiposVariaveis = lerTiposVariaveis(linhas)

    sentidoOtimizacao, coeficientesObjetivo = lerFuncaoObjetivo(linhas)

    matrizCoeficientes, operadores, termosIndependentes = lerRestricoes(
        linhas,
        numeroVariaveis,
        numeroRestricoes
    )

    problema = {
        "numeroVariaveis": numeroVariaveis,
        "numeroRestricoes": numeroRestricoes,
        "tiposVariaveis": tiposVariaveis,
        "sentidoOtimizacao": sentidoOtimizacao,
        "coeficientesObjetivo": coeficientesObjetivo,
        "matrizCoeficientes": matrizCoeficientes,
        "operadores": operadores,
        "termosIndependentes": termosIndependentes
    }

    return problema


def validarProblema(problema):
    numeroVariaveis = problema["numeroVariaveis"]
    numeroRestricoes = problema["numeroRestricoes"]

    if numeroVariaveis <= 0 or numeroRestricoes < 0:
        raise ValueError("Quantidade de variáveis ou restrições inválida.")

    if len(problema["tiposVariaveis"]) != numeroVariaveis:
        raise ValueError("Quantidade de tipos de variáveis incorreta.")

    if any(tipo not in [-1, 0, 1] for tipo in problema["tiposVariaveis"]):
        raise ValueError("Tipo de variável inválido.")

    if len(problema["coeficientesObjetivo"]) != numeroVariaveis:
        raise ValueError(
            "Quantidade de coeficientes da função objetivo incorreta."
        )

    if problema["sentidoOtimizacao"] not in [None, "max", "min"]:
        raise ValueError("Sentido de otimização inválido.")

    if problema["matrizCoeficientes"].shape != (
        numeroRestricoes,
        numeroVariaveis
    ):
        raise ValueError("Dimensões da matriz de restrições incorretas.")

    if len(problema["operadores"]) != numeroRestricoes:
        raise ValueError("Quantidade de operadores incorreta.")

    if any(op not in ["<=", ">=", "=="] for op in problema["operadores"]):
        raise ValueError("Operador de restrição inválido.")

    if len(problema["termosIndependentes"]) != numeroRestricoes:
        raise ValueError("Quantidade de termos independentes incorreta.")

    return True


def lerArgumentos():
    parser = argparse.ArgumentParser()

    parser.add_argument("nomeArquivo", help="Arquivo de entrada da PL")
    parser.add_argument("--decimals", type=int, default=7)
    parser.add_argument("--digits", type=int)
    parser.add_argument(
        "--policy",
        choices=["largest", "smallest", "bland"],
        default="largest"
    )

    return parser.parse_args()


def resolverProblema(problema, politica="largest"):
    sentido = problema["sentidoOtimizacao"]

    if sentido is None:
        sentido = "max"

    # Transformação das variáveis
    print("\n[1/5] Transformando variáveis...")

    matriz, objetivo, mapeamento = transformarVariaveis(problema)

    # Normalização das restrições
    print("[2/5] Normalizando restrições...")

    (
        matriz,
        operadores,
        termos,
        sinaisNormalizacao
    ) = normalizarRestricoes(
        matriz,
        problema["operadores"],
        problema["termosIndependentes"],
        retornarSinais=True
    )

    # Adição de variáveis auxiliares
    print("[3/5] Adicionando variáveis auxiliares...")

    (
        matriz,
        objetivo,
        base,
        artificiais,
        nomesVariaveis
    ) = adicionarVariaveisFolga(
        matriz,
        operadores,
        termos,
        objetivo
    )

    # Preserva a matriz sem as colunas artificiais
    matrizBaseOriginal = np.delete(
        matriz,
        sorted(artificiais),
        axis=1
    )

    # Fase I
    print("[4/5] Executando Fase I...")

    resultadoFaseUm = executarFaseUm(
        matriz,
        termos,
        objetivo,
        base,
        artificiais,
        politica=politica
    )

    print(
        "    Iterações realizadas:",
        len(resultadoFaseUm["iteracoes"])
    )

    if resultadoFaseUm["status"] != "viavel":
        return resultadoFaseUm

    # Remove da matriz e dos sinais as restrições redundantes
    linhasRemovidas = resultadoFaseUm["linhasRemovidas"]

    if linhasRemovidas:
        matrizBaseOriginal = np.delete(
            matrizBaseOriginal,
            linhasRemovidas,
            axis=0
        )

    sinaisFinais = [
        sinal
        for indice, sinal in enumerate(sinaisNormalizacao)
        if indice not in linhasRemovidas
    ]

    # Fase II
    print("[5/5] Executando Fase II...")

    resultadoFaseDois = executarFaseDois(
        resultadoFaseUm["matriz"],
        resultadoFaseUm["termosIndependentes"],
        resultadoFaseUm["objetivoOriginal"],
        resultadoFaseUm["base"],
        sentido,
        politica=politica,
        matrizBaseOriginal=matrizBaseOriginal,
        sinaisNormalizacao=sinaisFinais
    )

    print(
        "    Iterações realizadas:",
        len(resultadoFaseDois["iteracoes"])
    )

    if resultadoFaseDois["status"] != "otimo":
        return resultadoFaseDois

    # Reconstrução das variáveis originais
    solucaoOriginal = []

    for mapeamentoVariavel in mapeamento:
        valor = sum(
            sinal * resultadoFaseDois["solucao"][indice]
            for indice, sinal in mapeamentoVariavel
        )

        solucaoOriginal.append(valor)

    resultadoFaseDois["solucaoOriginal"] = solucaoOriginal
    resultadoFaseDois["sentidoOtimizacao"] = sentido

    resultadoFaseDois["iteracoesFaseUm"] = resultadoFaseUm["iteracoes"]
    resultadoFaseDois["iteracoesFaseDois"] = resultadoFaseDois["iteracoes"]

    return resultadoFaseDois


def formatarNumero(valor, decimais, digitos=None):
    epsilon = 10 ** (-(decimais + 1))

    if abs(valor) < epsilon:
        valor = 0.0

    if digitos is None:
        return f"{valor:.{decimais}f}"

    return f"{valor:{digitos}.{decimais}f}"


def imprimirResultado(resultado, decimais=7, digitos=None):
    print("\n========================================")

    status = resultado["status"]

    if status == "inviavel":
        print("Status: inviavel")

    elif status == "ilimitado":
        print("Status: ilimitado")

    elif status == "limite_iteracoes":
        print("Status: limite de iteracoes")

    elif status == "otimo":
        print("Status: otimo")

        print(
            "Objetivo:",
            formatarNumero(
                resultado["valorOtimo"],
                decimais,
                digitos
            )
        )

        print("Solucao:")

        valoresSolucao = [
            formatarNumero(valor, decimais, digitos)
            for valor in resultado["solucaoOriginal"]
        ]

        print(" ".join(valoresSolucao))

        if "solucaoDual" in resultado:
            print("Dual:")

            valoresDuais = [
                formatarNumero(valor, decimais, digitos)
                for valor in resultado["solucaoDual"]
            ]

            print(" ".join(valoresDuais))

    else:
        print("Status:", status)

    print("========================================")


if __name__ == "__main__":
    argumentos = lerArgumentos()

    try:
        linhas = lerArquivo(argumentos.nomeArquivo)

        problema = interpretarProblema(linhas)

        validarProblema(problema)

        print("Problema carregado e validado.")
        print("Política:", argumentos.policy)

        resultado = resolverProblema(
            problema,
            politica=argumentos.policy
        )

        imprimirResultado(
            resultado,
            argumentos.decimals,
            argumentos.digits
        )

    except FileNotFoundError:
        print("Erro: arquivo de entrada não encontrado.")

    except ValueError as erro:
        print("Erro:", erro)

    except np.linalg.LinAlgError:
        print("Erro: não foi possível calcular a solução dual.")

