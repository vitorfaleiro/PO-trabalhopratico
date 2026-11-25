import argparse
import numpy as np

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
        raise ValueError("Quantidade de linhas do arquivo incompatível com o cabeçalho.")

    tiposVariaveis = lerTiposVariaveis(linhas)
    sentidoOtimizacao, coeficientesObjetivo = lerFuncaoObjetivo(linhas)

    matrizCoeficientes, operadores, termosIndependentes = lerRestricoes(
        linhas, numeroVariaveis, numeroRestricoes
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
        raise ValueError("Quantidade de coeficientes da função objetivo incorreta.")

    if problema["sentidoOtimizacao"] not in [None, "max", "min"]:
        raise ValueError("Sentido de otimização inválido.")

    if problema["matrizCoeficientes"].shape != (numeroRestricoes, numeroVariaveis):
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
    parser.add_argument("--decimals", type=int)
    parser.add_argument("--digits", type=int)
    parser.add_argument(
        "--policy",
        choices=["largest", "smallest", "bland"],
        default="largest"
    )

    return parser.parse_args()





if __name__ == "__main__":
    argumentos = lerArgumentos()

    try:
        linhas = lerArquivo(argumentos.nomeArquivo)
        problema = interpretarProblema(linhas)
        validarProblema(problema)

        print("Problema carregado e validado com sucesso!")
        print("Política:", argumentos.policy)

    except FileNotFoundError:
        print("Erro: arquivo de entrada não encontrado.")

    except ValueError as erro:
        print("Erro:", erro)



