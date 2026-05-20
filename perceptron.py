def calcular_ativacao(x, w):
    """Calcula a soma ponderada dos atributos pelos pesos."""
    if len(x) != len(w):
        raise ValueError(
            f"O exemplo tem {len(x)} atributo(s), mas existem {len(w)} peso(s). "
            "Use a mesma quantidade de atributos e pesos."
        )

    return sum(x_j * w_j for x_j, w_j in zip(x, w))


def calcular_saida(ativacao):
    """Aplica a funcao degrau: h(x) = 1 se ativacao > 0, senao 0."""
    return 1 if ativacao > 0 else 0


def treinar_perceptron(X, Y, w, alpha, ciclos=1):
    """
    Treina o perceptron usando a regra:
    theta_j = theta_j + alpha * E(i) * x_j(i)

    X: lista de exemplos, em que cada exemplo e um vetor de atributos
    Y: lista com as saidas esperadas
    w: vetor de pesos iniciais
    alpha: taxa de aprendizagem
    ciclos: quantidade de ciclos de treinamento
    """
    if len(X) != len(Y):
        raise ValueError("X e Y precisam ter a mesma quantidade de exemplos.")

    pesos = w.copy()
    historico_log = []

    for ciclo in range(ciclos):
        historico_log.append(f"### Ciclo {ciclo + 1}")

        for i, (x_i, y_i) in enumerate(zip(X, Y), start=1):
            ativacao = calcular_ativacao(x_i, pesos)
            h_x = calcular_saida(ativacao)
            erro = y_i - h_x

            historico_log.append(
                f"Exemplo {i}: x = {x_i} | y = {y_i} | ativacao = {ativacao:.4f} | h(x) = {h_x} | erro = {erro}"
            )

            if erro == 0:
                historico_log.append("Erro 0: os pesos nao foram atualizados.")
                historico_log.append("")
                continue

            pesos_anteriores = pesos.copy()

            for j in range(len(pesos)):
                pesos[j] = pesos[j] + (alpha * erro * x_i[j])

            historico_log.append(f"Pesos anteriores: {formatar_vetor(pesos_anteriores)}")
            historico_log.append(
                f"Atualizacao: w_j = w_j + {alpha} * {erro} * x_j"
            )
            historico_log.append(f"Pesos atualizados: {formatar_vetor(pesos)}")
            historico_log.append("")

    return pesos, historico_log


def formatar_vetor(vetor):
    return "[" + ", ".join(f"{valor:.4f}" for valor in vetor) + "]"


def salvar_log(nome_arquivo, log, pesos_finais):
    with open(nome_arquivo, "w", encoding="utf-8") as arquivo:
        arquivo.write("# Atualizacoes de pesos do perceptron\n\n")
        arquivo.write("Codificacao: Nao = 0, Sim = 1\n")
        arquivo.write("Atributos: [bias, estudou, fez_trabalho]\n\n")
        arquivo.write("Funcao de ativacao: se ativacao > 0, h(x) = 1; se ativacao <= 0, h(x) = 0\n\n")
        arquivo.write("\n".join(log))
        arquivo.write(f"\nPesos finais: {formatar_vetor(pesos_finais)}\n")


# ==========================================
# TESTE DO EXERCICIO
# ==========================================
# Codificacao:
# Nao = 0
# Sim = 1
#
# Atributos:
# [bias, estudou, fez_trabalho]
X_treino = [
    [1, 0, 0],  # Joaozinho: nao estudou, nao fez o trabalho
    [1, 0, 1],  # Huguinho: nao estudou, fez o trabalho
    [1, 1, 0],  # Zezinho: estudou, nao fez o trabalho
    [1, 1, 1],  # Luizinho: estudou, fez o trabalho
]
Y_treino = [0, 0, 1, 1]

pesos_iniciais = [0.0, 0.0, 0.0]
taxa_aprendizagem = 0.1

pesos_finais, log = treinar_perceptron(
    X_treino,
    Y_treino,
    pesos_iniciais,
    taxa_aprendizagem,
    ciclos=2,
)

for linha in log:
    print(linha)

print(f"Pesos finais: {formatar_vetor(pesos_finais)}")
salvar_log("atualizacoes_pesos.txt", log, pesos_finais)