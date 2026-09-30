"""
ANÁLISE E DECISÃO/AUTOMAÇÃO: indicadores, padrões de consumo e alertas.
Tudo aqui é calculado a partir das movimentações cadastradas — nada é inventado.
"""
import pandas as pd

from src.classificacao import normalizar
from src.dados import formatar_reais

# Limites usados pelos alertas (fáceis de explicar e de ajustar)
PERCENTUAL_SALARIO_ALERTA = 0.80   # usou 80% do salário
SALDO_BAIXO = 0.10                 # saldo menor que 10% das receitas
AUMENTO_RELEVANTE = 0.15           # categoria cresceu 15% ou mais
VARIACAO_RECORRENTE = 0.05         # valor quase igual todo mês (±5%)
MIN_TRANSACOES_ATIPICO = 5         # precisa de histórico para achar exceções

ORCAMENTO_PADRAO = {
    "Alimentação": 800.0, "Transporte": 400.0, "Moradia": 1200.0, "Saúde": 200.0,
    "Educação": 350.0, "Lazer": 300.0, "Compras": 500.0, "Assinaturas": 150.0,
    "Contas": 450.0, "Outros": 200.0,
}


def preparar(df: pd.DataFrame) -> pd.DataFrame:
    """Garante tipos corretos e cria a coluna 'mes' (ex.: '2026-09')."""
    df = df.copy()
    df["data"] = pd.to_datetime(df["data"])
    df["valor"] = df["valor"].astype(float)
    df["mes"] = df["data"].dt.strftime("%Y-%m")
    return df


def meses_disponiveis(df: pd.DataFrame) -> list[str]:
    return sorted(preparar(df)["mes"].unique()) if len(df) else []


def mes_anterior(meses: list[str], mes: str) -> str | None:
    anteriores = [m for m in meses if m < mes]
    return anteriores[-1] if anteriores else None


def despesas(df: pd.DataFrame) -> pd.DataFrame:
    return df[df["tipo"] == "Despesa"]


# ---------------- INDICADORES DO DASHBOARD ----------------
def indicadores(df_mes: pd.DataFrame, salario: float) -> dict:
    receitas = df_mes.loc[df_mes["tipo"] == "Receita", "valor"].sum()
    gastos = despesas(df_mes)
    total_despesas = gastos["valor"].sum()
    por_categoria = gastos.groupby("categoria")["valor"].sum()

    return {
        "salario": salario,
        "receitas": receitas,
        "despesas": total_despesas,
        "saldo": receitas - total_despesas,
        "pct_salario": total_despesas / salario if salario > 0 else 0,
        "maior_categoria": por_categoria.idxmax() if len(por_categoria) else "-",
        "maior_categoria_valor": por_categoria.max() if len(por_categoria) else 0,
        "qtd_transacoes": len(df_mes),
    }


def gastos_por_categoria(df_mes: pd.DataFrame) -> pd.DataFrame:
    tabela = despesas(df_mes).groupby("categoria", as_index=False)["valor"].sum()
    total = tabela["valor"].sum()
    tabela["percentual"] = tabela["valor"] / total if total else 0
    return tabela.sort_values("valor", ascending=False)


def gasto_acumulado_por_dia(df_mes: pd.DataFrame) -> pd.DataFrame:
    """Soma das despesas dia a dia, acumulada (para o gráfico de evolução)."""
    por_dia = despesas(df_mes).groupby("data", as_index=False)["valor"].sum().sort_values("data")
    por_dia["acumulado"] = por_dia["valor"].cumsum()
    return por_dia


def comparar_orcamento(df_mes: pd.DataFrame, orcamento: dict) -> pd.DataFrame:
    """GASTO REAL x LIMITE por categoria."""
    gastos = despesas(df_mes).groupby("categoria")["valor"].sum()
    linhas = []
    for categoria, limite in orcamento.items():
        gasto = float(gastos.get(categoria, 0))
        if limite <= 0 and gasto == 0:
            continue
        linhas.append({
            "categoria": categoria,
            "limite": limite,
            "gasto": gasto,
            "diferenca": gasto - limite,
            "uso": gasto / limite if limite > 0 else 0,
            "estourou": limite > 0 and gasto > limite,
        })
    return pd.DataFrame(linhas)


# ---------------- PADRÕES ----------------
def variacao_mes_anterior(df: pd.DataFrame, mes: str) -> pd.DataFrame:
    """Compara o gasto de cada categoria com o mês anterior."""
    df = preparar(df)
    anterior = mes_anterior(meses_disponiveis(df), mes)
    if anterior is None:
        return pd.DataFrame()
    gastos = despesas(df)
    atual_cat = gastos[gastos["mes"] == mes].groupby("categoria")["valor"].sum()
    ant_cat = gastos[gastos["mes"] == anterior].groupby("categoria")["valor"].sum()
    tabela = pd.DataFrame({"anterior": ant_cat, "atual": atual_cat}).fillna(0)
    tabela = tabela[tabela["anterior"] > 0]
    tabela["variacao"] = tabela["atual"] / tabela["anterior"] - 1
    return tabela.reset_index(names="categoria").sort_values("variacao", ascending=False)


def despesas_recorrentes(df: pd.DataFrame) -> pd.DataFrame:
    """
    Recorrente = mesma descrição, UMA vez por mês, em 2 ou mais meses,
    com valor parecido (ex.: Netflix, aluguel, internet).
    """
    gastos = despesas(preparar(df)).copy()
    if gastos.empty:
        return pd.DataFrame()
    gastos["chave"] = gastos["descricao"].map(normalizar)
    por_mes = gastos.groupby(["chave", "mes"]).agg(qtd=("valor", "size"), valor=("valor", "sum"))
    resultado = []
    for chave, grupo in por_mes.groupby(level="chave"):
        if len(grupo) < 2 or (grupo["qtd"] > 1).any():
            continue
        media = grupo["valor"].mean()
        if (abs(grupo["valor"] - media) / media).max() <= VARIACAO_RECORRENTE:
            exemplo = gastos[gastos["chave"] == chave].iloc[-1]
            resultado.append({
                "descricao": exemplo["descricao"],
                "categoria": exemplo["categoria"],
                "valor_medio": media,
                "meses": len(grupo),
            })
    return pd.DataFrame(resultado).sort_values("valor_medio", ascending=False) if resultado else pd.DataFrame()


def despesas_atipicas(df: pd.DataFrame, mes: str) -> pd.DataFrame:
    """
    Despesa atípica = valor acima de (média + 2 desvios-padrão) da sua categoria,
    considerando todo o histórico. Só vale para categorias com histórico suficiente.
    """
    gastos = despesas(preparar(df))
    estatisticas = gastos.groupby("categoria")["valor"].agg(["mean", "std", "count"])
    estatisticas = estatisticas[estatisticas["count"] >= MIN_TRANSACOES_ATIPICO]
    do_mes = gastos[(gastos["mes"] == mes) & gastos["categoria"].isin(estatisticas.index)].copy()
    if do_mes.empty:
        return pd.DataFrame()
    do_mes["media_categoria"] = do_mes["categoria"].map(estatisticas["mean"])
    do_mes["limite_normal"] = do_mes["media_categoria"] + 2 * do_mes["categoria"].map(estatisticas["std"])
    return do_mes[do_mes["valor"] > do_mes["limite_normal"]]


def gerar_insights(df: pd.DataFrame, mes: str, salario: float) -> list[str]:
    """Frases de padrão de consumo, todas calculadas a partir dos dados."""
    df = preparar(df)
    df_mes = df[df["mes"] == mes]
    insights = []

    por_cat = gastos_por_categoria(df_mes)
    if not por_cat.empty:
        topo = por_cat.iloc[0]
        insights.append(f"**{topo['categoria']}** é a categoria que mais consumiu dinheiro: "
                        f"{formatar_reais(topo['valor'])} ({topo['percentual']:.0%} das despesas).")
        top3 = por_cat.head(3)
        insights.append(f"As 3 maiores categorias ({', '.join(top3['categoria'])}) somam "
                        f"{top3['percentual'].sum():.0%} das despesas do mês.")

    ind = indicadores(df_mes, salario)
    if salario > 0:
        insights.append(f"Você utilizou **{ind['pct_salario']:.0%} do salário** neste mês.")

    variacao = variacao_mes_anterior(df, mes)
    for linha in variacao.itertuples():
        if linha.variacao >= AUMENTO_RELEVANTE:
            insights.append(f"Seus gastos com **{linha.categoria}** aumentaram {linha.variacao:.0%} "
                            f"em relação ao mês anterior ({formatar_reais(linha.anterior)} → "
                            f"{formatar_reais(linha.atual)}).")
        elif linha.variacao <= -AUMENTO_RELEVANTE:
            insights.append(f"Seus gastos com **{linha.categoria}** diminuíram {abs(linha.variacao):.0%} "
                            f"em relação ao mês anterior.")

    assinaturas = despesas(df_mes)[despesas(df_mes)["categoria"] == "Assinaturas"]
    if not assinaturas.empty:
        insights.append(f"Você possui **{len(assinaturas)} assinaturas** neste mês, somando "
                        f"{formatar_reais(assinaturas['valor'].sum())}.")

    recorrentes = despesas_recorrentes(df)
    if not recorrentes.empty:
        insights.append(f"Foram detectadas **{len(recorrentes)} despesas recorrentes** "
                        f"(fixas todo mês), somando {formatar_reais(recorrentes['valor_medio'].sum())} por mês.")
    return insights


# ---------------- ALERTAS AUTOMÁTICOS (decisão por regras) ----------------
def gerar_alertas(df: pd.DataFrame, mes: str, salario: float, orcamento: dict) -> list[dict]:
    """Cada alerta: {'nivel': 'erro'|'aviso'|'info', 'titulo': ..., 'mensagem': ...}"""
    df = preparar(df)
    df_mes = df[df["mes"] == mes]
    ind = indicadores(df_mes, salario)
    alertas = []

    for linha in comparar_orcamento(df_mes, orcamento).itertuples():
        if linha.estourou:
            alertas.append({
                "nivel": "erro",
                "titulo": f"Orçamento de {linha.categoria} ultrapassado",
                "mensagem": f"Você gastou {formatar_reais(linha.gasto)} com {linha.categoria.lower()}. "
                            f"Seu limite era {formatar_reais(linha.limite)}. "
                            f"Você ultrapassou o orçamento em {formatar_reais(linha.diferenca)}.",
            })

    if salario > 0 and ind["pct_salario"] >= PERCENTUAL_SALARIO_ALERTA:
        alertas.append({
            "nivel": "erro" if ind["pct_salario"] >= 1 else "aviso",
            "titulo": "Uso elevado do salário",
            "mensagem": f"As despesas do mês já somam {ind['pct_salario']:.0%} do seu salário "
                        f"({formatar_reais(ind['despesas'])} de {formatar_reais(salario)}).",
        })

    if ind["receitas"] > 0 and ind["saldo"] < ind["receitas"] * SALDO_BAIXO:
        alertas.append({
            "nivel": "erro" if ind["saldo"] < 0 else "aviso",
            "titulo": "Saldo baixo",
            "mensagem": f"Seu saldo disponível é {formatar_reais(ind['saldo'])}, menos de "
                        f"{SALDO_BAIXO:.0%} das receitas do mês.",
        })

    for linha in despesas_atipicas(df, mes).itertuples():
        alertas.append({
            "nivel": "aviso",
            "titulo": "Despesa muito acima da média",
            "mensagem": f"\"{linha.descricao}\" ({formatar_reais(linha.valor)}) está bem acima da média "
                        f"de {linha.categoria} ({formatar_reais(linha.media_categoria)}).",
        })

    recorrentes = despesas_recorrentes(df)
    if not recorrentes.empty:
        alertas.append({
            "nivel": "info",
            "titulo": "Despesas recorrentes",
            "mensagem": f"{len(recorrentes)} despesas se repetem todo mês "
                        f"({', '.join(recorrentes['descricao'].head(5))}...), somando "
                        f"{formatar_reais(recorrentes['valor_medio'].sum())} por mês.",
        })

    revisar = df_mes[df_mes["origem"] == "Revisar"]
    if not revisar.empty:
        alertas.append({
            "nivel": "info",
            "titulo": "Movimentações para revisar",
            "mensagem": f"{len(revisar)} movimentação(ões) não puderam ser classificadas com segurança "
                        f"e estão como 'Outros': {', '.join(revisar['descricao'])}.",
        })
    return alertas
