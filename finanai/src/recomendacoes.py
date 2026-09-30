"""
RECOMENDAÇÕES: sugestões simples e EXPLICÁVEIS — cada uma diz em quais dados se baseou.
Não é aconselhamento financeiro profissional; a decisão final é sempre do usuário.
"""
import pandas as pd

from src import analise
from src.dados import formatar_reais

CATEGORIAS_FIXAS = {"Moradia", "Contas", "Educação"}  # difíceis de cortar no curto prazo


def gerar_recomendacoes(df: pd.DataFrame, mes: str, salario: float, orcamento: dict) -> list[dict]:
    """Cada recomendação: {'titulo', 'texto', 'baseado_em'}"""
    df = analise.preparar(df)
    df_mes = df[df["mes"] == mes]
    ind = analise.indicadores(df_mes, salario)
    recomendacoes = []

    # 1. Categorias que estouraram o orçamento
    orc = analise.comparar_orcamento(df_mes, orcamento)
    for linha in orc[orc["estourou"]].itertuples() if not orc.empty else []:
        recomendacoes.append({
            "titulo": f"Revise seus gastos com {linha.categoria}",
            "texto": f"Seu gasto com {linha.categoria.lower()} ultrapassou o limite definido. "
                     f"Considere revisar as despesas dessa categoria nos próximos dias.",
            "baseado_em": f"Gasto de {formatar_reais(linha.gasto)} x limite de {formatar_reais(linha.limite)} "
                          f"({linha.uso:.0%} do orçamento).",
        })

    # 2. Uso do salário
    if salario > 0 and ind["pct_salario"] >= analise.PERCENTUAL_SALARIO_ALERTA:
        recomendacoes.append({
            "titulo": "Atenção ao restante do mês",
            "texto": f"Você já utilizou {ind['pct_salario']:.0%} do salário. Evite novas despesas "
                     f"não essenciais até o próximo pagamento.",
            "baseado_em": f"Despesas de {formatar_reais(ind['despesas'])} / salário de {formatar_reais(salario)}.",
        })

    # 3. Assinaturas
    assinaturas = analise.despesas(df_mes)
    assinaturas = assinaturas[assinaturas["categoria"] == "Assinaturas"]
    if len(assinaturas) >= 3:
        recomendacoes.append({
            "titulo": "Reavalie suas assinaturas",
            "texto": f"Suas assinaturas representam {formatar_reais(assinaturas['valor'].sum())} mensais. "
                     f"Verifique se todas estão sendo usadas.",
            "baseado_em": f"{len(assinaturas)} assinaturas no mês: {', '.join(assinaturas['descricao'])}.",
        })

    # 4. Maior gasto variável (onde cortar é mais fácil)
    por_cat = analise.gastos_por_categoria(df_mes)
    variaveis = por_cat[~por_cat["categoria"].isin(CATEGORIAS_FIXAS | {"Outros", "Assinaturas"})]
    if not variaveis.empty:
        maior = variaveis.iloc[0]
        economia = maior["valor"] * 0.10
        recomendacoes.append({
            "titulo": f"Pequena economia em {maior['categoria']}",
            "texto": f"{maior['categoria']} é seu maior gasto variável. Reduzir 10% nessa categoria "
                     f"economizaria cerca de {formatar_reais(economia)} por mês "
                     f"({formatar_reais(economia * 12)} por ano).",
            "baseado_em": f"{maior['categoria']}: {formatar_reais(maior['valor'])} no mês "
                          f"({maior['percentual']:.0%} das despesas).",
        })

    # 5. Categoria que mais cresceu
    variacao = analise.variacao_mes_anterior(df, mes)
    if not variacao.empty and variacao.iloc[0]["variacao"] >= analise.AUMENTO_RELEVANTE:
        linha = variacao.iloc[0]
        recomendacoes.append({
            "titulo": f"{linha['categoria']} está crescendo",
            "texto": f"O gasto com {linha['categoria'].lower()} subiu {linha['variacao']:.0%} em relação "
                     f"ao mês anterior. Veja se foi algo pontual ou um novo hábito.",
            "baseado_em": f"{formatar_reais(linha['anterior'])} no mês anterior → "
                          f"{formatar_reais(linha['atual'])} neste mês.",
        })

    # 6. Itens sem categoria
    revisar = df_mes[df_mes["origem"] == "Revisar"]
    if not revisar.empty:
        recomendacoes.append({
            "titulo": "Classifique as movimentações pendentes",
            "texto": "Algumas movimentações não foram reconhecidas. Informe a categoria correta: "
                     "isso melhora a análise e ensina o modelo de ML.",
            "baseado_em": f"Pendentes: {', '.join(revisar['descricao'])}.",
        })

    if not recomendacoes:
        recomendacoes.append({
            "titulo": "Tudo dentro do planejado",
            "texto": "Nenhuma categoria ultrapassou o orçamento e o uso do salário está sob controle.",
            "baseado_em": "Comparação entre gastos e orçamentos do mês.",
        })
    return recomendacoes
