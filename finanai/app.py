"""
FinanAI — Para onde foi meu salário?
Protótipo acadêmico: ENTRADA → PROCESSAMENTO → CLASSIFICAÇÃO/IA → ANÁLISE → DECISÃO → SAÍDA

Executar:  python -m streamlit run app.py
"""
from datetime import date

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src import analise
from src.classificacao import (CATEGORIAS, ORIGEM_USUARIO, ClassificadorML,
                               classificar, classificar_por_regras, classificar_tabela)
from src.dados import (ARQUIVO_DEMO, TIPOS, formatar_reais, ler_csv, ler_treino,
                       tabela_vazia, validar_movimentacao)
from src.recomendacoes import gerar_recomendacoes

st.set_page_config(page_title="FinanAI", page_icon="💰", layout="wide")

COR_PRINCIPAL = "#2563eb"
COR_DESPESA = "#dc2626"
COR_RECEITA = "#16a34a"
COR_NEUTRA = "#94a3b8"
MESES = ["Janeiro", "Fevereiro", "Março", "Abril", "Maio", "Junho", "Julho",
         "Agosto", "Setembro", "Outubro", "Novembro", "Dezembro"]


def md(texto: str) -> str:
    """Escapa o '$' para o Streamlit não interpretar 'R$ ... R$' como fórmula matemática."""
    return texto.replace("$", "\\$")


def nome_mes(mes: str) -> str:
    ano, numero = mes.split("-")
    return f"{MESES[int(numero) - 1]}/{ano}"


# =====================================================================
# ESTADO DA APLICAÇÃO (fica na memória enquanto a página estiver aberta)
# =====================================================================
def treinar_modelo() -> ClassificadorML:
    """Treina o ML com os exemplos base + as correções feitas pelo usuário."""
    exemplos = pd.concat([ler_treino(), pd.DataFrame(st.session_state.correcoes,
                                                     columns=["descricao", "categoria"])])
    modelo = ClassificadorML()
    modelo.treinar(exemplos)
    return modelo


def carregar_demonstracao() -> None:
    df, _ = ler_csv(ARQUIVO_DEMO)
    st.session_state.correcoes = []
    st.session_state.modelo = treinar_modelo()
    st.session_state.movimentacoes = classificar_tabela(df, st.session_state.modelo)
    st.session_state.salario = 4500.0
    st.session_state.orcamento = dict(analise.ORCAMENTO_PADRAO)
    st.session_state.orcamento_tabela = pd.DataFrame({"Categoria": list(analise.ORCAMENTO_PADRAO),
                                                      "Limite (R$)": list(analise.ORCAMENTO_PADRAO.values())})


if "movimentacoes" not in st.session_state:
    carregar_demonstracao()

df_todas: pd.DataFrame = st.session_state.movimentacoes
modelo: ClassificadorML = st.session_state.modelo


def adicionar_movimentacao(data_mov, descricao, valor, tipo, categoria_escolhida="") -> dict:
    categoria, origem = classificar(descricao, tipo, modelo, categoria_escolhida)
    nova = pd.DataFrame([{"data": data_mov, "descricao": descricao.strip(), "valor": float(valor),
                          "tipo": tipo, "categoria": categoria, "origem": origem}])
    st.session_state.movimentacoes = pd.concat([st.session_state.movimentacoes, nova], ignore_index=True)
    return {"descricao": descricao.strip(), "valor": valor, "categoria": categoria, "origem": origem}


def explicar_origem(origem: str) -> str:
    if origem.startswith("ML"):
        return "Nenhuma regra reconheceu o texto; o **modelo de Machine Learning** sugeriu a categoria."
    return {
        "Regra": "Encontrada por **regra de palavra-chave** (automação simples).",
        "Usuário": "Categoria **escolhida pelo usuário**.",
        "Revisar": "Nem as regras nem o ML tiveram confiança suficiente → ficou como **Outros** para revisão humana.",
    }.get(origem, origem)


# =====================================================================
# SIDEBAR
# =====================================================================
with st.sidebar:
    st.title("💰 FinanAI")
    st.caption("Para onde foi meu salário?")
    pagina = st.radio("Navegação", ["📊 Dashboard", "🧾 Movimentações", "🎯 Orçamentos",
                                    "🧠 Análise Inteligente", "💡 Recomendações", "ℹ️ Sobre a solução"],
                      label_visibility="collapsed")
    st.divider()
    st.session_state.salario = st.number_input("Salário mensal (R$)", min_value=0.0, step=100.0,
                                               value=float(st.session_state.salario), format="%.2f")
    meses = analise.meses_disponiveis(df_todas)
    mes = st.selectbox("Mês analisado", meses, index=len(meses) - 1, format_func=nome_mes) if meses else None
    st.divider()
    if st.button("🔄 Restaurar dados de demonstração", width="stretch"):
        carregar_demonstracao()
        st.rerun()
    if st.button("🗑️ Limpar todas as movimentações", width="stretch"):
        st.session_state.movimentacoes = tabela_vazia()
        st.rerun()
    st.warning("⚠️ Todos os dados são **fictícios**, apenas para demonstração.", icon=None)

salario = st.session_state.salario
orcamento = st.session_state.orcamento

if mes is None and pagina not in ("🧾 Movimentações", "ℹ️ Sobre a solução"):
    st.info("Nenhuma movimentação cadastrada. Vá em **Movimentações** para cadastrar ou importar, "
            "ou clique em **Restaurar dados de demonstração**.")
    st.stop()

if mes:
    df_prep = analise.preparar(df_todas)
    df_mes = df_prep[df_prep["mes"] == mes]


def mostrar_alertas(alertas: list[dict]) -> None:
    if not alertas:
        st.success("✅ Nenhum alerta para este mês.")
    for alerta in alertas:
        texto = md(f"**{alerta['titulo']}** — {alerta['mensagem']}")
        {"erro": st.error, "aviso": st.warning, "info": st.info}[alerta["nivel"]](texto, icon="⚠️"
                                                                                 if alerta["nivel"] != "info" else "ℹ️")


def estilo_grafico(fig, altura=340):
    fig.update_layout(template="plotly_white", height=altura, margin=dict(l=10, r=10, t=40, b=10),
                      font=dict(size=13), legend=dict(orientation="h", y=-0.15), separators=",.")
    return fig


# =====================================================================
# PÁGINA: DASHBOARD
# =====================================================================
if pagina == "📊 Dashboard":
    st.header(f"📊 Dashboard — {nome_mes(mes)}")
    ind = analise.indicadores(df_mes, salario)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Salário mensal", formatar_reais(ind["salario"]), border=True)
    c2.metric("Total de receitas", formatar_reais(ind["receitas"]), border=True)
    c3.metric("Total de despesas", formatar_reais(ind["despesas"]), border=True)
    c4.metric("Saldo disponível", formatar_reais(ind["saldo"]), border=True)
    c5, c6, c7 = st.columns(3)
    c5.metric("% do salário utilizado", f"{ind['pct_salario']:.0%}", border=True)
    c6.metric(f"Categoria com maior gasto ({formatar_reais(ind['maior_categoria_valor'])})",
              ind["maior_categoria"], border=True)
    c7.metric("Quantidade de transações", ind["qtd_transacoes"], border=True)

    col_a, col_b = st.columns([3, 2])
    with col_a:
        por_cat = analise.gastos_por_categoria(df_mes).sort_values("valor")
        fig = px.bar(por_cat, x="valor", y="categoria", orientation="h", title="Gastos por categoria",
                     text=[formatar_reais(v) for v in por_cat["valor"]],
                     color_discrete_sequence=[COR_PRINCIPAL])
        fig.update_layout(xaxis_title=None, yaxis_title=None)
        st.plotly_chart(estilo_grafico(fig), width="stretch")
    with col_b:
        fig = go.Figure(go.Bar(x=["Receitas", "Despesas"], y=[ind["receitas"], ind["despesas"]],
                               marker_color=[COR_RECEITA, COR_DESPESA],
                               text=[formatar_reais(ind["receitas"]), formatar_reais(ind["despesas"])]))
        fig.update_layout(title="Receitas x Despesas")
        st.plotly_chart(estilo_grafico(fig), width="stretch")

    col_c, col_d = st.columns(2)
    with col_c:
        evolucao = analise.gasto_acumulado_por_dia(df_mes)
        fig = px.line(evolucao, x="data", y="acumulado", markers=True,
                      title="Evolução dos gastos ao longo do mês (acumulado)",
                      color_discrete_sequence=[COR_DESPESA])
        if salario > 0:
            fig.add_hline(y=salario, line_dash="dash", line_color=COR_NEUTRA,
                          annotation_text="Salário", annotation_position="top left")
        fig.update_layout(xaxis_title=None, yaxis_title="R$", xaxis_tickformat="%d/%m")
        st.plotly_chart(estilo_grafico(fig), width="stretch")
    with col_d:
        comp = analise.comparar_orcamento(df_mes, orcamento)
        fig = go.Figure([
            go.Bar(name="Orçamento", x=comp["categoria"], y=comp["limite"], marker_color=COR_NEUTRA),
            go.Bar(name="Gasto real", x=comp["categoria"], y=comp["gasto"],
                   marker_color=[COR_DESPESA if e else COR_PRINCIPAL for e in comp["estourou"]]),
        ])
        fig.update_layout(title="Orçamento x Gasto real (vermelho = estourou)", barmode="group")
        st.plotly_chart(estilo_grafico(fig), width="stretch")

    st.subheader("🚨 Alertas")
    mostrar_alertas(analise.gerar_alertas(df_todas, mes, salario, orcamento))

    st.subheader("💡 Principais recomendações")
    for rec in gerar_recomendacoes(df_todas, mes, salario, orcamento)[:3]:
        st.markdown(md(f"- **{rec['titulo']}:** {rec['texto']}"))


# =====================================================================
# PÁGINA: MOVIMENTAÇÕES (ENTRADA)
# =====================================================================
elif pagina == "🧾 Movimentações":
    st.header("🧾 Movimentações")

    col_form, col_teste = st.columns([3, 2])
    with col_form:
        with st.container(border=True):
            st.subheader("➕ Nova movimentação")
            with st.form("nova", clear_on_submit=True):
                f1, f2 = st.columns(2)
                data_mov = f1.date_input("Data", value=date.today(), format="DD/MM/YYYY")
                tipo = f2.radio("Tipo", TIPOS, horizontal=True)
                descricao = st.text_input("Descrição", placeholder="ex.: iFood, Uber, Farmácia...")
                f3, f4 = st.columns(2)
                valor = f3.number_input("Valor (R$)", min_value=0.0, step=10.0, format="%.2f")
                categoria_escolhida = f4.selectbox("Categoria (opcional)", ["Automática"] + CATEGORIAS)
                enviar = st.form_submit_button("Adicionar", type="primary", width="stretch")

            if enviar:
                erros = validar_movimentacao(data_mov, descricao, valor, tipo)
                if erros:
                    for erro in erros:
                        st.error(erro)
                else:
                    escolha = "" if categoria_escolhida == "Automática" else categoria_escolhida
                    st.session_state.ultimo = adicionar_movimentacao(data_mov, descricao, valor, tipo, escolha)
                    st.rerun()

            if st.button("⚡ Demonstração: adicionar 'iFood - jantar de sexta' (R$ 150,00)",
                         width="stretch"):
                dia = date(int(mes[:4]), int(mes[5:]), 28) if mes else date.today()
                st.session_state.ultimo = adicionar_movimentacao(dia, "iFood - jantar de sexta", 150.0, "Despesa")
                st.rerun()

            if "ultimo" in st.session_state:
                u = st.session_state.pop("ultimo")
                st.success(md(f"✔ **{u['descricao']}** ({formatar_reais(u['valor'])}) → categoria "
                              f"**{u['categoria']}**  \nOrigem: {explicar_origem(u['origem'])}"))

    with col_teste:
        with st.container(border=True):
            st.subheader("🔍 Testar a classificação")
            st.caption("Digite uma descrição e compare as regras com o Machine Learning.")
            texto = st.text_input("Descrição para testar", value="Hamburgueria do Centro")
            if texto.strip():
                regra = classificar_por_regras(texto)
                cat_ml, conf = modelo.prever(texto)
                st.markdown(f"**Regras (palavra-chave):** {regra or '❌ nenhuma regra encontrada'}")
                st.markdown(f"**Machine Learning:** {cat_ml} — confiança {conf:.0%}")
                final, origem = classificar(texto, "Despesa", modelo)
                st.markdown(f"**Decisão final:** {final} ({origem})")

    with st.expander("📥 Importar / exportar CSV"):
        st.caption("Formato: `data,descricao,valor,tipo,categoria` — data AAAA-MM-DD, tipo Receita ou "
                   "Despesa, categoria opcional. Use apenas dados fictícios.")
        arquivo = st.file_uploader("Arquivo CSV", type="csv")
        substituir = st.checkbox("Substituir as movimentações atuais")
        if arquivo and st.button("Importar"):
            try:
                importado, descartadas = ler_csv(arquivo)
                importado = classificar_tabela(importado, modelo)
                base = tabela_vazia() if substituir else st.session_state.movimentacoes
                st.session_state.movimentacoes = pd.concat([base, importado], ignore_index=True)
                st.success(f"{len(importado)} movimentações importadas. {descartadas} linha(s) inválida(s) ignorada(s).")
                st.rerun()
            except ValueError as erro:
                st.error(str(erro))
        st.download_button("Baixar movimentações (CSV)",
                           st.session_state.movimentacoes.to_csv(index=False).encode("utf-8"),
                           "movimentacoes.csv", "text/csv")

    # ---- Revisão humana: corrigir categoria e ensinar o modelo ----
    pendentes = st.session_state.movimentacoes
    pendentes = pendentes[pendentes["origem"] == "Revisar"]
    if not pendentes.empty:
        with st.container(border=True):
            st.subheader("✋ Revisão humana")
            st.caption("Estas movimentações não foram classificadas com segurança. Ao corrigir, "
                       "o exemplo entra no treino e o modelo de ML aprende.")
            r1, r2, r3 = st.columns([3, 2, 1])
            indice = r1.selectbox("Movimentação", pendentes.index,
                                  format_func=lambda i: f"{pendentes.loc[i, 'descricao']} — "
                                                        f"{formatar_reais(pendentes.loc[i, 'valor'])}")
            nova_cat = r2.selectbox("Categoria correta", CATEGORIAS)
            if r3.button("Salvar", type="primary", width="stretch"):
                movs = st.session_state.movimentacoes
                movs.loc[indice, ["categoria", "origem"]] = [nova_cat, ORIGEM_USUARIO]
                st.session_state.correcoes.append((movs.loc[indice, "descricao"], nova_cat))
                st.session_state.modelo = treinar_modelo()
                st.session_state.movimentacoes = classificar_tabela(movs, st.session_state.modelo)
                st.rerun()

    st.subheader(f"Movimentações de {nome_mes(mes)}" if mes else "Movimentações")
    if mes:
        tabela = df_mes.sort_values("data", ascending=False).copy()
        tabela["data"] = tabela["data"].dt.strftime("%d/%m/%Y")
        tabela["valor"] = [formatar_reais(v if t == "Receita" else -v) for v, t in zip(tabela["valor"], tabela["tipo"])]
        st.dataframe(tabela[["data", "descricao", "valor", "tipo", "categoria", "origem"]],
                     hide_index=True, width="stretch",
                     column_config={"data": "Data", "descricao": "Descrição", "valor": "Valor",
                                    "tipo": "Tipo", "categoria": "Categoria", "origem": "Classificado por"})


# =====================================================================
# PÁGINA: ORÇAMENTOS
# =====================================================================
elif pagina == "🎯 Orçamentos":
    st.header(f"🎯 Orçamentos — {nome_mes(mes)}")
    col_edit, col_graf = st.columns([2, 3])
    with col_edit:
        st.caption("Defina o limite mensal de cada categoria (edite a coluna Limite).")
        editado = st.data_editor(
            st.session_state.orcamento_tabela, key="editor_orcamento", hide_index=True, width="stretch", disabled=["Categoria"],
            column_config={"Limite (R$)": st.column_config.NumberColumn(min_value=0, step=50, format="R$ %.2f")})
        st.session_state.orcamento = dict(zip(editado["Categoria"], editado["Limite (R$)"].fillna(0)))
        orcamento = st.session_state.orcamento

    comp = analise.comparar_orcamento(df_mes, orcamento)
    with col_graf:
        fig = go.Figure([
            go.Bar(name="Orçamento", y=comp["categoria"], x=comp["limite"], orientation="h", marker_color=COR_NEUTRA),
            go.Bar(name="Gasto real", y=comp["categoria"], x=comp["gasto"], orientation="h",
                   marker_color=[COR_DESPESA if e else COR_PRINCIPAL for e in comp["estourou"]]),
        ])
        fig.update_layout(title="Gasto real x Limite", barmode="group", yaxis=dict(autorange="reversed"))
        st.plotly_chart(estilo_grafico(fig, 460), width="stretch")

    tabela = pd.DataFrame({
        "Categoria": comp["categoria"],
        "Limite": comp["limite"].map(formatar_reais),
        "Gasto": comp["gasto"].map(formatar_reais),
        "Uso do orçamento": comp["uso"],
        "Situação": ["🔴 Ultrapassou " + formatar_reais(d) if e else "🟢 Dentro do limite"
                     for e, d in zip(comp["estourou"], comp["diferenca"])],
    })
    st.dataframe(tabela, hide_index=True, width="stretch",
                 column_config={"Uso do orçamento": st.column_config.ProgressColumn(
                     format="percent", min_value=0, max_value=max(1.0, comp["uso"].max()))})

    st.subheader("Alertas de orçamento")
    mostrar_alertas([a for a in analise.gerar_alertas(df_todas, mes, salario, orcamento)
                     if a["titulo"].startswith("Orçamento")])


# =====================================================================
# PÁGINA: ANÁLISE INTELIGENTE
# =====================================================================
elif pagina == "🧠 Análise Inteligente":
    st.header(f"🧠 Análise Inteligente — {nome_mes(mes)}")

    st.subheader("1. Como as despesas foram classificadas")
    desp_mes = analise.despesas(df_mes)
    origem_simples = desp_mes["origem"].str.replace(r" \(.*\)", "", regex=True)
    contagem = origem_simples.value_counts()
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("⚙️ Regras (automação)", int(contagem.get("Regra", 0)), border=True)
    k2.metric("🤖 Machine Learning", int(contagem.get("ML", 0)), border=True)
    k3.metric("👤 Escolhidas pelo usuário", int(contagem.get("Usuário", 0)), border=True)
    k4.metric("✋ Para revisar", int(contagem.get("Revisar", 0)), border=True)

    classificadas_ml = desp_mes[origem_simples.isin(["ML", "Revisar"])]
    if not classificadas_ml.empty:
        st.caption("Movimentações em que as regras falharam e o ML foi consultado:")
        st.dataframe(classificadas_ml[["descricao", "categoria", "origem"]], hide_index=True,
                     width="stretch",
                     column_config={"descricao": "Descrição", "categoria": "Categoria", "origem": "Resultado"})
    st.caption(f"Modelo: TF-IDF (pedaços de 2 a 4 letras) + Naive Bayes, treinado com "
               f"{modelo.qtd_exemplos} exemplos. Confiança mínima para aceitar: 40%.")

    st.subheader("2. Padrões de consumo identificados")
    for frase in analise.gerar_insights(df_todas, mes, salario):
        st.markdown(md(f"- {frase}"))

    col1, col2 = st.columns(2)
    with col1:
        variacao = analise.variacao_mes_anterior(df_todas, mes)
        if variacao.empty:
            st.info("Sem mês anterior para comparar.")
        else:
            fig = px.bar(variacao.sort_values("variacao"), x="variacao", y="categoria", orientation="h",
                         title="Variação em relação ao mês anterior",
                         color=variacao.sort_values("variacao")["variacao"] > 0,
                         color_discrete_map={True: COR_DESPESA, False: COR_RECEITA},
                         text=[f"{v:+.0%}" for v in variacao.sort_values("variacao")["variacao"]])
            fig.update_layout(showlegend=False, xaxis_tickformat=".0%", xaxis_title=None, yaxis_title=None)
            st.plotly_chart(estilo_grafico(fig, 380), width="stretch")
    with col2:
        st.markdown("**🔁 Despesas recorrentes** (uma vez por mês, valor parecido)")
        recorrentes = analise.despesas_recorrentes(df_todas)
        if recorrentes.empty:
            st.caption("Nenhuma despesa recorrente detectada (é preciso ter 2 meses ou mais).")
        else:
            st.dataframe(pd.DataFrame({"Descrição": recorrentes["descricao"],
                                       "Categoria": recorrentes["categoria"],
                                       "Valor médio": recorrentes["valor_medio"].map(formatar_reais),
                                       "Meses": recorrentes["meses"]}),
                         hide_index=True, width="stretch")

    st.markdown("**📈 Despesas muito acima da média da categoria** (média + 2 desvios-padrão)")
    atipicas = analise.despesas_atipicas(df_todas, mes)
    if atipicas.empty:
        st.caption("Nenhuma despesa fora do padrão neste mês.")
    else:
        st.dataframe(pd.DataFrame({"Descrição": atipicas["descricao"], "Categoria": atipicas["categoria"],
                                   "Valor": atipicas["valor"].map(formatar_reais),
                                   "Média da categoria": atipicas["media_categoria"].map(formatar_reais)}),
                     hide_index=True, width="stretch")


# =====================================================================
# PÁGINA: RECOMENDAÇÕES
# =====================================================================
elif pagina == "💡 Recomendações":
    st.header(f"💡 Recomendações — {nome_mes(mes)}")
    st.caption("Geradas automaticamente a partir dos seus dados. Cada uma mostra em que se baseou.")
    for rec in gerar_recomendacoes(df_todas, mes, salario, orcamento):
        with st.container(border=True):
            st.markdown(f"#### {rec['titulo']}")
            st.markdown(md(rec["texto"]))
            st.caption(md(f"📌 Baseado em: {rec['baseado_em']}"))
    st.info("ℹ️ Estas sugestões são informativas e **não substituem orientação financeira profissional**. "
            "Decisões financeiras continuam sendo responsabilidade do usuário.")


# =====================================================================
# PÁGINA: SOBRE A SOLUÇÃO
# =====================================================================
elif pagina == "ℹ️ Sobre a solução":
    st.header("ℹ️ Sobre a solução")
    st.markdown("""
**Fluxo:** ENTRADA → PROCESSAMENTO → CLASSIFICAÇÃO/IA → ANÁLISE → DECISÃO/AUTOMAÇÃO → SAÍDA

| Etapa | No FinanAI |
|---|---|
| Entrada | Salário, cadastro manual, importação de CSV, dados fictícios |
| Processamento | Validação dos dados, conversão de datas e valores |
| Classificação | Regras de palavra-chave → Machine Learning → revisão humana |
| Análise | Totais, variação mensal, recorrências, despesas atípicas |
| Decisão/Automação | Alertas de orçamento, salário, saldo e padrões |
| Saída | Dashboard, gráficos, alertas e recomendações explicáveis |
""")

    st.subheader("As 7 perguntas da atividade")
    perguntas = {
        "1. Qual é exatamente o problema?":
            "Uma pessoa tem dezenas de movimentações no mês e não consegue enxergar para onde o dinheiro vai, "
            "nem perceber a tempo quando está gastando demais.",
        "2. O que hoje é feito manualmente?":
            "Ler o extrato linha por linha, anotar cada gasto em uma planilha, escolher a categoria, somar, "
            "comparar com o mês anterior e lembrar dos limites de cada categoria.",
        "3. O que pode ser automatizado?":
            "Classificação por palavras-chave, somas por categoria, comparação com orçamento e com o mês "
            "anterior, detecção de recorrências e geração de alertas. Tudo isso são **regras simples** — não é IA.",
        "4. Onde IA realmente seria necessária?":
            "Em descrições ambíguas ou sem padrão (\"Pix para Carlos\", \"PAG*XPTO\"), que regras não entendem; "
            "e numa versão futura com IA generativa para responder perguntas em linguagem natural "
            "(\"quanto gastei com comida nas últimas semanas?\").",
        "5. Onde aprendizado de máquina poderia aparecer?":
            "Já aparece: um modelo TF-IDF + Naive Bayes classifica descrições que nenhuma regra reconhece e "
            "**aprende com as correções do usuário**. No futuro: prever o gasto do fim do mês e detectar "
            "anomalias com mais histórico.",
        "6. Quais dados seriam necessários?":
            "Data, descrição, valor e tipo de cada movimentação; salário; limites por categoria; exemplos "
            "rotulados para treinar o modelo. Obtidos por cadastro manual, CSV exportado do banco ou, "
            "no futuro, Open Finance (com consentimento).",
        "7. Quais riscos, erros, vieses ou problemas de privacidade?":
            "Classificação errada (ex.: o modelo pode achar que \"Barbearia\" é Alimentação), recomendações "
            "inadequadas para a realidade da pessoa, dados incompletos distorcendo a análise, viés dos "
            "exemplos de treino e exposição de dados financeiros sensíveis.",
    }
    for pergunta, resposta in perguntas.items():
        with st.expander(pergunta):
            st.markdown(resposta)

    st.subheader("Automação x IA x Machine Learning")
    st.markdown("""
| | O que é | Onde está no FinanAI |
|---|---|---|
| **Automação simples** | Regras fixas escritas por pessoas (se/então) | Palavras-chave, somas, orçamentos, alertas, recomendações |
| **Machine Learning** | Modelo que aprende padrões a partir de exemplos | TF-IDF + Naive Bayes para descrições desconhecidas; reaprende com correções |
| **IA generativa** | Modelo que entende/gera linguagem | Não usada no protótipo — seria o próximo passo (chat sobre os gastos) |
""")

    st.subheader("🔒 Privacidade e riscos")
    st.markdown("""
- **Dados financeiros são sensíveis** (LGPD): exigem consentimento, armazenamento protegido e acesso restrito.
- **Exposição de informações:** este protótipo roda localmente e não envia dados para a internet.
- **Classificação incorreta:** por isso existe a revisão humana e a coluna "Classificado por".
- **Recomendações inadequadas:** são regras genéricas e não conhecem a realidade de cada pessoa.
- **Dados incompletos:** se faltar movimentação, a análise fica distorcida.
- **Vieses no modelo:** o ML só sabe o que está nos exemplos de treino.
- **Responsabilidade:** decisões financeiras importantes continuam sendo **do usuário**.
- **Este protótipo usa apenas dados fictícios.**
""")
