"""
CLASSIFICAÇÃO DAS DESPESAS em 3 camadas, da mais simples para a mais "inteligente":

  1. Usuário   -> se a pessoa escolheu a categoria, ela manda (decisão humana).
  2. Regras    -> AUTOMAÇÃO SIMPLES: procura palavras-chave ("uber" -> Transporte).
  3. ML        -> MACHINE LEARNING: quando nenhuma regra serve, um modelo
                  TF-IDF + Naive Bayes, treinado com exemplos, sugere a categoria.
                  Se a confiança for baixa, a despesa vai para "Outros" e fica
                  marcada para revisão humana.

Para trocar o classificador no futuro (ex.: IA generativa), basta mudar
a função `classificar` — o resto do sistema não precisa saber como foi feito.
"""
import re
import unicodedata

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline

CATEGORIAS = [
    "Alimentação", "Transporte", "Moradia", "Saúde", "Educação",
    "Lazer", "Compras", "Assinaturas", "Contas", "Outros",
]
CATEGORIA_RECEITA = "Receita"
CONFIANCA_MINIMA_ML = 0.40  # abaixo disso o modelo "não tem certeza"

ORIGEM_USUARIO = "Usuário"
ORIGEM_REGRA = "Regra"
ORIGEM_ML = "ML"
ORIGEM_REVISAR = "Revisar"

# ---------------- CAMADA 2: REGRAS (automação simples) ----------------
PALAVRAS_CHAVE = {
    "Alimentação": ["ifood", "rappi", "supermercado", "padaria", "restaurante", "lanchonete", "pizzaria"],
    "Transporte":  ["uber", "99", "taxi", "posto", "gasolina", "combustivel", "estacionamento", "onibus", "metro"],
    "Moradia":     ["aluguel", "condominio", "iptu"],
    "Saúde":       ["farmacia", "drogaria", "medico", "dentista", "hospital", "laboratorio"],
    "Educação":    ["faculdade", "curso", "livraria", "escola", "mensalidade"],
    "Lazer":       ["cinema", "bar", "show", "teatro", "ingresso"],
    "Compras":     ["mercado livre", "shopee", "amazon", "magalu", "renner", "shein"],
    "Assinaturas": ["netflix", "spotify", "disney", "globoplay", "hbo", "youtube premium"],
    "Contas":      ["energia", "cemig", "agua", "copasa", "internet", "celular", "telefone", "gas"],
}


def normalizar(texto: str) -> str:
    """Minúsculas e sem acentos: 'Farmácia' -> 'farmacia'."""
    texto = unicodedata.normalize("NFKD", str(texto).lower())
    return "".join(c for c in texto if not unicodedata.combining(c))


def classificar_por_regras(descricao: str) -> str | None:
    """Devolve a categoria da primeira palavra-chave encontrada, ou None."""
    texto = normalizar(descricao)
    for categoria, palavras in PALAVRAS_CHAVE.items():
        for palavra in palavras:
            if re.search(rf"\b{re.escape(palavra)}\b", texto):  # palavra inteira
                return categoria
    return None


# ---------------- CAMADA 3: MACHINE LEARNING ----------------
class ClassificadorML:
    """
    TF-IDF transforma o texto em números (pedaços de 2 a 4 letras),
    e o Naive Bayes aprende qual categoria cada pedaço costuma indicar.
    Ex.: aprendeu "hamburgueria" -> Alimentação, então "Hamburgueria Artesanal"
    é reconhecida mesmo sem existir regra para ela.
    """

    def __init__(self):
        self.modelo = make_pipeline(
            TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), preprocessor=normalizar),
            MultinomialNB(alpha=0.3),
        )
        self.qtd_exemplos = 0

    def treinar(self, exemplos: pd.DataFrame) -> None:
        """`exemplos` precisa ter as colunas 'descricao' e 'categoria'."""
        exemplos = exemplos.dropna(subset=["descricao", "categoria"])
        exemplos = exemplos[exemplos["categoria"].isin(CATEGORIAS)]
        self.modelo.fit(exemplos["descricao"], exemplos["categoria"])
        self.qtd_exemplos = len(exemplos)

    def prever(self, descricao: str) -> tuple[str, float]:
        """Devolve (categoria mais provável, confiança de 0 a 1)."""
        probabilidades = self.modelo.predict_proba([descricao])[0]
        melhor = probabilidades.argmax()
        return self.modelo.classes_[melhor], float(probabilidades[melhor])


# ---------------- FUNÇÃO ÚNICA USADA PELO SISTEMA ----------------
def classificar(descricao: str, tipo: str, modelo: ClassificadorML,
                categoria_usuario: str = "") -> tuple[str, str]:
    """Devolve (categoria, origem) seguindo as 3 camadas."""
    if tipo == "Receita":
        return CATEGORIA_RECEITA, ORIGEM_REGRA

    if categoria_usuario in CATEGORIAS:
        return categoria_usuario, ORIGEM_USUARIO

    categoria = classificar_por_regras(descricao)
    if categoria:
        return categoria, ORIGEM_REGRA

    categoria, confianca = modelo.prever(descricao)
    if confianca >= CONFIANCA_MINIMA_ML:
        return categoria, f"{ORIGEM_ML} ({confianca:.0%})"

    return "Outros", ORIGEM_REVISAR


def classificar_tabela(df: pd.DataFrame, modelo: ClassificadorML) -> pd.DataFrame:
    """
    Classifica todas as linhas. Categorias escolhidas pelo usuário são mantidas;
    as demais são recalculadas (útil depois que o modelo aprende algo novo).
    """
    df = df.copy()
    resultado = []
    for linha in df.itertuples():
        escolha_usuario = linha.categoria if linha.origem in ("", ORIGEM_USUARIO) else ""
        resultado.append(classificar(linha.descricao, linha.tipo, modelo, escolha_usuario))
    df["categoria"] = [r[0] for r in resultado]
    df["origem"] = [r[1] for r in resultado]
    return df
