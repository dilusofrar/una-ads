"""
ENTRADA DE DADOS: leitura, validação e formatação das movimentações.
"""
from datetime import date
from pathlib import Path

import pandas as pd

PASTA_DADOS = Path(__file__).resolve().parent.parent / "data"
ARQUIVO_DEMO = PASTA_DADOS / "movimentacoes.csv"
ARQUIVO_TREINO = PASTA_DADOS / "treino_categorias.csv"

TIPOS = ["Despesa", "Receita"]
COLUNAS_OBRIGATORIAS = ["data", "descricao", "valor", "tipo"]
COLUNAS = ["data", "descricao", "valor", "tipo", "categoria", "origem"]


def formatar_reais(valor: float) -> str:
    """1234.5 -> 'R$ 1.234,50'"""
    texto = f"{abs(valor):,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")
    return f"-R$ {texto}" if valor < 0 else f"R$ {texto}"


def tabela_vazia() -> pd.DataFrame:
    return pd.DataFrame(columns=COLUNAS)


def validar_movimentacao(data_mov, descricao: str, valor, tipo: str) -> list[str]:
    """Devolve a lista de erros encontrados (lista vazia = dados válidos)."""
    erros = []
    if not isinstance(data_mov, date):
        erros.append("Informe uma data válida.")
    if not descricao or len(descricao.strip()) < 2:
        erros.append("A descrição precisa ter pelo menos 2 caracteres.")
    try:
        if float(valor) <= 0:
            erros.append("O valor precisa ser maior que zero.")
    except (TypeError, ValueError):
        erros.append("O valor precisa ser um número.")
    if tipo not in TIPOS:
        erros.append("O tipo precisa ser 'Receita' ou 'Despesa'.")
    return erros


def ler_csv(origem) -> tuple[pd.DataFrame, int]:
    """
    Lê um CSV de movimentações (caminho ou arquivo enviado pelo usuário).
    Linhas inválidas são descartadas. Retorna (tabela, quantidade_descartada).
    Formato esperado: data,descricao,valor,tipo[,categoria]
    """
    try:
        df = pd.read_csv(origem, dtype=str, encoding="utf-8")
    except Exception as erro:
        raise ValueError(f"Não foi possível ler o arquivo: {erro}") from erro

    df.columns = [c.strip().lower() for c in df.columns]
    faltando = [c for c in COLUNAS_OBRIGATORIAS if c not in df.columns]
    if faltando:
        raise ValueError(f"Colunas obrigatórias ausentes no CSV: {', '.join(faltando)}")

    total_linhas = len(df)
    df["data"] = pd.to_datetime(df["data"], errors="coerce").dt.date
    df["valor"] = pd.to_numeric(df["valor"].str.replace(",", "."), errors="coerce")
    df["descricao"] = df["descricao"].fillna("").str.strip()
    df["tipo"] = df["tipo"].fillna("").str.strip().str.capitalize()
    if "categoria" not in df.columns:
        df["categoria"] = ""
    df["categoria"] = df["categoria"].fillna("").str.strip()

    validas = (
        df["data"].notna()
        & (df["valor"] > 0)
        & (df["descricao"].str.len() >= 2)
        & df["tipo"].isin(TIPOS)
    )
    df = df[validas].reset_index(drop=True)
    df["origem"] = ""
    return df[COLUNAS], total_linhas - len(df)


def ler_treino() -> pd.DataFrame:
    """Exemplos rotulados (descrição -> categoria) usados para treinar o modelo de ML."""
    return pd.read_csv(ARQUIVO_TREINO, encoding="utf-8")
