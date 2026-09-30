# 💰 FinanAI — Para onde foi meu salário?

Protótipo acadêmico do **Desafio 2 — Dinheiro** da atividade *"Automatize meu dia: IA resolvendo problemas reais"*.

O FinanAI recebe o salário e as movimentações financeiras de uma pessoa. Com isso, ele **classifica as despesas**, **identifica padrões**, **compara com o orçamento** e gera **alertas e recomendações explicáveis**.

> ⚠️ Todos os dados do projeto são **fictícios**. Não use dados bancários reais.

---

## 1. Como executar (Windows)

Pré-requisito: **Python 3.10 ou superior** instalado.

```powershell
cd "$env:USERPROFILE\Downloads\finanai"
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

O navegador abre sozinho em `http://localhost:8501`. Para encerrar, aperte `Ctrl + C` no terminal.
Depois de instalar a primeira vez, só o último comando é necessário. O sistema funciona **sem internet**.

---

## 2. Estrutura do projeto

```
finanai/
├── app.py                    # Interface (Streamlit): páginas, formulários, gráficos
├── requirements.txt
├── README.md
├── .streamlit/config.toml    # Cores do tema
├── data/
│   ├── movimentacoes.csv     # Dados fictícios: agosto e setembro/2026
│   └── treino_categorias.csv # Exemplos rotulados para treinar o ML
└── src/
    ├── dados.py              # ENTRADA: leitura de CSV, validação, formato R$
    ├── classificacao.py      # CLASSIFICAÇÃO: regras + Machine Learning
    ├── analise.py            # ANÁLISE e ALERTAS: indicadores, padrões, orçamento
    └── recomendacoes.py      # SAÍDA: recomendações explicáveis
```

## 3. Arquitetura (fluxo)

```
ENTRADA           salário, cadastro manual, CSV, dados fictícios         (dados.py)
   ↓
PROCESSAMENTO     validação, datas, valores, mês                          (dados.py / analise.py)
   ↓
CLASSIFICAÇÃO     1º usuário → 2º regras → 3º ML → senão "Revisar"        (classificacao.py)
   ↓
ANÁLISE           totais, variação mensal, recorrências, atípicas         (analise.py)
   ↓
DECISÃO           alertas: orçamento, % salário, saldo, média             (analise.py)
   ↓
SAÍDA             dashboard, gráficos, alertas, recomendações             (app.py / recomendacoes.py)
```

Cada arquivo tem uma responsabilidade só. A interface (`app.py`) não sabe *como* a classificação é feita: ela apenas chama `classificar()`. Por isso dá para trocar as regras e o ML por uma IA generativa no futuro sem mexer no resto.

---

## 4. Onde está a AUTOMAÇÃO (regras simples, **não** é IA)

| O quê | Como funciona |
|---|---|
| Classificação por palavra-chave | "uber" → Transporte, "netflix" → Assinaturas (`PALAVRAS_CHAVE`) |
| Totais e percentuais | Somas por categoria e por mês |
| Orçamento | `gasto > limite` → alerta com o valor excedido |
| Uso do salário | Despesas ≥ 80% do salário → alerta |
| Saldo baixo | Saldo < 10% das receitas → alerta |
| Variação mensal | Categoria subiu 15% ou mais em relação ao mês anterior |
| Recorrências | Mesma descrição, 1 vez por mês, em 2+ meses, valor com variação de até ±5% |
| Despesa atípica | Valor > média + 2 desvios-padrão da categoria (estatística simples) |
| Recomendações | Regras "se → então", e cada uma mostra em que dado se baseou |

## 5. Onde está o MACHINE LEARNING

Fica em `src/classificacao.py`, classe `ClassificadorML`:

- **TF-IDF** transforma a descrição em números a partir de pedaços de 2 a 4 letras.
- **Naive Bayes** aprende, a partir de ~100 exemplos rotulados, qual categoria cada pedaço indica.
- O modelo **só é usado quando nenhuma regra reconhece o texto**. Exemplo: "Hamburgueria Artesanal" não tem regra, mas o ML reconhece como Alimentação (79%).
- Se a confiança ficar **abaixo de 40%**, a despesa vai para **Outros/Revisar**, e uma pessoa decide. Exemplo: "Pix para Carlos".
- **Aprende com o usuário:** quando alguém corrige uma movimentação em *Revisão humana*, o exemplo entra no treino e o modelo é re-treinado na hora.

**Por que ML aqui?** Regras não conseguem prever todas as descrições possíveis. O ML generaliza a partir de exemplos.

## 6. Onde a IA (generativa) seria necessária — **não implementada**

- Entender descrições bem confusas de extrato ("PAG*JOSE123", "COMPRA CARTAO 4432").
- Um **chat** para perguntas em linguagem natural ("quanto gastei com comida nas últimas 2 semanas?").
- Explicar o mês em texto corrido, personalizado.

Ela não entrou no protótipo porque precisaria de internet/API e o problema principal já é resolvido com regras + ML simples.

## 7. Limitações do protótipo

- Os dados ficam na memória: ao fechar o navegador, as alterações se perdem (use *Baixar CSV* para guardar).
- A base de treino é pequena (~100 exemplos), então o ML erra. Exemplo: acha que "Barbearia" é Alimentação, com confiança baixa.
- As regras de palavra-chave podem falhar com nomes ambíguos.
- A comparação mensal e as recorrências precisam de pelo menos 2 meses de dados.
- As recomendações são genéricas: não conhecem a realidade da pessoa e **não são aconselhamento financeiro**.
- Não há login, criptografia nem integração bancária (Open Finance).

---

## 8. Roteiro de demonstração (~5 minutos)

| Tempo | Ação | O que falar |
|---|---|---|
| 0:00 | Abrir o sistema no **Dashboard** | "Esta é a Ana (fictícia), salário de R$ 4.500. Ela não sabe para onde o dinheiro vai." |
| 0:30 | Mostrar os cards e o gráfico por categoria | "Já usou 88% do salário; Moradia e Alimentação são os maiores gastos." |
| 1:00 | Ir em **Movimentações** e mostrar a tabela e a coluna *Classificado por* | "Antes ela fazia isso na mão, no extrato. Aqui cada gasto é classificado automaticamente." |
| 1:30 | Clicar em **⚡ Demonstração: adicionar iFood (R$ 150)** | "A regra reconheceu 'iFood' → Alimentação. Isso é **automação**, não IA." |
| 2:00 | Em *Testar a classificação*, digitar **"Hamburgueria do Centro"** | "Nenhuma regra conhece essa palavra, mas o **Machine Learning** reconheceu com 84%." |
| 2:30 | Voltar ao **Dashboard** | "O dashboard atualizou. Alimentação passou do orçamento: **R$ 927,75 de R$ 800**. Apareceu também o alerta de saldo baixo." |
| 3:00 | **Análise Inteligente** | "Aqui aparecem os padrões: Compras subiu 109%, 4 assinaturas, 10 despesas recorrentes." |
| 3:45 | **Recomendações** | "Cada recomendação diz em que dados se baseou. É explicável, e a decisão final é da pessoa." |
| 4:15 | **Sobre a solução** → Privacidade | "Dados financeiros são sensíveis; os riscos são erro de classificação e viés; o protótipo roda local e usa dados fictícios." |
| 4:45 | Encerrar | "Automação resolve o básico, ML entra onde as regras falham e o humano decide." |

Extra, se sobrar tempo: em *Movimentações*, cadastrar **"Açaí da esquina"**. Ele vai para *Revisar*. Corrija para Alimentação e depois teste **"Açaí do Parque"**: o ML aprendeu com a correção.

Antes de começar, clique em **🔄 Restaurar dados de demonstração** para voltar ao estado inicial.

---

## 9. As 7 perguntas da atividade

1. **Qual é exatamente o problema?** A pessoa tem dezenas de movimentações no mês e não enxerga para onde o dinheiro vai, nem percebe a tempo quando está gastando demais.
2. **O que hoje é feito manualmente?** Ler o extrato linha por linha, anotar numa planilha, escolher a categoria, somar, comparar com o mês anterior e lembrar dos limites.
3. **O que pode ser automatizado?** Classificação por palavra-chave, somas, comparação com orçamento e com o mês anterior, recorrências e alertas (seção 4).
4. **Onde IA realmente seria necessária?** Em descrições ambíguas e numa interface de conversa em linguagem natural (seção 6).
5. **Onde aprendizado de máquina aparece?** Na classificação de descrições desconhecidas (TF-IDF + Naive Bayes), que aprende com as correções. No futuro, também na previsão do gasto do mês e na detecção de anomalias (seção 5).
6. **Quais dados seriam necessários?** Data, descrição, valor e tipo das movimentações, salário, limites por categoria e exemplos rotulados. Eles viriam de cadastro manual, de CSV exportado do banco ou, futuramente, do Open Finance com consentimento.
7. **Riscos, erros, vieses e privacidade?** Classificação incorreta, recomendações inadequadas, dados incompletos, viés dos exemplos de treino e exposição de dados sensíveis (LGPD). Por isso existem a revisão humana, a explicação de cada recomendação, o processamento local e o uso de dados fictícios.

**Decisão que continua humana:** corrigir categorias duvidosas e decidir se corta ou mantém um gasto. O sistema só informa e sugere.

---

## 10. Uso de IA no desenvolvimento (declaração)

Usamos IA generativa (Claude) para gerar a base do código, os dados fictícios e a documentação. A equipe executou, testou e revisou o sistema e é responsável por explicá-lo.
