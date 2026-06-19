<img src="../assets/logo-fiap.png" alt="FIAP - Faculdade de Informática e Admnistração Paulista" border="0" width=30% height=30%>

# AI Project Document - FIAP

## Nome do Grupo

> _Preencha com o nome do seu grupo._

#### Nomes dos integrantes do grupo

- Pedro Henrique Lima Schneider — RM 573999
- Leandro Tenório da Silva — RM 572083
- Diego Alexandre Lemos Nóbrega — RM 572085
- João Pedro Lurovschi de Almeida Bessa — RM 570160
- Nícolas Xavier Costa — RM 570336


## Sumário

[1. Introdução](#c1)

[2. Visão Geral do Projeto](#c2)

[3. Desenvolvimento do Projeto](#c3)

[4. Resultados e Avaliações](#c4)

[5. Conclusões e Trabalhos Futuros](#c5)

[6. Referências](#c6)

[Anexos](#c7)

<br>

# <a name="c1"></a>1. Introdução

## 1.1. Escopo do Projeto

### 1.1.1. Contexto da Inteligência Artificial

A Inteligência Artificial, em especial o **Aprendizado de Máquina (Machine Learning)**,
vem transformando o agronegócio por meio da chamada **Agricultura de Precisão**. Tarefas
historicamente manuais — inspeção, triagem e classificação de produtos — passam a ser
automatizadas por modelos que aprendem padrões a partir de dados, reduzindo custo, tempo
e erro humano. No segmento de **classificação de grãos**, técnicas de visão computacional
e modelos de classificação supervisionada já são aplicadas mundialmente para padronizar a
qualidade de lotes e dar escala a operações que dependeriam de especialistas. A abrangência
é ampla — de grandes tradings internacionais a **pequenas cooperativas regionais**, foco
deste trabalho.

### 1.1.2. Descrição da Solução Desenvolvida

Desenvolvemos um **classificador automático de variedades de grãos de trigo** (Kama, Rosa
e Canadian) a partir de sete medições físicas do grão (área, perímetro, compacidade,
comprimento e largura do núcleo, coeficiente de assimetria e comprimento do sulco). A
solução substitui a triagem manual feita por especialistas em cooperativas de pequeno
porte, entregando uma classificação **rápida, consistente e reprodutível**. O modelo final
(Random Forest) atinge cerca de **90–92% de acurácia**, e o fluxo prevê encaminhamento dos
casos de baixa confiança para conferência humana (*human-in-the-loop*).

# <a name="c2"></a>2. Visão Geral do Projeto

## 2.1. Objetivos do Projeto

- **Objetivo de negócio:** reduzir o tempo e o erro da classificação manual de grãos de
  trigo em cooperativas de pequeno porte.
- **Objetivo técnico:** treinar um modelo de **classificação supervisionada multiclasse**
  que, a partir de 7 atributos físicos, prediga a variedade do grão.
- **Critério de sucesso:** acurácia elevada (referência ≥ 90%) com bom equilíbrio entre
  precisão e recall nas três variedades.

## 2.2. Público-Alvo

Cooperativas agrícolas de **pequeno porte** e seus especialistas em classificação de
grãos, que hoje executam a triagem manualmente. Beneficiam-se também produtores e a cadeia
de comercialização, que ganham padronização e rastreabilidade na qualidade dos lotes.

## 2.3. Metodologia

Adotamos o **CRISP-DM** (*Cross Industry Standard Process for Data Mining*), com suas seis
fases: Entendimento do Negócio, Entendimento dos Dados, Preparação dos Dados, Modelagem,
Avaliação e Implantação. Cada fase está implementada e documentada no notebook
`src/classificacao_graos_trigo.ipynb`.

# <a name="c3"></a>3. Desenvolvimento do Projeto

## 3.1. Tecnologias Utilizadas

- **Linguagem:** Python 3.
- **Manipulação de dados:** pandas, numpy.
- **Visualização:** matplotlib, seaborn.
- **Machine Learning:** scikit-learn (modelos, pré-processamento, métricas, Grid Search).
- **Ambiente:** Jupyter Notebook (compatível com Google Colab).

## 3.2. Modelagem e Algoritmos

Comparamos **cinco** algoritmos de famílias distintas para garantir uma avaliação ampla:

| Algoritmo | Família | Justificativa |
|-----------|---------|---------------|
| **KNN** | Baseado em distância | Eficaz quando as classes formam grupos no espaço |
| **SVM** | Margem máxima / kernel | Forte em fronteiras não-lineares com poucos dados |
| **Random Forest** | Ensemble de árvores | Robusto, lida com outliers e fornece importância das features |
| **Naive Bayes** | Probabilístico | *Baseline* rápido |
| **Regressão Logística** | Linear | *Baseline* interpretável |

Algoritmos sensíveis a escala (KNN, SVM, Regressão Logística) receberam **padronização
Z-score**. Na otimização, cada modelo foi encapsulado em um `Pipeline` (scaler + modelo)
para que a padronização fosse reajustada dentro de cada *fold*, evitando *data leakage*.

## 3.3. Treinamento e Teste

- **Conjunto de dados:** Seeds Dataset (UCI), 210 amostras, 7 atributos, 3 classes
  balanceadas; sem valores ausentes ou duplicatas.
- **Divisão:** 70% treino / 30% teste, **estratificada** por classe.
- **Otimização:** Grid Search com validação cruzada estratificada (5 folds), métrica de
  otimização **F1-score macro**.
- **Métricas de avaliação:** acurácia, precisão, recall, F1-score (macro) e matriz de
  confusão por modelo.

# <a name="c4"></a>4. Resultados e Avaliações

## 4.1. Análise dos Resultados

Desempenho no conjunto de teste:

| Modelo | Acurácia (base) | Acurácia (otimizado) | F1 base → otim. |
|--------|:---:|:---:|:---:|
| **Random Forest** | **0,921** | **0,905** | 0,919 → 0,903 |
| KNN | 0,873 | 0,889 | 0,871 → 0,888 |
| Regressão Logística | 0,857 | 0,889 | 0,854 → 0,888 |
| SVM | 0,873 | 0,857 | 0,871 → 0,854 |
| Naive Bayes | 0,825 | 0,825 | 0,825 → 0,825 |

- **Random Forest foi o melhor modelo**, atingindo o critério de sucesso (≥ 90%).
- O **Naive Bayes** teve o pior desempenho, coerente com sua premissa de independência
  entre atributos — violada pela forte correlação entre as medidas de tamanho.
- A **otimização teve efeito misto**: melhorou KNN e Regressão Logística, mas piorou
  levemente Random Forest e SVM no teste. Os scores de validação cruzada no treino foram
  altos (0,91–0,96), mas não se traduziram integralmente no teste de 63 amostras — efeito
  esperado em **datasets pequenos** e uma lição metodológica importante (mais
  hiperparâmetro não garante mais acerto).
- **Onde os modelos erram:** os erros concentram-se na variedade **Kama** (recall ~0,76 no
  melhor modelo), frequentemente confundida com **Canadian**. A Kama possui valores
  **intermediários** de tamanho, ficando geometricamente entre as outras duas — exatamente
  como antecipado na análise exploratória. A **Rosa** (grãos grandes) é a mais distinta e
  quase nunca confundida.
- **Características mais relevantes** (Random Forest): área (0,232), perímetro (0,218) e
  comprimento do sulco (0,177) — todas medidas de tamanho. Compacidade e coeficiente de
  assimetria contribuem menos, mas agregam o sinal de **forma**, útil nos casos de fronteira.

## 4.2. Feedback dos Usuários

Não houve avaliação com usuários finais nesta etapa (projeto acadêmico). Como trabalho
futuro, propõe-se um piloto em uma cooperativa real, coletando *feedback* dos especialistas
sobre os casos encaminhados para conferência manual.

# <a name="c5"></a>5. Conclusões e Trabalhos Futuros

A metodologia CRISP-DM demonstrou ser **viável automatizar a classificação de grãos de
trigo** com Machine Learning: o melhor modelo atingiu ~90–92% de acurácia em um conjunto
pequeno (210 amostras), com erros restritos à fronteira naturalmente ambígua entre Kama e
Canadian.

**Pontos fortes:** processo bem estruturado (CRISP-DM), comparação ampla de algoritmos,
metodologia sem *data leakage* e interpretação honesta dos resultados.

**Pontos a melhorar / trabalhos futuros:**
1. **Coletar mais dados** — 210 amostras é pouco; mais dados estabilizariam as métricas e
   estreitariam a fronteira Kama/Canadian.
2. **Fluxo human-in-the-loop** — usar `predict_proba` para enviar predições de baixa
   confiança ao especialista.
3. **Serialização e deploy** — empacotar o pipeline (scaler + modelo) com `joblib` e
   expô-lo via API ou aplicação simples na cooperativa.
4. **Engenharia de atributos** — explorar combinações de forma que ajudem a separar a Kama.

# <a name="c6"></a>6. Referências

- UCI Machine Learning Repository — *Seeds Dataset*:
  https://archive.ics.uci.edu/dataset/236/seeds
- M. Charytanowicz et al., "A Complete Gradient Clustering Algorithm for Features Analysis
  of X-ray Images", 2010 (origem do dataset).
- scikit-learn: https://scikit-learn.org
- CRISP-DM: *Cross Industry Standard Process for Data Mining*.

# <a name="c7"></a>Anexos

Todos os gráficos (histogramas, boxplots, pairplot, matriz de correlação, comparações de
modelos e matrizes de confusão) e o código completo encontram-se no notebook
`src/classificacao_graos_trigo.ipynb`, já executado com as saídas salvas.
