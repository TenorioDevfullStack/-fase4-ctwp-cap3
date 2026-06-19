# FIAP - Faculdade de Informática e Administração Paulista

<p align="center">
<a href= "https://www.fiap.com.br/"><img src="assets/logo-fiap.png" alt="FIAP - Faculdade de Informática e Admnistração Paulista" border="0" width=40% height=40%></a>
</p>

<br>

# Da Terra ao Código: Automatizando a Classificação de Grãos com Machine Learning

## Nome do grupo

> _Preencha com o nome do seu grupo._

## 👨‍🎓 Integrantes:
- <a href="https://www.linkedin.com/company/inova-fusca">Pedro Henrique Lima Schneider — RM 573999</a>
- <a href="https://www.linkedin.com/company/inova-fusca">Leandro Tenório da Silva — RM 572083</a>
- <a href="https://www.linkedin.com/company/inova-fusca">Diego Alexandre Lemos Nóbrega — RM 572085</a>
- <a href="https://www.linkedin.com/company/inova-fusca">João Pedro Lurovschi de Almeida Bessa — RM 570160</a>
- <a href="https://www.linkedin.com/company/inova-fusca">Nícolas Xavier Costa — RM 570336</a>

## 👩‍🏫 Professores:
### Tutor(a)
- <a href="https://www.linkedin.com/company/inova-fusca">Nome do Tutor</a>
### Coordenador(a)
- <a href="https://www.linkedin.com/company/inova-fusca">Nome do Coordenador</a>


## 📜 Descrição

Em cooperativas agrícolas de pequeno porte, a classificação dos grãos de trigo é feita
**manualmente por especialistas** — um processo demorado, subjetivo e sujeito a erros
humanos. Este projeto aplica a metodologia **CRISP-DM** para desenvolver um modelo de
**Machine Learning** que classifica automaticamente três variedades de trigo
(**Kama**, **Rosa** e **Canadian**) a partir de sete medições físicas dos grãos,
aumentando a eficiência e a precisão da triagem.

Utilizamos o **Seeds Dataset** do *UCI Machine Learning Repository* (210 amostras, 70 de
cada variedade, perfeitamente balanceado). O trabalho percorre todas as fases do
CRISP-DM:

1. **Business Understanding** — definição do problema da cooperativa e do critério de sucesso.
2. **Data Understanding** — estatísticas descritivas (média, mediana, desvio-padrão),
   histogramas, boxplots, gráficos de dispersão (pairplot) e matriz de correlação.
3. **Data Preparation** — tratamento de ausentes, divisão treino/teste **70/30
   estratificada** e **padronização** (`StandardScaler`).
4. **Modeling** — treino e comparação de **cinco** algoritmos: **KNN, SVM, Random Forest,
   Naive Bayes e Regressão Logística**, avaliados com acurácia, precisão, recall, F1-score
   e matrizes de confusão.
5. **Evaluation/Optimization** — **Grid Search** com validação cruzada estratificada
   (5 folds) para otimização de hiperparâmetros, evitando *data leakage* via *pipelines*.
6. **Deployment** — recomendações de uso na cooperativa.

### Principais resultados

| Modelo | Acurácia (base) | Acurácia (otimizado) |
|--------|:---:|:---:|
| **Random Forest** | **0,921** | **0,905** |
| KNN | 0,873 | 0,889 |
| Regressão Logística | 0,857 | 0,889 |
| SVM | 0,873 | 0,857 |
| Naive Bayes | 0,825 | 0,825 |

- **Melhor modelo: Random Forest** (~90–92% de acurácia), robusto e interpretável.
- Erros concentrados na variedade **Kama** (valores intermediários de tamanho),
  confundida com Canadian — consistente com a análise exploratória.
- Características mais importantes: **área, perímetro e comprimento do sulco**.
- A otimização teve efeito **misto** (melhorou KNN e Regressão Logística, piorou
  levemente RF e SVM no teste) — efeito esperado em um conjunto de teste pequeno
  (63 amostras), uma lição metodológica relevante.

O relatório completo, com todos os gráficos e a interpretação detalhada, está no notebook
[`src/classificacao_graos_trigo.ipynb`](src/classificacao_graos_trigo.ipynb).


## 📁 Estrutura de pastas

Dentre os arquivos e pastas presentes na raiz do projeto, definem-se:

- <b>.github</b>: arquivos de configuração específicos do GitHub.

- <b>assets</b>: elementos não-estruturados, como imagens (ex.: logo da FIAP).

- <b>config</b>: arquivos de configuração de parâmetros do projeto.

- <b>document</b>: documentos do projeto. Contém o `ai_project_document_fiap.md` com a
  documentação detalhada da solução de IA.

- <b>scripts</b>: scripts auxiliares.

- <b>src</b>: todo o código-fonte do projeto:
  - `classificacao_graos_trigo.ipynb` — **notebook principal** (CRISP-DM completo, já
    executado com saídas e gráficos);
  - `build_notebook.py` — script que gera o notebook de forma reprodutível;
  - `requirements.txt` — dependências Python;
  - `seeds/seeds_dataset.txt` — Seeds Dataset original do UCI.

- <b>README.md</b>: este arquivo, guia geral do projeto.


## 🔧 Como executar o código

**Pré-requisitos:** Python 3.10+ e as bibliotecas listadas em `src/requirements.txt`
(pandas, numpy, scikit-learn, matplotlib, seaborn, jupyter).

### 1. Instalar dependências
```bash
cd src
pip install -r requirements.txt
```

### 2. Abrir e executar o notebook
Abra `src/classificacao_graos_trigo.ipynb` no **Jupyter**, **VS Code** ou **Google Colab**
e execute todas as células (Kernel → *Restart & Run All*). O notebook já vem com as
saídas salvas, portanto também pode ser apenas lido.

> Ao executar, mantenha o diretório de trabalho em `src/`, pois o notebook lê o dataset
> pelo caminho relativo `seeds/seeds_dataset.txt`.

### 3. (Opcional) Regenerar o notebook do zero
```bash
cd src
python build_notebook.py    # recria o .ipynb sem saídas
```


## 🗃 Histórico de lançamentos

* 1.0.0 - 19/06/2026
    * Entrega final: CRISP-DM completo, 5 algoritmos, otimização por Grid Search,
      interpretação dos resultados e documentação.


## 📋 Licença

<img style="height:22px!important;margin-left:3px;vertical-align:text-bottom;" src="https://mirrors.creativecommons.org/presskit/icons/cc.svg?ref=chooser-v1"><img style="height:22px!important;margin-left:3px;vertical-align:text-bottom;" src="https://mirrors.creativecommons.org/presskit/icons/by.svg?ref=chooser-v1"><p xmlns:cc="http://creativecommons.org/ns#" xmlns:dct="http://purl.org/dc/terms/"><a property="dct:title" rel="cc:attributionURL" href="https://github.com/agodoi/template">MODELO GIT FIAP</a> por <a rel="cc:attributionURL dct:creator" property="cc:attributionName" href="https://fiap.com.br">Fiap</a> está licenciado sobre <a href="http://creativecommons.org/licenses/by/4.0/?ref=chooser-v1" target="_blank" rel="license noopener noreferrer" style="display:inline-block;">Attribution 4.0 International</a>.</p>
