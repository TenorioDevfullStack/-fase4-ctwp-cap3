"""Gera o PDF de entrega (1 pagina) com integrantes, RM e link do repositorio."""
import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.backends.backend_pdf import PdfPages

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOGO = os.path.join(ROOT, "assets", "logo-fiap.png")
OUT = os.path.join(ROOT, "Entrega_FIAP_classificacao_graos.pdf")

REPO = "https://github.com/TenorioDevfullStack/-fase4-ctwp-cap3"
integrantes = [
    ("Pedro Henrique Lima Schneider", "573999"),
    ("Leandro Tenório da Silva", "572083"),
    ("Diego Alexandre Lemos Nóbrega", "572085"),
    ("João Pedro Lurovschi de Almeida Bessa", "570160"),
    ("Nícolas Xavier Costa", "570336"),
]

fig = plt.figure(figsize=(8.27, 11.69))  # A4 retrato
fig.patch.set_facecolor("white")
ax = fig.add_axes([0, 0, 1, 1]); ax.axis("off")

COR = "#ED145B"  # magenta FIAP

# Logo FIAP centralizado no topo
try:
    logo = mpimg.imread(LOGO)
    h, w = logo.shape[0], logo.shape[1]
    lw = 0.30
    lh = lw * (h / w) * (8.27 / 11.69)
    logo_ax = fig.add_axes([0.5 - lw / 2, 0.86, lw, lh]); logo_ax.axis("off")
    logo_ax.imshow(logo)
except Exception as e:
    print("logo nao carregada:", e)

def txt(y, s, size=11, weight="normal", color="black", style="normal",
        x=0.5, ha="center", url=None):
    return ax.text(x, y, s, fontsize=size, fontweight=weight, color=color,
                   style=style, ha=ha, va="center", transform=ax.transAxes,
                   wrap=True, url=url)

# Cabecalho institucional
txt(0.835, "FIAP — Faculdade de Informática e Administração Paulista",
    size=10.5, color="#444444")
ax.plot([0.12, 0.88], [0.815, 0.815], color=COR, lw=2, transform=ax.transAxes)

# Titulo da atividade
txt(0.775, "Atividade “Ir Além”", size=12, weight="bold", color=COR)
txt(0.745, "Da Terra ao Código: Automatizando a Classificação", size=15, weight="bold")
txt(0.720, "de Grãos com Machine Learning", size=15, weight="bold")
txt(0.690, "Classificação de variedades de grãos de trigo (Kama, Rosa, Canadian)",
    size=10.5, style="italic", color="#555555")
txt(0.672, "com metodologia CRISP-DM", size=10.5, style="italic", color="#555555")

# Bloco integrantes
txt(0.610, "Integrantes do grupo", size=13, weight="bold", color=COR)

# Cabecalho da tabela
yt = 0.575
txt(yt, "Aluno", size=11, weight="bold", x=0.20, ha="left")
txt(yt, "RM", size=11, weight="bold", x=0.78, ha="left")
ax.plot([0.18, 0.86], [yt - 0.012, yt - 0.012], color="#cccccc", lw=1, transform=ax.transAxes)

y = yt - 0.040
for nome, rm in integrantes:
    txt(y, nome, size=11, x=0.20, ha="left")
    txt(y, rm, size=11, x=0.78, ha="left")
    y -= 0.034

# Bloco repositorio
y -= 0.030
txt(y, "Repositório do projeto (GitHub)", size=13, weight="bold", color=COR)
y -= 0.040
txt(y, REPO, size=11.5, color="#0645AD", weight="bold", url=REPO)
y -= 0.028
txt(y, "Repositório público — código-fonte, notebook executado e documentação.",
    size=9.5, style="italic", color="#555555")

# Resumo
y -= 0.055
txt(y, "Resumo", size=13, weight="bold", color=COR)
resumo = (
    "Projeto que aplica a metodologia CRISP-DM para automatizar a classificação de\n"
    "grãos de trigo a partir de 7 medições físicas (Seeds Dataset / UCI, 210 amostras).\n"
    "Foram comparados cinco algoritmos — KNN, SVM, Random Forest, Naive Bayes e\n"
    "Regressão Logística — com avaliação por acurácia, precisão, recall, F1-score e\n"
    "matrizes de confusão, além de otimização de hiperparâmetros via Grid Search.\n"
    "Melhor modelo: Random Forest (~90–92% de acurácia)."
)
y -= 0.045
ax.text(0.12, y, resumo, fontsize=10.3, color="#222222", ha="left", va="top",
        transform=ax.transAxes, linespacing=1.6)

# Rodape
ax.plot([0.12, 0.88], [0.055, 0.055], color="#dddddd", lw=1, transform=ax.transAxes)
txt(0.035, "Entrega gerada para submissão no Portal do Aluno — FIAP",
    size=8.5, color="#888888")

with PdfPages(OUT) as pdf:
    pdf.savefig(fig)
plt.close(fig)
print("PDF gerado em:", OUT)
print("Tamanho:", os.path.getsize(OUT), "bytes")
