# Sistema Inteligente de Detecção e Classificação de Placas de Trânsito

AED / Projeto Integrador, 2ª Etapa, Checkpoint 1 (N1)

Pontifícia Universidade Católica de Goiás · Escola Politécnica e de Artes
Ciência de Dados e Inteligência Artificial · CDI1021 Visão Computacional (2026/2)
Prof. Welington Júlio Dias Rodrigues

Pipeline de Processamento Digital de Imagens clássico, sem aprendizado de máquina, que recebe
uma imagem de via, segmenta as placas de sinalização vertical e extrai contagem, área, centroide,
caixa envolvente e classe geométrica de cada uma.

A justificativa completa de cada decisão está no
[relatório técnico](docs/relatorio_tecnico_parcial.pdf) e no próprio notebook. Este README
cobre o que é preciso para rodar, os parâmetros adotados e os resultados.

---

## 1. O problema

O inventário e a fiscalização da sinalização vertical ainda são feitos por vistoria manual em
campo, um processo lento, amostral e difícil de auditar. A dificuldade técnica está em isolar a
placa de uma cena saturada (fachadas, vegetação, veículos), sob contraluz, sombra, desbotamento
e oclusão. Esta etapa resolve essa localização; a identificação do significado da placa fica
para a 2ª Etapa, com aprendizado profundo.

---

## 2. Execução

### Google Colab (recomendado)

1. Abra [`notebooks/checkpoint1_pipeline_pdi.ipynb`](notebooks/checkpoint1_pipeline_pdi.ipynb)
   no Google Colab.
2. Selecione *Runtime → Executar tudo* (`Ctrl+F9`).
3. Quando for solicitado, informe sua chave da API do Roboflow. Ela é gratuita e fica em
   `roboflow.com`, em *Settings → API Keys*.

A primeira célula instala as dependências, e todos os parâmetros são calculados a partir do
próprio dataset. A execução completa leva cerca de 25 minutos.

### Execução local

```bash
git clone https://github.com/CaioHenri99/aed-visao-placas-transito.git
cd aed-visao-placas-transito

python -m venv .venv
# Windows:  .venv\Scripts\activate
# Linux/macOS:  source .venv/bin/activate

pip install -r requirements.txt
jupyter lab notebooks/checkpoint1_pipeline_pdi.ipynb
```

Para não digitar a chave a cada execução, defina-a como variável de ambiente. Ela nunca é
gravada em arquivo:

```bash
export ROBOFLOW_API_KEY="sua-chave"          # Linux/macOS
$env:ROBOFLOW_API_KEY = "sua-chave"          # Windows PowerShell
```

### Sem chave da API

A Seção 1 do notebook oferece três caminhos, e basta um funcionar:

| Caminho | Como usar |
|---|---|
| A. SDK do Roboflow | O padrão. Informe a chave quando o notebook pedir |
| B. Link direto | Em *Download Dataset → YOLOv8 → show download code*, copie a URL para `URL_DOWNLOAD_DIRETO` |
| C. Pasta local | Aponte `PASTA_LOCAL` para uma pasta já baixada ou com fotos próprias |

---

## 3. Estrutura do repositório

```
aed-visao-placas-transito/
├── README.md
├── requirements.txt
├── .github/workflows/validar.yml     # CI: notebook íntegro e sem chave de API
├── notebooks/
│   └── checkpoint1_pipeline_pdi.ipynb
├── docs/
│   ├── relatorio_tecnico_parcial.pdf # relatório técnico do PP1
│   ├── arquitetura.md                # diagrama da solução em Mermaid
│   ├── arquitetura_pipeline.png      # diagrama gerado pelo notebook
│   └── referencias/                  # enunciado e proposta da 1ª Etapa
├── data/                             # dataset baixado pelo notebook (não versionado)
│   └── placas-de-transito-br-wq5tp-1/
└── outputs/                          # gerado pelo notebook e versionado
    ├── parametros_adotados.json      # registro completo e reprodutível da execução
    ├── parametros_adotados.md        # tabela de parâmetros (reproduzida na Seção 6)
    ├── inventario_dataset.csv        # uma linha por imagem: split, dimensões, objetos
    ├── busca_em_grade.csv            # todas as configurações testadas
    ├── avaliacao_por_imagem.csv      # VP, FP, FN e limiar T, imagem a imagem
    ├── descritores_objetos.csv       # saída numérica, um objeto por linha
    └── figuras/                      # evidências visuais
```

---

## 4. Dataset

Base [Placas de Trânsito BR](https://universe.roboflow.com/caios-workspace-01wh5/placas-de-transito-br-wq5tp/dataset/1)
(Roboflow Universe, *Public Domain*), cópia em resolução nativa da
[base original](https://universe.roboflow.com/stefano-tommasini-coelho-euf67/placas-de-transito-br).
São quadros de câmera veicular em rodovias brasileiras, anotados com os códigos do CONTRAN.

| Item | Valor |
|---|---|
| Imagens | 2.963 (train 2.073, valid 593, test 297) |
| Objetos anotados | 2.305, em 68 classes do CONTRAN |
| Dimensão | 1.270 × 636 px na mediana |
| Escala do objeto | placa mediana com cerca de 37 px de lado equivalente |
| Anotação | caixa delimitadora em YOLO; 990 imagens sem anotação ficam fora da avaliação |

**Critério de seleção.** Sinalização brasileira, porque cor, forma e proporção são definidas pelo
CONTRAN e permitem justificar cada parâmetro por norma. Cenas completas, e não recortes da placa.

**A base publicada estava esticada.** Todas as versões da base original aplicam *Resize to
640×640 (Stretch)* sobre imagens de 1.270 × 636, o que espreme a largura pela metade: placa
circular vira elipse deitada. Reexportamos sem o resize. Imagens e anotações são as mesmas, e o
efeito foi medido sem mudar uma linha do pipeline:

| Validação retida | Base esticada | Base nativa |
|---|---|---|
| Precisão | 0,137 | **0,236** |
| F1 | 0,146 | **0,201** |
| Falsos positivos | 278 | **159** |
| Formas ambíguas | 89,4% | **79,8%** |

**Dois tratamentos na leitura dos dados**, sem remover nada da base:
1. 5 rótulos vêm como polígono em vez de caixa; o `ler_rotulos` converte para a caixa envolvente.
2. A divisão entre ajuste e validação é por **trecho de gravação**, e não por imagem, para a
   mesma placa não aparecer nos dois lados. O quadro de validação mais próximo de um de ajuste
   está a 31 s.

**Licença e ética.** As imagens não são redistribuídas: o notebook baixa da fonte e `data/` fica
fora do Git. Imagens de via podem conter pessoas e placas de veículo (LGPD); fotos próprias da
equipe serão anonimizadas antes de publicadas.

---

## 5. O pipeline

```
imagem → redimensionar → corrigir iluminação → suavizar → mapa de evidência cromática
       → limiarizar → morfologia → contornos → descritores
```

| # | Etapa | Técnica |
|---|---|---|
| 1 | Entrada | Largura de 1.248 px, preservando a proporção |
| 2 | Iluminação | CLAHE no canal `L*` do LAB, que corrige a luz sem deslocar a matiz |
| 3 | Suavização | Gaussiano, antes de limiarizar, para o ruído não deslocar o limiar |
| 4 | Evidência de cor | Mapa escalar em HSV: proximidade das matizes normativas × saturação |
| 5 | Limiarização | Global, Otsu, Otsu restrito e adaptativa comparados; global adotada |
| 6 | Morfologia | Abertura, fechamento e preenchimento de buracos |
| 7 | Contornos | `findContours(RETR_EXTERNAL)` sobre a máscara, nunca sobre o Canny |
| 8 | Filtros | Área mínima e máxima, razão de aspecto, extensão e solidez |
| 9 | Descritores | Área, perímetro, centroide, circularidade, solidez, extensão e momentos de Hu |

---

## 6. Parâmetros adotados

Nenhum valor foi escolhido por tentativa e erro. Os parâmetros têm três origens:

- **Escala do objeto:** os kernels saem do lado equivalente das placas anotadas.
- **Histograma:** as faixas de matiz e as portas de saturação saem dos pixels dentro das caixas
  anotadas, comparados com o fundo.
- **Busca em grade:** método de limiarização, corte e limites de área são escolhidos por F1
  numa amostra de ajuste, separada da amostra em que o desempenho é medido.

Duas regras de desempate foram declaradas antes de rodar, porque diferenças menores que 0,005
de F1 cabem no ruído de 250 imagens: uma faixa de cor só fica se ganhar do descarte por mais que
isso, e entre métodos empatados fica o mais simples. Os quatro métodos de limiarização empataram
(global 0,184, Otsu restrito 0,181, Otsu 0,180, adaptativa 0,179), e a global foi adotada.

Execução de 22/09/2026, semente 42, OpenCV 5.0.0, Python 3.14.4:

| Parâmetro | Valor | Origem |
|---|---|---|
| Largura de trabalho | 1.248 px | Resolução nativa da base |
| CLAHE (clip e grade) | 2,0 e 8×8 | Canal `L*` do LAB |
| Suavização | gaussiano 3×3 | `ímpar(0,10 × 19,2) = 3`, mais estreito que a orla da menor placa |
| Faixas de matiz | vermelho H 8 ± 6 e amarelo H 16 ± 6 | Medidas nas anotações, a partir das âncoras do CONTRAN |
| Portas de saturação | vermelho 100 e amarelo 120 | Estágio 0 da busca em grade |
| Faixas descartadas | azul e verde | Verde sem objetos suficientes; azul abaixo da margem de desempate |
| **Kernel de abertura** | **3×3** | `ímpar(0,10 × 19,2) = 3` |
| **Kernel de fechamento** | **9×9** | `ímpar(0,25 × 37,5) = 9` |
| Método de limiarização | global | Empate técnico entre os quatro, desempate pelo mais simples |
| **Limiar** | **96** | Estágio 1b da busca, entre 64, 96 e 128 |
| Limiar T de Otsu (diagnóstico) | média 80,1, de 37 a 138 | Recalculado por imagem; mostra a variação de iluminação |
| **Área mínima de contorno** | **479 px²** | Quantil 0,40 das placas anotadas × 0,45 de preenchimento |
| Área máxima de contorno | 12.887 px² | Quantil 0,95 das placas anotadas |
| Razão de aspecto, extensão, solidez | 0,35 a 2,85 · 0,35 · 0,70 | Rejeitam poste, vegetação e forma côncava |

A tabela é gerada pelo notebook em `outputs/parametros_adotados.md`.

---

## 7. Resultados

Medidos em 250 imagens de validação que não participaram de nenhuma escolha de parâmetro,
com casamento por IoU ≥ 0,30:

| Métrica | Valor |
|---|---|
| Precisão | 0,236 |
| Recall | 0,175 |
| F1 | 0,201 |
| F1 na amostra de ajuste | 0,193 |

O desempenho é baixo, e isso é informação: um detector puramente cromático é a linha de base
contra a qual o modelo treinado da 2ª Etapa será comparado. O erro de contagem (0,87 objeto por
imagem) não serve de medida nesta base, porque 226 das 250 imagens têm exatamente uma placa
anotada e chutar sempre "1" erra só 0,12.

As evidências antes e depois estão em `outputs/figuras/04_pipeline_*.jpg`: seis imagens
sorteadas com semente fixa entre as de validação, sem escolher casos favoráveis.

---

## 8. Divisão de tarefas da equipe

| # | Integrante | Matrícula | Responsabilidade principal | Seções do notebook |
|---|---|---|---|---|
| 1 | Caio Henrique | 20241013700250 | Aquisição e curadoria do dataset, inventário e versionamento no Git | 1 e 2 |
| 2 | Fernanda Andrade | 20241013700048 | Pré-processamento: espaços de cor, iluminação, filtragem e histograma | 3.2 a 3.5 e 6.1 |
| 3 | Alisson Leonardo | 20241013700170 | Segmentação por cor, limiarização, morfologia e contornos | 3.6 a 3.8 e 6 |
| 4 | Vitor Manoel | 20241013700307 | Descritores geométricos, relatório técnico, figuras e documentação | 3.8, 7 a 11 |

A revisão de código é cruzada, para nenhuma parte depender de uma única pessoa.

---

## 9. Limitações conhecidas

1. **Anotação esparsa.** As cenas têm mais placas visíveis do que anotadas, e cada acerto numa
   placa não anotada conta como falso positivo. A precisão de 0,236 é um piso.
2. **Falso positivo de mesma cor.** Lanterna, carro vermelho, solo exposto e grama seca têm matiz
   e saturação de placa.
3. **Placa pequena limita o recall.** Um décimo das placas tem menos de 19 px de lado, mesmo na
   resolução nativa.
4. **Faixas descartadas.** Placas azuis e verdes não têm cobertura nesta execução.
5. **Círculo e octógono sob perspectiva** não se separam por descritor clássico; o código marca
   a forma como ambígua em vez de arbitrar.
6. **Corte fixo não é transferível.** Com outra câmera, o limiar de 96 precisa ser recalibrado.
7. **Sem significado da placa.** O pipeline entrega forma e posição, não a categoria.

Os delineadores (classe `Del`, um quarto dos objetos) não são placa pelo CONTRAN. Medimos o
efeito de excluí-los do gabarito: o F1 iria de 0,201 para 0,202, dentro da margem de desempate,
então o gabarito foi mantido inteiro.

---

## 10. Próximos passos (2ª Etapa)

Um detector treinado da família YOLO, usando as próprias classes do CONTRAN já anotadas na base,
substitui a decisão por cor e forma. O ganho esperado está nos falsos positivos cromáticos e nas
placas desbotadas. Escala e anotação esparsa continuam sendo limites e precisam de tratamento
explícito: entrada em resolução maior e revisão das anotações no subconjunto de avaliação.

O pipeline clássico permanece como pré-processamento (correção de iluminação), pós-processamento
(filtro por área) e linha de base comparativa.
