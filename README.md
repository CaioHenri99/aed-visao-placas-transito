# Sistema Inteligente de Detecção e Classificação de Placas de Trânsito

AED / Projeto Integrador · Checkpoint 1 (N1) e Checkpoint 2 (N2)

Pontifícia Universidade Católica de Goiás · Escola Politécnica e de Artes
Ciência de Dados e Inteligência Artificial · CDI1021 Visão Computacional (2026/2)
Prof. Welington Júlio Dias Rodrigues

O sistema recebe uma foto de câmera veicular e devolve as placas de trânsito que aparecem nela:
onde está cada uma e qual é, pelo código do CONTRAN. Na N1, a decisão era de um pipeline clássico
de Processamento Digital de Imagens, por cor e forma. Na N2, ela passa para um detector treinado
(YOLO26), e o pipeline da N1 vira a linha de base contra a qual o ganho é medido.

| Etapa | Notebook | O que entrega |
|---|---|---|
| N2 · Checkpoint 2 | [`checkpoint2_modelo.ipynb`](notebooks/checkpoint2_modelo.ipynb) | Divisão sem vazamento, linha de base medida, modelo treinado e diagnosticado, pipeline de inferência ([`inferir.py`](inferir.py)) |
| N1 · Checkpoint 1 | [`checkpoint1_pipeline_pdi.ipynb`](notebooks/checkpoint1_pipeline_pdi.ipynb) | Pipeline clássico e seus parâmetros, justificados no [relatório técnico](docs/relatorio_tecnico_parcial.pdf) |

----

## N2 · Modelo treinado

### Resultado no teste

O teste são as 403 imagens de dois dias de captura (10/10 e 11/10), com 263 placas anotadas,
que não participaram de nenhuma escolha. O limiar de confiança do modelo (0,41) foi fixado na
validação. Linha de base e modelo são medidos com o mesmo gabarito e o mesmo casamento por IoU:

| | Linha de base (N1) | Modelo (YOLO26s) |
|---|---|---|
| Placas encontradas, IoU ≥ 0,50 | 31 de 263 | **229 de 263** |
| Detecções sem placa anotada | 282 | 150 |
| F1 de detecção, IoU ≥ 0,50 | 0,108 | **0,713** |
| F1 exigindo também a classe certa | não classifica | **0,639** |
| mAP50 · mAP50-95 | não se aplica | 0,736 · 0,478 |

Na validação, o mAP50 era 0,770: o teste confirmou a validação. A maior parte das detecções "sem
placa anotada" são placas reais que a base não anotou (49 de 54 numa amostra do teste, Seção 9.3
do notebook), então a precisão medida, de 0,604, é um piso.

### Como foi feito

- **Divisão por dia de captura**, e não por imagem: os 8 dias de gravação vão inteiros para treino
  (5 dias, 2.161 imagens), validação (28/09, 399) ou teste (10/10 e 11/10, 403). A divisão que vem
  do Roboflow foi descartada porque põe fotos tiradas com 1 s de diferença em conjuntos diferentes.
- **Mesma placa em dias diferentes, verificada.** A cena em volta de cada placa de validação e
  teste foi comparada com a das placas de treino (SIFT e RANSAC, 44 mil pares), e os pares mais
  parecidos foram revistos a olho. Foram achadas 5 placas repetidas, em dois lugares (4 das 263 do
  teste e 1 das 254 da validação). Sem elas, o F1 com a classe certa no teste vai de 0,639 para
  0,636 (Seção 9.4 do notebook).
- **26 classes de placa e `outra_placa`.** Ficam como classe própria as placas com pelo menos 10
  exemplos no treino; as 41 classes raras viram `outra_placa`. O delineador (`Del`) sai do modelo,
  porque só foi anotado em metade dos dias.
- **Aumento de dados medido nas placas do treino:** sem espelhamento (troca curva à direita por
  curva à esquerda), brilho e saturação até o percentil 5 medido, escala até o tamanho das placas
  mais distantes.
- **YOLO26 nano e small**, a 1.280 px, 20 épocas cada no Colab. Pela regra fixada antes do treino,
  o small só ficaria se ganhasse 0,02 de mAP50 na validação; ganhou 0,147 (0,770 contra 0,623).

### Como executar a inferência

```bash
git clone https://github.com/CaioHenri99/aed-visao-placas-transito.git
cd aed-visao-placas-transito
pip install -r requirements.txt

python inferir.py caminho/da/foto.jpg                 # uma imagem
python inferir.py pasta_de_fotos/ --saida resultados  # todas as imagens de uma pasta
```

O modelo treinado já vem no repositório (`modelos/modelo_n2.pt`, 20 MB), junto com o limiar
escolhido na validação (`modelos/modelo_n2.json`). Não precisa de GPU. Para cada imagem, o
script:

1. lê a imagem (JPG, PNG, BMP ou WebP);
2. faz o mesmo pré-processamento da validação no treino: *letterbox* para 1.280 px, RGB, 0 a 1;
3. roda o modelo;
4. imprime as placas acima do limiar e grava, na pasta de saída (`saida_inferencia/` por padrão),
   a imagem com as caixas desenhadas e um JSON.

```
$ python inferir.py captura_2023-10-10_17-29-24_png.rf.b80e265d20f80a4d38bb7a881b00c232.jpg
Modelo modelo_n2.pt | entrada de 1280 px | limiar de confiança 0.41

captura_2023-10-10_17-29-24_png.rf.b80e265d20f80a4d38bb7a881b00c232.jpg: 1 placa(s) em 2057 ms
  R-19         0.86  caixa (540, 424, 560, 445)  Velocidade máxima permitida
```

A caixa vem em pixels da imagem original (x0, y0, x1, y1). A primeira imagem demora uns 2 s, por
causa da carga do modelo; as seguintes, cerca de 0,35 s cada numa CPU de notebook.

| Opção | Efeito |
|---|---|
| `--saida PASTA` | Onde gravar as imagens anotadas e os JSON |
| `--limiar 0.25` | Outro limiar de confiança (o padrão é o da validação, 0,41) |
| `--sem-arquivos` | Só imprime, sem gravar nada |
| `--modelo ARQ.pt` | Outro peso treinado |

### Como reproduzir o treino

1. Abra [`notebooks/checkpoint2_modelo.ipynb`](notebooks/checkpoint2_modelo.ipynb) no Google
   Colab e escolha *Ambiente de execução → Alterar o tipo de ambiente de execução → GPU* (T4).
2. *Executar tudo* (`Ctrl+F9`). Logo no início, a Seção 0 pede acesso ao Google Drive, onde o
   treino é salvo, e a Seção 1 pede a chave gratuita da API do Roboflow. Depois disso, o resto
   roda sozinho.
3. As Seções 1 a 6 levam poucos minutos. A Seção 7 treina o nano (cerca de 25 min numa T4) e o
   small (cerca de 35 min). Se a sessão cair, basta rodar de novo: o treino retoma do último
   *checkpoint*, e um modelo já concluído não é treinado outra vez.
4. No fim, as curvas e matrizes vão para `outputs/n2/treino/` e o peso escolhido para
   `modelos/modelo_n2.pt`. No Colab, uma cópia fica em `MyDrive/aed_visao_n2/para_o_repositorio/`,
   na mesma estrutura do repositório, para ser trazida ao Git.

Sem GPU, o notebook roda inteiro mesmo assim: a Seção 7 pula o treino, e as Seções 8 a 10 usam o
modelo que já está no repositório. Para rodar local, com a base já baixada, aponte a variável de
ambiente `AED_PASTA_LOCAL` para a pasta dela; sem isso, a Seção 1 baixa a base com a chave do
Roboflow (veja abaixo, em *Execução*).

----

## Estrutura do repositório

```
aed-visao-placas-transito/
├── README.md
├── requirements.txt
├── inferir.py                         # N2: pipeline de inferência (imagem -> placas)
├── .github/workflows/validar.yml      # CI: notebooks íntegros, script compila, sem chave de API
├── notebooks/
│   ├── checkpoint1_pipeline_pdi.ipynb # N1: pipeline clássico (linha de base)
│   └── checkpoint2_modelo.ipynb       # N2: divisão, linha de base, treino, diagnóstico, inferência
├── modelos/                           # N2: modelo treinado (ficha em modelos/README.md)
│   ├── modelo_n2.pt                   # pesos do YOLO26s
│   └── modelo_n2.json                 # classes e limiar de confiança escolhido na validação
├── outputs/                           # gerado pelos notebooks e versionado
│   ├── n2/                            # N2 (índice dos arquivos em outputs/n2/README.md)
│   │   ├── treino/                    # curvas, configuração e escolha de cada treino
│   │   └── figuras/
│   ├── figuras/                       # N1: evidências visuais
│   ├── parametros_adotados.json       # N1: registro completo e reprodutível da execução
│   ├── parametros_adotados.md         # N1: tabela de parâmetros
│   ├── avaliacao_por_imagem.csv       # N1: VP, FP, FN e limiar T, imagem a imagem
│   └── ...                            # N1: inventário, busca em grade, descritores
├── docs/
│   ├── relatorio_tecnico_parcial.pdf  # relatório técnico do PP1
│   ├── arquitetura.md                 # diagrama da solução em Mermaid
│   ├── arquitetura_pipeline.png       # diagrama gerado pelo notebook da N1
│   └── referencias/                   # enunciado e proposta da 1ª Etapa
└── data/                              # não versionado: o notebook baixa a base aqui
    ├── placas-de-transito-br-wq5tp-1/ # base do Roboflow
    └── n2_yolo/                       # base no formato do YOLO, montada pela N2
```

As saídas da N1 ficam na raiz de `outputs/`, e não numa pasta `n1/`, porque é ali que o notebook
entregue na N1 grava e que o notebook da N2 lê os parâmetros congelados; mover quebraria a
reprodução dos dois.

----

# N1 · Pipeline clássico (linha de base)

Pipeline de Processamento Digital de Imagens clássico, sem aprendizado de máquina, que recebe
uma imagem de via, segmenta as placas de sinalização vertical e extrai contagem, área, centroide,
caixa envolvente e classe geométrica de cada uma. A justificativa completa de cada decisão está
no [relatório técnico](docs/relatorio_tecnico_parcial.pdf) e no próprio notebook.

----

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

## 3. Dataset

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

## 4. O pipeline

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

## 5. Parâmetros adotados

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

## 6. Resultados

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

## 7. Divisão de tarefas da equipe

| # | Integrante | Matrícula | Responsabilidade principal | Seções do notebook |
|---|---|---|---|---|
| 1 | Caio Henrique | 20241013700250 | Aquisição e curadoria do dataset, inventário e versionamento no Git | 1 e 2 |
| 2 | Fernanda Andrade | 20241013700048 | Pré-processamento: espaços de cor, iluminação, filtragem e histograma | 3.2 a 3.5 e 6.1 |
| 3 | Alisson Leonardo | 20241013700170 | Segmentação por cor, limiarização, morfologia e contornos | 3.6 a 3.8 e 6 |
| 4 | Vitor Manoel | 20241013700307 | Descritores geométricos, avaliação, figuras comparativas, relatório técnico e documentação | 3.8, 4, 7 e 8 |
| 5 | Luiz Antônio Cecilio Andrade | 20241013700374 | Arquitetura da solução, registro dos parâmetros, limitações e planejamento da 2ª Etapa | 9 a 11 |

A revisão de código é cruzada, para nenhuma parte depender de uma única pessoa.

---

## 8. Limitações conhecidas

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

## 9. O que veio depois (N2)

O detector treinado previsto aqui foi feito na N2 (seção *N2 · Modelo treinado*, no topo deste
README). O pipeline clássico ficou como linha de base comparativa. Ele não entrou como
pré-processamento do modelo: o detector aprendeu com as imagens cruas, e um CLAHE aplicado só na
inferência mudaria a imagem que ele recebe. Escala e anotação esparsa continuam sendo os limites
principais, agora medidos no notebook da N2 (Seções 8 e 11).
