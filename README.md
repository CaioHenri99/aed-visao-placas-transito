# Sistema Inteligente de Detecção e Classificação de Placas de Trânsito

**AED · Projeto Integrador — 2ª Etapa · Checkpoint 1 (N1)**

Pontifícia Universidade Católica de Goiás — Escola Politécnica e de Artes
Curso de Ciência de Dados e Inteligência Artificial
Disciplina **CDI1021 — Visão Computacional (2026/2)** · Prof. Welington Júlio Dias Rodrigues

---

## 1. O problema

A sinalização vertical de trânsito é o principal canal de comunicação entre a via e o
condutor. Seu inventário e a fiscalização do estado de conservação ainda são feitos por
vistoria manual em campo — processo lento, amostral e de difícil auditoria em malhas urbanas
extensas.

O desafio técnico não está em identificar o significado da placa, mas na etapa anterior:
**isolá-la de uma cena visualmente saturada**, em que ela concorre com fachadas, vegetação,
veículos e sinalização publicitária, sob iluminação irregular, contraluz, sombra projetada,
desbotamento da película refletiva e oclusão parcial.

Este repositório entrega a solução dessa etapa anterior por **Processamento Digital de
Imagens clássico**, sem aprendizado de máquina: dada uma imagem de via urbana, o sistema
produz uma máscara binária estável das placas e dela extrai **contagem, área, centroide,
caixa envolvente e classe geométrica**, além do recorte normalizado da região de interesse
que alimentará o modelo de IA da 2ª Etapa.

---

## 2. Execução

### Google Colab (recomendado)

1. Abra [`notebooks/checkpoint1_pipeline_pdi.ipynb`](notebooks/checkpoint1_pipeline_pdi.ipynb)
   no Google Colab.
2. **Runtime → Executar tudo** (`Ctrl+F9`).
3. Quando solicitado, informe sua **chave da API do Roboflow** (gratuita:
   `roboflow.com` → *Settings* → *API Keys*).

Não há nenhum outro ajuste manual. Todas as dependências são instaladas pela primeira
célula e todos os parâmetros são calculados a partir do próprio dataset.

### Execução local

```bash
git clone <URL-DO-REPOSITORIO>
cd aed_visao

python -m venv .venv
# Windows:  .venv\Scripts\activate
# Linux/macOS:  source .venv/bin/activate

pip install -r requirements.txt
jupyter lab notebooks/checkpoint1_pipeline_pdi.ipynb
```

Para não digitar a chave a cada execução, exporte-a como variável de ambiente antes de abrir
o notebook — ela nunca é gravada em arquivo:

```bash
# Linux/macOS
export ROBOFLOW_API_KEY="sua-chave"
# Windows PowerShell
$env:ROBOFLOW_API_KEY = "sua-chave"
```

### Sem chave da API do Roboflow

A Seção 1 do notebook oferece três caminhos de aquisição, e basta um funcionar:

| Caminho | Como usar |
|---|---|
| **A — SDK do Roboflow** | Padrão. Informe a chave quando o notebook pedir. |
| **B — link direto** | Na página do dataset, *Download Dataset* → `YOLOv8` → *show download code*; cole a URL em `URL_DOWNLOAD_DIRETO`. |
| **C — pasta local** | Aponte `PASTA_LOCAL` para uma pasta já baixada — é também por aqui que entram as imagens de captura autoral da equipe. |

---

## 3. Estrutura do repositório

```
aed_visao/
├── README.md                       # este arquivo
├── requirements.txt                # dependências fixadas
├── .gitignore                      # imagens fora do Git; figuras versionadas
├── .gitattributes                  # normalização de fim de linha (Windows + Colab)
├── notebooks/
│   └── checkpoint1_pipeline_pdi.ipynb   # pipeline completo, executável de ponta a ponta
├── docs/
│   ├── arquitetura.md                   # diagrama da solução em Mermaid
│   ├── arquitetura_pipeline.png         # diagrama gerado pelo notebook
│   └── referencias/
│       ├── orientacoes-aed-1a-etapa.pdf     # enunciado da disciplina
│       └── proposta-tema-e-equipe.docx      # documento da 1ª Etapa
├── data/                           # dataset baixado (não versionado)
│   └── placas-de-transito-2/
│       ├── data.yaml
│       ├── train/  images/  labels/
│       ├── valid/  images/  labels/
│       └── test/   images/  labels/
└── outputs/                        # gerado pelo notebook — versionado
    ├── inventario_dataset.csv           # 1 linha por imagem: split, dimensões, nº de objetos
    ├── parametros_adotados.json         # registro completo e reprodutível da execução
    ├── parametros_adotados.md           # tabela pronta para colar na Seção 6 deste README
    ├── busca_em_grade.csv               # grade (método × área mínima) na amostra de ajuste
    ├── comparacao_limiarizacao.csv      # melhor configuração de cada método
    ├── ablacao_preprocessamento.csv     # efeito medido de CLAHE e suavização
    ├── avaliacao_por_imagem.csv         # VP/FP/FN e erro de contagem, imagem a imagem
    ├── descritores_objetos.csv          # saída numérica: um objeto por linha
    └── figuras/                         # evidências visuais antes/depois
```

---

## 4. Dataset

**Fonte primária:** [Placas de Trânsito — `paic/placas-de-transito-1jw8i`, versão 2](https://universe.roboflow.com/paic/placas-de-transito-1jw8i/dataset/2),
publicado no Roboflow Universe.

Base de **sinalização vertical brasileira**, anotada em caixa delimitadora com os códigos do
CONTRAN (`A-1a` curva acentuada à esquerda, `R-1` parada obrigatória, `A-18` lombada etc.).
A aderência normativa é o motivo da escolha: cor, forma e proporção das placas são definidas
por resolução, o que permite fundamentar cada parâmetro do pipeline em uma especificação
verificável em vez de tentativa e erro.

| Item | Valor |
|---|---|
| Imagens | **5.297** — `train` 3.708 · `valid` 1.060 · `test` 529 |
| Objetos anotados | **8.627** |
| Classes | **75**, nomeadas pelos códigos do CONTRAN |
| Dimensão mediana | 1280 × 720 px (mínimo 191 × 174, máximo 4032 × 3024) |
| Escala do objeto | A placa mediana ocupa **0,45%** da área da imagem — são cenas completas, há de fato segmentação a ser realizada |
| Formato de anotação | Caixa delimitadora, exportada em YOLO (`classe cx cy w h`, normalizados) |
| Sem anotação | 272 imagens (excluídas das amostras de ajuste e validação) |

> **Licença.** O `data.yaml` do export declara `license: Private` — o autor do projeto no
> Roboflow não atribuiu uma licença aberta, ainda que a página seja de acesso público. Por
> isso **este repositório não redistribui as imagens**: o notebook as baixa da fonte, e o
> `.gitignore` mantém `data/` fora do versionamento. O que fica versionado são as figuras
> derivadas usadas como evidência do checkpoint.

> **Os números acima são gerados, não digitados.** A Seção 2 do notebook percorre o dataset e
> grava `outputs/inventario_dataset.csv` com uma linha por imagem. É esse arquivo — e não uma
> anotação manual neste README — que documenta a base efetivamente usada.

### Fontes complementares previstas

| Fonte | Volume | Uso |
|---|---|---|
| GTSDB — *German Traffic Sign Detection Benchmark* | 900 imagens, 1360×800 | Validação independente em cena completa |
| GTSRB — *German Traffic Sign Recognition Benchmark* | 51.839 recortes, 43 classes | Treinamento do classificador na 2ª Etapa |
| Captura autoral (Goiânia-GO) | ≈ 60 imagens | Validação local; entra pelo caminho C da Seção 1 |

### Considerações éticas

A placa de trânsito não constitui, em si, dado pessoal. Imagens de via, contudo, capturam
incidentalmente pedestres, rostos e placas de identificação veicular, alcançados pela Lei nº
13.709/2018 (LGPD). As imagens de captura própria são anonimizadas antes da publicação no
repositório; as bases acadêmicas são usadas estritamente nos termos de suas licenças.

---

## 5. O pipeline

```
imagem → redimensionar → corrigir iluminação → suavizar → mapa de evidência cromática
       → limiarizar → morfologia → contornos → descritores
```

| # | Etapa | Técnica | Por quê |
|---|---|---|---|
| 1 | Entrada | Redimensionamento para 640 px de largura (`INTER_AREA`) | Torna os parâmetros em pixels comparáveis entre imagens de resoluções diferentes |
| 2 | Iluminação | **CLAHE no canal `L*` do LAB** | Equalizar RGB canal a canal deslocaria a matiz — justamente o atributo em que a segmentação se apoia. No LAB, corrigir `L*` preserva a cor normativa. *Top-hat* fica disponível como alternativa |
| 3 | Ruído | Filtro gaussiano 5×5 | O ruído de alta frequência polui o histograma e desloca o limiar de Otsu. Suavizar **antes** de limiarizar não é opcional |
| 4 | Evidência | Mapa escalar HSV com pesos gaussianos nas matizes normativas × saturação | Converte "parece uma placa" em um único canal contínuo, apto à limiarização. Usa distância **circular** de matiz, porque o vermelho ocupa as duas pontas da escala `H` |
| 5 | Segmentação | Limiarização — global, Otsu, **Otsu restrito** e adaptativa | Escolhida por métrica, não por preferência (Seção 6) |
| 6 | Morfologia | Abertura → fechamento → preenchimento | A abertura remove ruído; o **fechamento é essencial**: a placa de regulamentação é uma orla vermelha em torno de um miolo branco, e sem fechá-la `findContours` devolveria um anel, com área e centroide errados |
| 7 | Contornos | `findContours(RETR_EXTERNAL)` sobre a máscara morfológica | **Nunca sobre a saída do Canny**: uma borda de um pixel tem dois lados e duplicaria a contagem |
| 8 | Filtros | Área **mínima e máxima**, razão de aspecto, extensão, solidez | O piso remove ruído; o teto ataca céu, fachadas e vegetação — só a escala os separa de uma placa, já que a forma não os separa (ver Seção 7) |
| 9 | Descritores | Área, perímetro, centroide, circularidade, solidez, extensão, vértices | Saída numérica que responde ao problema |

### Classificação geométrica

Os limiares de forma foram obtidos rasterizando cada forma normativa em cinco escalas
(`r = 12` a `100 px`) e medindo seus descritores:

| Forma | Vértices | Circularidade | Extensão | Área / círculo mínimo |
|---|---|---|---|---|
| Triangular (`R-2` Dê a preferência) | 3 | 0,55 | ~0,50 | — |
| Losangular (advertência `A-*`) | 4 | 0,76–0,78 | **~0,50** | — |
| Retangular (indicação) | 4 | 0,74 | **~1,00** | — |
| Circular (regulamentação) | ≥ 5 | 0,86–0,89 | ~0,79 | **0,93–0,99** |
| Octogonal (`R-1` Pare) | ≥ 5 | 0,95 | ~0,83 | **0,88–0,90** |

Duas consequências práticas dessa medição:

- **Losango é separado de retângulo pela extensão, não pelo ângulo do `minAreaRect`** — a
  convenção desse ângulo mudou entre OpenCV 4 e 5, e o código quebraria conforme a versão do
  ambiente.
- **Círculo e octógono só são distinguidos em visada frontal.** Sob perspectiva oblíqua,
  nenhum descritor clássico os separa; nesses casos o objeto é marcado com
  `forma_ambigua = True` em vez de receber um rótulo falsamente confiante.

---

## 6. Parâmetros adotados

Os parâmetros críticos — **limiar, tamanho de kernel e área mínima** — não são constantes
escolhidas à mão. A Seção 5 do notebook os obtém em duas etapas, porque são de naturezas
diferentes.

**5.1 — Kernels morfológicos: consequência geométrica da escala do objeto.**

| Parâmetro | Regra de derivação | Justificativa |
|---|---|---|
| `k_abertura` | `ímpar(0,10 × p10 do lado equivalente da placa)` | Precisa apagar ruído sem apagar a **menor** placa detectável — por isso ancora no percentil 10, não na mediana |
| `k_fechamento` | `ímpar(0,25 × mediana do lado equivalente)` | Precisa vencer a espessura do miolo branco de uma placa **típica** para que as margens da orla se toquem |

**5.2 — Área mínima e método de limiarização: escolha por busca em grade.**

Estes dois não têm valor "correto" derivável da geometria — governam um compromisso entre
precisão e recall. O notebook mede a curva de trade-off e escolhe um ponto por critério
declarado:

- Cada candidato a área é expresso como **"descartar o quantil `q` das placas anotadas"**, e
  não como um número solto de pixels. Assim o parâmetro significa algo: `q = 0,25` no piso
  quer dizer *"o pipeline abre mão do quartil de placas mais distantes, cuja área é comparável
  à do ruído cromático residual"*. O piso ainda multiplica por 0,45, fator que converte área da
  *caixa* em área da *figura inscrita*.
- A busca tem **dois estágios** (busca por coordenadas): o estágio 1 percorre
  `q ∈ {0,05 … 0,60}` × os 4 métodos de limiarização; o estágio 2, com o vencedor fixado,
  varre o **teto de área** em `q ∈ {0,90; 0,95; 0,99; sem teto}`.

O estágio 2 nasceu de uma falha concreta — o pipeline "detectando o céu" — e a Seção 7 conta o
episódio, inclusive a parte em que o teto ótimo por F1 **não** resolve o caso por completo.

> **Protocolo anti-viés.** A grade roda sobre uma **amostra de ajuste**; as métricas
> reportadas na Seção 8 do notebook vêm de uma **amostra de validação disjunta**, que não
> participa de nenhuma decisão. Sem essa separação, o desempenho publicado seria otimista por
> construção — o parâmetro teria sido escolhido no mesmo conjunto em que é avaliado.

O `limiar` em si não é constante: com Otsu (clássico ou restrito) ele é **recalculado por
imagem**, e o registro guarda a média e o desvio-padrão dos valores obtidos.

### ▸ Valores da última execução

Bloco gerado pela Seção 10 do notebook em `outputs/parametros_adotados.md` — reproduzido aqui.
Ao reexecutar, substitua por aquele arquivo.

> Execução de **09/09/2026** · semente 42 · OpenCV 5.0.0 · Python 3.14.4

| Parâmetro | Valor | Origem |
|---|---|---|
| Largura de trabalho | 640 px | Padroniza a escala em pixels entre imagens |
| CLAHE (clip / grade) | 2,0 / 8×8 | Canal `L*` do LAB, preserva a matiz |
| Suavização | gaussiano 5×5 | Etapa exigida pelo checkpoint; ver a ressalva medida abaixo |
| **Kernel de abertura** | **3×3** | `ímpar(0,10 × 10,3) = 3` |
| **Kernel de fechamento** | **9×9** | `ímpar(0,25 × 34,5) = 9` |
| **Método de limiarização** | **`otsu_restrito`** | Busca por coordenadas em 2 estágios, 32 configurações sobre 250 imagens de ajuste, maior F1 com IoU ≥ 0,30 |
| **Área mínima de contorno** | **536 px²** | Descarta o quantil 0,50 inferior das placas anotadas (× 0,45 de preenchimento) |
| **Área máxima de contorno** | **79.520 px²** | Descarta o quantil superior a 0,99 — ataca céu, fachadas e vegetação (alcance real medido na Seção 7) |
| Razão de aspecto aceita | 0,35–2,85 | Rejeita postes, faixas e meios-fios |
| Extensão mínima | 0,35 | Rejeita contornos rendilhados (vegetação) |
| Solidez mínima | 0,70 | Toda placa normativa é convexa |

**Protocolo de avaliação** — 250 imagens de *ajuste* (escolha dos parâmetros) e 250 de
*validação*, disjuntas, sorteadas com semente fixa. Casamento detecção ↔ anotação por
IoU ≥ 0,30.

**Desempenho da linha de base clássica** (amostra de validação retida, 250 imagens):

| Métrica | Valor |
|---|---|
| Precisão | 0,314 |
| Recall | 0,221 |
| **F1** | **0,259** |
| F1 na amostra de ajuste | 0,294 |
| Diferença ajuste → validação | 0,035 |
| Erro absoluto médio de contagem | 1,25 objetos por imagem |
| Contagem exata | 70 de 250 imagens |

A diferença de 0,035 entre ajuste e validação é pequena — foi para chegar a esse patamar que
as amostras foram ampliadas para 250 imagens cada. Com 80 imagens por amostra, a mesma
diferença era de 0,177: escolher entre dezenas de configurações num conjunto pequeno estava
selecionando ruído, não sinal.

**Ressalva medida sobre a suavização.** A ablação **não** confirma ganho do filtro gaussiano
*neste* dataset — desligá-lo sai ligeiramente à frente (F1 0,313 contra 0,294). A causa é a
origem das imagens: o export do Roboflow já passou por redimensionamento e recompressão JPEG,
que atenuam o ruído de sensor que o filtro removeria; o que resta é a erosão das placas
menores. O filtro é mantido por duas razões declaradas — é etapa exigida pelo Checkpoint 1, e
as imagens de captura autoral da equipe chegarão sem esse pré-tratamento. O CLAHE, esse sim,
melhora o F1 em todos os pares comparados.

---

## 7. Como as decisões foram validadas

O notebook não afirma que as escolhas são boas — ele as mede.

| Seção | O que mede | Saída |
|---|---|---|
| 5.2 | Grade (método de limiarização × área mínima) na amostra de ajuste | `busca_em_grade.csv`, `figuras/01_escolha_de_parametros.png` |
| 6 | Ablação: CLAHE ligado/desligado × gaussiano/mediana/nenhum | `ablacao_preprocessamento.csv`, `figuras/02_ablacao_preprocessamento.png` |
| 6.1 | Histograma do mapa de evidência com o corte de cada método sobreposto | `figuras/03_histograma_limiares.png` |
| 7 | Pipeline etapa a etapa em várias imagens sorteadas com semente fixa | `figuras/04_pipeline_*.png`, `figuras/05_mosaico_deteccoes.png` |
| 8 | Precisão, recall, F1 e erro de contagem na **amostra retida** | `avaliacao_por_imagem.csv`, `figuras/07_avaliacao_quantitativa.png` |

A ablação existe por um motivo específico: ela responde com número, e não com opinião, ao
erro mais comum apontado na orientação da AED — *"segmentar sem suavizar antes, e concluir
que Otsu não funciona"*.

### O episódio do céu

Vale registrar porque explica um filtro do pipeline — e porque o desfecho não é o que se
esperaria.

A primeira execução sobre o dataset real apontava, em uma imagem de rodovia, dois "objetos
detectados", e **nenhum deles era a placa**. O maior ocupava 13,6% da imagem (31.355 px²): era
o **céu**. A causa é estrutural, não um bug — céu azul limpo tem a matiz normativa do azul de
indicação, saturação alta e forma convexa. Passava pelo piso de área, pela razão de aspecto
(2,74, logo abaixo do limite de 2,85), pela extensão (0,56) e pela solidez (0,80). Faltava a
única restrição que o separa de uma placa: **a escala**. A placa anotada mediana ocupa 0,48%
da imagem, e 20% das detecções passavam de 5%.

Daí o teto de área. Mas a varredura do estágio 2 mostrou um compromisso que convém não
maquiar:

| Teto (quantil das anotações) | Área | Precisão | Recall | F1 |
|---|---|---|---|---|
| `q = 0,90` | 10.834 px² | 0,387 | 0,208 | 0,271 |
| `q = 0,95` | 20.132 px² | **0,393** | 0,227 | 0,288 |
| `q = 0,99` ← adotado | 79.520 px² | 0,366 | 0,246 | **0,294** |
| sem teto | — | 0,359 | 0,248 | 0,294 |

A precisão sobe monotonicamente conforme o teto aperta, mas o **F1 é praticamente plano** — e
o valor que o maximiza (79.520 px²) ainda é largo demais para remover aquele blob de 31.355
px². Apertar até `q = 0,95` o eliminaria, ao custo de recall nas placas fotografadas de perto,
que existem neste dataset.

Duas conclusões ficam registradas, e nenhuma delas é "resolvido":

1. **Filtros de forma não substituem filtro de escala.** Céu e placa têm geometria parecida e
   tamanho incomparável.
2. **O critério de seleção importa tanto quanto o parâmetro.** Para o caso de uso de
   inventário viário — em que contar placas a mais é pior que perder uma — a escolha correta é
   apertar `quantil_area_max` para 0,95, e o custo dessa decisão está medido acima. O notebook
   mantém o critério declarado de antemão (maior F1) em vez de trocá-lo depois de ver o
   resultado; trocar o critério para favorecer a conclusão desejada seria o mesmo vício que a
   separação de amostras existe para evitar.

O desempenho desta etapa **não é alto, e isso é informação e não fracasso**: um detector
puramente cromático é a linha de base contra a qual o modelo treinado da 2ª Etapa será
comparado. O valor do número está em existir, ser reprodutível e ter sido medido do mesmo
jeito nas duas etapas.

---

## 8. Divisão de tarefas da equipe

A responsabilidade abaixo é a **principal** de cada integrante; a revisão de código é
cruzada, de modo que nenhuma entrega dependa de um único integrante.

| # | Integrante | Matrícula | Responsabilidade principal | Seções do notebook |
|---|---|---|---|---|
| 1 | Caio Henrique | 20241013700250 | Aquisição e curadoria do dataset; organização dos diretórios, inventário das imagens e versionamento no Git | 1 e 2 |
| 2 | Fernanda Andrade | _a preencher_ | Pré-processamento: conversão de espaços de cor, correção de iluminação, filtragem espacial e análise de histograma | 3.2 a 3.5 e 6.1 |
| 3 | Alisson Leonardo | _a preencher_ | Segmentação por cor e limiarização, operações morfológicas e extração de contornos | 3.6 a 3.8 e 6 |
| 4 | Vitor Manoel | _a preencher_ | Descritores geométricos, relatório técnico, figuras comparativas, README e documentação de reprodução | 3.8, 7 a 11 |

---

## 9. Limitações conhecidas

1. **Falsos positivos de mesma cromaticidade** — céu azul, vegetação seca (que cai na faixa de
   matiz do amarelo), lanternas traseiras, veículos vermelhos e toldos compartilham matiz e
   saturação com as placas. É o componente dominante do erro: com precisão 0,31, cerca de dois
   terços das detecções são falsos positivos. O teto de área ataca os maiores, mas no ponto de
   operação escolhido por F1 ainda deixa passar regiões grandes de céu (Seção 7).
2. **Placas verdes fora do escopo** — o verde foi deliberadamente excluído das faixas de
   matiz: a vegetação urbana ocupa exatamente a mesma faixa e cobre área ordens de grandeza
   maior em cenas de via.
3. **Círculo e octógono sob perspectiva** — indistinguíveis por descritor clássico; o código
   sinaliza a ambiguidade em vez de arbitrar.
4. **Placas pequenas e distantes** — abaixo da área mínima calibrada são descartadas por
   construção; o compromisso é explícito e ajustável.
5. **Desbotamento severo e oclusão** — reduzem a saturação abaixo da porta mínima ou
   fragmentam o contorno, produzindo falsos negativos.
6. **Sem identificação do significado da placa** — entrega forma e posição, não a categoria.
   Essa é, por definição, a tarefa da 2ª Etapa.

7. **Classes pouco cromáticas ficam sem cobertura** — das 75 classes anotadas, várias não têm
   assinatura de cor forte (por exemplo `-MA-` marcador de alinhamento e a família `-MP-` de
   passagem, em preto e amarelo sobre fundo variável). Um pipeline que decide por matiz e
   saturação não tem como alcançá-las.

8. **A escolha de parâmetros carrega variância amostral** — a Seção 8 do notebook reporta o F1
   nas duas amostras. A diferença entre o resultado no conjunto de ajuste e no conjunto retido
   mede o otimismo de escolher entre dezenas de configurações; é por existir essa diferença que
   o número publicado é o da amostra retida.

---

## 10. Roadmap — 2ª Etapa (N2)

| Limitação atual | Tratamento previsto |
|---|---|
| Falsos positivos cromáticos | Detector YOLO treinado em cena completa: aprende contexto e textura, não só cor |
| Placas verdes e desbotadas | Aprendizado supervisionado sobre exemplos reais anotados |
| Ambiguidade círculo/octógono | Classificação por CNN sobre a ROI, não por descritor geométrico |
| Significado da placa | CNN classificadora treinada no GTSRB (43 categorias) |

O pipeline clássico **permanece em uso na N2** em três papéis: correção de iluminação como
pré-processamento do detector, filtro por área mínima como pós-processamento das caixas
propostas, e contagem clássica como linha de base comparativa — é contra os números da Seção
8 que o ganho do modelo treinado será medido, por mAP, IoU e acurácia.

---

## 11. Cronograma

| Data | Marco | Situação |
|---|---|---|
| 31/08/2026 | Definição de tema e equipe | Concluído |
| 06/09/2026 | Obtenção e estruturação do dataset | Concluído |
| **11/09/2026** | **Checkpoint 1 — pipeline de PDI executável** | **Este entregável** |
| 29/09 e 02/10/2026 | Defesa técnica parcial | A preparar |
| 02/10/2026 | Entrega final N1 (PP1) — relatório em PDF | A preparar |
