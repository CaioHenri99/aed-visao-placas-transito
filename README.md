# Sistema Inteligente de Detecção e Classificação de Placas de Trânsito

AED / Projeto Integrador, 2ª Etapa, Checkpoint 1 (N1)

Pontifícia Universidade Católica de Goiás
Escola Politécnica e de Artes, Curso de Ciência de Dados e Inteligência Artificial
Disciplina CDI1021 (Visão Computacional, 2026/2), Prof. Welington Júlio Dias Rodrigues

---

## 1. O problema

A sinalização vertical de trânsito é o principal canal de comunicação entre a via e o
condutor. O inventário dessas placas e a fiscalização do seu estado de conservação ainda são
feitos por vistoria manual em campo, um processo lento, amostral e de difícil auditoria em
malhas urbanas extensas.

A dificuldade técnica não está em identificar o significado da placa, e sim na etapa anterior:
isolá-la de uma cena visualmente saturada, em que ela concorre com fachadas, vegetação,
veículos e sinalização publicitária, sob iluminação irregular, contraluz, sombra projetada,
desbotamento da película refletiva e oclusão parcial.

Este repositório resolve essa etapa anterior com Processamento Digital de Imagens clássico,
sem aprendizado de máquina. Dada uma imagem de via urbana, o sistema produz uma máscara
binária estável das placas e dela extrai contagem, área, centroide, caixa envolvente e classe
geométrica, além do recorte normalizado da região de interesse que vai alimentar o modelo de
IA da 2ª Etapa.

---

## 2. Execução

### Google Colab (recomendado)

1. Abra [`notebooks/checkpoint1_pipeline_pdi.ipynb`](notebooks/checkpoint1_pipeline_pdi.ipynb)
   no Google Colab.
2. Selecione Runtime e depois Executar tudo (`Ctrl+F9`).
3. Quando for solicitado, informe sua chave da API do Roboflow. Ela é gratuita e fica em
   `roboflow.com`, em Settings e depois API Keys.

Não é preciso nenhum outro ajuste manual. A primeira célula instala as dependências e todos os
parâmetros são calculados a partir do próprio dataset.

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

Para não digitar a chave a cada execução, exporte-a como variável de ambiente antes de abrir o
notebook. Ela nunca é gravada em arquivo:

```bash
# Linux/macOS
export ROBOFLOW_API_KEY="sua-chave"
# Windows PowerShell
$env:ROBOFLOW_API_KEY = "sua-chave"
```

### Sem chave da API do Roboflow

A Seção 1 do notebook oferece três caminhos de aquisição, e basta um deles funcionar.

| Caminho | Como usar |
|---|---|
| A. SDK do Roboflow | É o padrão. Informe a chave quando o notebook pedir. |
| B. Link direto | Na página do dataset, use Download Dataset, escolha `YOLOv8` e depois *show download code*. Cole a URL em `URL_DOWNLOAD_DIRETO`. |
| C. Pasta local | Aponte `PASTA_LOCAL` para uma pasta já baixada. É também por aqui que entram as imagens de captura autoral da equipe. |

---

## 3. Estrutura do repositório

```
aed-visao-placas-transito/
├── README.md                       # este arquivo
├── requirements.txt                # dependências fixadas
├── .gitignore                      # imagens fora do Git, figuras versionadas
├── .gitattributes                  # normalização de fim de linha (Windows e Colab)
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
└── outputs/                        # gerado pelo notebook e versionado
    ├── inventario_dataset.csv           # 1 linha por imagem: split, dimensões, nº de objetos
    ├── parametros_adotados.json         # registro completo e reprodutível da execução
    ├── parametros_adotados.md           # tabela pronta para colar na Seção 6 deste README
    ├── busca_em_grade.csv               # grade de método e área mínima na amostra de ajuste
    ├── comparacao_limiarizacao.csv      # melhor configuração de cada método
    ├── ablacao_preprocessamento.csv     # efeito medido de CLAHE e suavização
    ├── avaliacao_por_imagem.csv         # VP, FP, FN e erro de contagem, imagem a imagem
    ├── descritores_objetos.csv          # saída numérica, um objeto por linha
    └── figuras/                         # evidências visuais de antes e depois
```

---

## 4. Dataset

A fonte primária é a base [Placas de Trânsito (`paic/placas-de-transito-1jw8i`, versão
2)](https://universe.roboflow.com/paic/placas-de-transito-1jw8i/dataset/2), publicada no
Roboflow Universe.

É uma base de sinalização vertical brasileira, anotada em caixa delimitadora com os códigos do
CONTRAN (`A-1a` curva acentuada à esquerda, `R-1` parada obrigatória, `A-18` lombada, entre
outros). A aderência normativa foi o motivo da escolha: cor, forma e proporção das placas são
definidas por resolução, o que permite fundamentar cada parâmetro do pipeline em uma
especificação verificável, em vez de tentativa e erro.

| Item | Valor |
|---|---|
| Imagens | 5.297 (train 3.708, valid 1.060, test 529) |
| Objetos anotados | 8.627 |
| Classes | 75, nomeadas pelos códigos do CONTRAN |
| Dimensão mediana | 1280 × 720 px (mínimo 191 × 174, máximo 4032 × 3024) |
| Escala do objeto | A placa mediana ocupa 0,45% da área da imagem, ou seja, são cenas completas e há de fato segmentação a ser realizada |
| Formato de anotação | Caixa delimitadora, exportada em YOLO (`classe cx cy w h`, normalizados) |
| Sem anotação | 272 imagens, excluídas das amostras de ajuste e validação |

**Licença.** O `data.yaml` do export declara `license: Private`. O autor do projeto no Roboflow
não atribuiu uma licença aberta, ainda que a página seja de acesso público. Por isso este
repositório não redistribui as imagens: o notebook as baixa da fonte e o `.gitignore` mantém
`data/` fora do versionamento. O que fica versionado são as figuras derivadas, usadas como
evidência do checkpoint.

**Sobre os números da tabela.** Eles são gerados pela Seção 2 do notebook, que percorre o
dataset e grava `outputs/inventario_dataset.csv` com uma linha por imagem. Esse arquivo é a
documentação da base efetivamente usada, e não a anotação manual reproduzida aqui.

### Fontes complementares previstas

| Fonte | Volume | Uso |
|---|---|---|
| GTSDB (*German Traffic Sign Detection Benchmark*) | 900 imagens, 1360×800 | Validação independente em cena completa |
| GTSRB (*German Traffic Sign Recognition Benchmark*) | 51.839 recortes, 43 classes | Treinamento do classificador na 2ª Etapa |
| Captura autoral em Goiânia (GO) | cerca de 60 imagens | Validação local, incorporada pelo caminho C da Seção 1 |

### Considerações éticas

A placa de trânsito não constitui, em si, dado pessoal. As imagens de via, no entanto, capturam
incidentalmente pedestres, rostos e placas de identificação veicular, elementos alcançados pela
Lei nº 13.709/2018 (LGPD). As imagens de captura própria são anonimizadas antes da publicação no
repositório, e as bases acadêmicas são utilizadas estritamente nos termos de suas licenças.

---

## 5. O pipeline

```
imagem → redimensionar → corrigir iluminação → suavizar → mapa de evidência cromática
       → limiarizar → morfologia → contornos → descritores
```

| # | Etapa | Técnica | Por quê |
|---|---|---|---|
| 1 | Entrada | Redimensionamento para 640 px de largura (`INTER_AREA`) | Torna os parâmetros em pixels comparáveis entre imagens de resoluções diferentes |
| 2 | Iluminação | CLAHE no canal `L*` do LAB | Equalizar RGB canal a canal deslocaria a matiz, justamente o atributo em que a segmentação se apoia. No LAB, corrigir `L*` preserva a cor normativa. O *top-hat* fica disponível como alternativa |
| 3 | Ruído | Filtro gaussiano 5×5 | O ruído de alta frequência polui o histograma e desloca o limiar de Otsu. A ordem importa: suavizar vem antes de limiarizar |
| 4 | Evidência | Mapa escalar em HSV, com pesos gaussianos nas matizes normativas multiplicados pela saturação | Converte a noção de "parece uma placa" em um único canal contínuo, apto à limiarização. Usa distância circular de matiz, porque o vermelho ocupa as duas pontas da escala `H` |
| 5 | Segmentação | Limiarização global, Otsu, Otsu restrito e adaptativa | A escolha entre elas é feita por métrica, conforme a Seção 6 |
| 6 | Morfologia | Abertura, fechamento e preenchimento | A abertura remove ruído. O fechamento é a operação essencial: a placa de regulamentação é uma orla vermelha em torno de um miolo branco e, sem fechá-la, o `findContours` devolveria um anel, com área e centroide errados |
| 7 | Contornos | `findContours(RETR_EXTERNAL)` sobre a máscara morfológica | Nunca sobre a saída do Canny, porque uma borda de um pixel tem dois lados e duplicaria a contagem |
| 8 | Filtros | Área mínima e máxima, razão de aspecto, extensão e solidez | O piso remove ruído. O teto ataca céu, fachadas e vegetação, que a forma não separa de uma placa, apenas a escala (ver Seção 7) |
| 9 | Descritores | Área, perímetro, centroide, circularidade, solidez, extensão e vértices | Saída numérica que responde ao problema |

### Classificação geométrica

Os limiares de forma foram obtidos rasterizando cada forma normativa em cinco escalas, com
raio de 12 a 100 px, e medindo seus descritores.

| Forma | Vértices | Circularidade | Extensão | Área / círculo mínimo |
|---|---|---|---|---|
| Triangular (`R-2`, Dê a preferência) | 3 | 0,55 | cerca de 0,50 | não se aplica |
| Losangular (advertência, `A-*`) | 4 | 0,76 a 0,78 | cerca de 0,50 | não se aplica |
| Retangular (indicação) | 4 | 0,74 | cerca de 1,00 | não se aplica |
| Circular (regulamentação) | 5 ou mais | 0,86 a 0,89 | cerca de 0,79 | 0,93 a 0,99 |
| Octogonal (`R-1`, Pare) | 5 ou mais | 0,95 | cerca de 0,83 | 0,88 a 0,90 |

Essa medição levou a duas decisões de implementação:

- O losango é separado do retângulo pela extensão, e não pelo ângulo do `minAreaRect`. A
  convenção desse ângulo mudou entre o OpenCV 4 e o 5, e o código quebraria conforme a versão
  do ambiente.
- Círculo e octógono só são distinguidos em visada frontal. Sob perspectiva oblíqua nenhum
  descritor clássico os separa, e nesses casos o objeto recebe a marca `forma_ambigua = True`
  em vez de um rótulo falsamente confiante.

---

## 6. Parâmetros adotados

Os parâmetros críticos (limiar, tamanho de kernel e área mínima) não são constantes escolhidas
à mão. A Seção 5 do notebook os obtém em duas etapas, porque eles têm naturezas diferentes.

**5.1. Kernels morfológicos.** São consequência geométrica da escala do objeto.

| Parâmetro | Regra de derivação | Justificativa |
|---|---|---|
| `k_abertura` | `ímpar(0,10 × p10 do lado equivalente da placa)` | Precisa apagar ruído sem apagar a menor placa detectável, por isso ancora no percentil 10 e não na mediana |
| `k_fechamento` | `ímpar(0,25 × mediana do lado equivalente)` | Precisa vencer a espessura do miolo branco de uma placa típica, para que as margens da orla se toquem |

**5.2. Área mínima e método de limiarização.** Estes dois não têm valor correto derivável da
geometria, porque governam um compromisso entre precisão e recall. O notebook mede a curva de
trade-off e escolhe um ponto segundo um critério declarado de antemão.

Cada candidato a área é expresso como "descartar o quantil `q` das placas anotadas", em vez de
um número solto de pixels. Assim o parâmetro carrega significado: `q = 0,25` no piso quer dizer
que o pipeline abre mão do quartil de placas mais distantes, cuja área é comparável à do ruído
cromático residual. O piso ainda multiplica por 0,45, fator que converte área da caixa em área
da figura inscrita.

A busca tem dois estágios, no formato de busca por coordenadas. O estágio 1 percorre `q` de
0,05 a 0,60 combinado com os quatro métodos de limiarização. O estágio 2, já com o vencedor
fixado, varre o teto de área em `q` igual a 0,90, 0,95, 0,99 e sem teto.

O estágio 2 nasceu de uma falha concreta, o pipeline detectando o céu. A Seção 7 conta o
episódio, inclusive a parte em que o teto ótimo por F1 não resolve o caso por completo.

**Protocolo anti-viés.** A grade roda sobre uma amostra de ajuste, enquanto as métricas
reportadas na Seção 8 do notebook vêm de uma amostra de validação disjunta, que não participa
de nenhuma decisão. Sem essa separação, o desempenho publicado seria otimista por construção,
já que o parâmetro teria sido escolhido no mesmo conjunto em que é avaliado.

O limiar em si não é constante. Com Otsu, clássico ou restrito, ele é recalculado por imagem, e
o registro guarda a média e o desvio-padrão dos valores obtidos.

### Valores da última execução

O bloco abaixo é gerado pela Seção 10 do notebook em `outputs/parametros_adotados.md` e está
reproduzido aqui. Ao reexecutar, substitua por aquele arquivo.

Execução de 09/09/2026, semente 42, OpenCV 5.0.0, Python 3.14.4.

| Parâmetro | Valor | Origem |
|---|---|---|
| Largura de trabalho | 640 px | Padroniza a escala em pixels entre imagens |
| CLAHE (clip e grade) | 2,0 e 8×8 | Canal `L*` do LAB, preserva a matiz |
| Suavização | gaussiano 5×5 | Etapa exigida pelo checkpoint, com a ressalva medida abaixo |
| Kernel de abertura | 3×3 | `ímpar(0,10 × 10,3) = 3` |
| Kernel de fechamento | 9×9 | `ímpar(0,25 × 34,5) = 9` |
| Método de limiarização | `otsu_restrito` | Busca por coordenadas em 2 estágios, 32 configurações sobre 250 imagens de ajuste, maior F1 com IoU de no mínimo 0,30 |
| Área mínima de contorno | 536 px² | Descarta o quantil 0,50 inferior das placas anotadas, multiplicado por 0,45 de preenchimento |
| Área máxima de contorno | 79.520 px² | Descarta o quantil superior a 0,99. Ataca céu, fachadas e vegetação, com o alcance real medido na Seção 7 |
| Razão de aspecto aceita | 0,35 a 2,85 | Rejeita postes, faixas e meios-fios |
| Extensão mínima | 0,35 | Rejeita contornos rendilhados, como vegetação |
| Solidez mínima | 0,70 | Toda placa normativa é convexa |

**Protocolo de avaliação.** São 250 imagens de ajuste, usadas na escolha dos parâmetros, e 250
de validação, disjuntas das primeiras, sorteadas com semente fixa. O casamento entre detecção e
anotação usa IoU de no mínimo 0,30.

**Desempenho da linha de base clássica**, medido na amostra de validação retida, com 250
imagens:

| Métrica | Valor |
|---|---|
| Precisão | 0,314 |
| Recall | 0,221 |
| F1 | 0,259 |
| F1 na amostra de ajuste | 0,294 |
| Diferença entre ajuste e validação | 0,035 |
| Erro absoluto médio de contagem | 1,25 objetos por imagem |
| Contagem exata | 70 de 250 imagens |

A diferença de 0,035 entre ajuste e validação é pequena, e foi para chegar a esse patamar que
as amostras foram ampliadas para 250 imagens cada. Com 80 imagens por amostra, a mesma
diferença era de 0,177: escolher entre dezenas de configurações num conjunto pequeno estava
selecionando ruído amostral.

**Ressalva medida sobre a suavização.** A ablação não confirma ganho do filtro gaussiano neste
dataset, e desligá-lo sai ligeiramente à frente (F1 de 0,313 contra 0,294). A causa está na
origem das imagens: o export do Roboflow já passou por redimensionamento e recompressão JPEG,
que atenuam o ruído de sensor que o filtro removeria, e o que resta do gaussiano é a erosão das
placas menores. O filtro foi mantido por duas razões declaradas: é etapa exigida pelo Checkpoint
1 e as imagens de captura autoral da equipe chegarão sem esse pré-tratamento. Já o CLAHE melhora
o F1 em todos os pares comparados.

---

## 7. Como as decisões foram validadas

Cada escolha do pipeline tem uma medição por trás, e não uma afirmação.

| Seção | O que mede | Saída |
|---|---|---|
| 5.2 | Grade de método de limiarização e área mínima, na amostra de ajuste | `busca_em_grade.csv`, `figuras/01_escolha_de_parametros.png` |
| 6 | Ablação com CLAHE ligado e desligado, combinado com gaussiano, mediana e nenhum filtro | `ablacao_preprocessamento.csv`, `figuras/02_ablacao_preprocessamento.png` |
| 6.1 | Histograma do mapa de evidência com o corte de cada método sobreposto | `figuras/03_histograma_limiares.png` |
| 7 | Pipeline etapa a etapa em várias imagens sorteadas com semente fixa | `figuras/04_pipeline_*.png`, `figuras/05_mosaico_deteccoes.png` |
| 8 | Precisão, recall, F1 e erro de contagem na amostra retida | `avaliacao_por_imagem.csv`, `figuras/07_avaliacao_quantitativa.png` |

A ablação existe por um motivo específico: responder com número ao erro mais comum apontado na
orientação da AED, que é "segmentar sem suavizar antes, e concluir que Otsu não funciona".

### O episódio do céu

Este caso está registrado porque explica um filtro do pipeline e porque o desfecho não é o que
se esperaria.

A primeira execução sobre o dataset real apontou, em uma imagem de rodovia, dois objetos
detectados, e nenhum deles era a placa. O maior ocupava 13,6% da imagem, ou 31.355 px², e era o
céu. A causa é estrutural: céu azul limpo tem a matiz normativa do azul de indicação, saturação
alta e forma convexa. Ele passava pelo piso de área, pela razão de aspecto (2,74, logo abaixo do
limite de 2,85), pela extensão (0,56) e pela solidez (0,80). Faltava a única restrição capaz de
separá-lo de uma placa, que é a escala. A placa anotada mediana ocupa 0,48% da imagem, enquanto
20% das detecções passavam de 5%.

Daí veio o teto de área. A varredura do estágio 2, porém, mostrou um compromisso que não convém
maquiar.

| Teto (quantil das anotações) | Área | Precisão | Recall | F1 |
|---|---|---|---|---|
| `q = 0,90` | 10.834 px² | 0,387 | 0,208 | 0,271 |
| `q = 0,95` | 20.132 px² | 0,393 | 0,227 | 0,288 |
| `q = 0,99` (adotado) | 79.520 px² | 0,366 | 0,246 | 0,294 |
| sem teto | não se aplica | 0,359 | 0,248 | 0,294 |

A precisão sobe de forma monotônica conforme o teto aperta, mas o F1 fica praticamente plano, e
o valor que o maximiza (79.520 px²) ainda é largo demais para remover aquele blob de 31.355 px².
Apertar até `q = 0,95` o eliminaria, ao custo de recall nas placas fotografadas de perto, que
existem neste dataset.

Duas conclusões ficaram registradas, e nenhuma delas equivale a considerar o problema resolvido:

1. Filtros de forma não substituem filtro de escala. Céu e placa têm geometria parecida e
   tamanho incomparável.
2. O critério de seleção importa tanto quanto o parâmetro. Para o caso de uso de inventário
   viário, em que contar placas a mais é pior do que perder uma, a escolha adequada é apertar
   `quantil_area_max` para 0,95, e o custo dessa decisão está medido na tabela acima. O notebook
   mantém o critério declarado de antemão, que é maximizar o F1, em vez de trocá-lo depois de
   ver o resultado. Trocar o critério para favorecer a conclusão desejada seria o mesmo vício
   que a separação de amostras existe para evitar.

O desempenho desta etapa é baixo, e isso também é informação. Um detector puramente cromático
serve como linha de base contra a qual o modelo treinado da 2ª Etapa será comparado. O valor do
número está em existir, ser reprodutível e ter sido medido do mesmo jeito nas duas etapas.

---

## 8. Divisão de tarefas da equipe

A responsabilidade indicada é a principal de cada integrante. A revisão de código é cruzada, de
modo que nenhuma entrega dependa de uma única pessoa.

| # | Integrante | Matrícula | Responsabilidade principal | Seções do notebook |
|---|---|---|---|---|
| 1 | Caio Henrique | 20241013700250 | Aquisição e curadoria do dataset, organização dos diretórios, inventário das imagens e versionamento no Git | 1 e 2 |
| 2 | Fernanda Andrade | _a preencher_ | Pré-processamento: conversão de espaços de cor, correção de iluminação, filtragem espacial e análise de histograma | 3.2 a 3.5 e 6.1 |
| 3 | Alisson Leonardo | _a preencher_ | Segmentação por cor e limiarização, operações morfológicas e extração de contornos | 3.6 a 3.8 e 6 |
| 4 | Vitor Manoel | _a preencher_ | Descritores geométricos, relatório técnico, figuras comparativas, README e documentação de reprodução | 3.8, 7 a 11 |

---

## 9. Limitações conhecidas

1. **Falsos positivos de mesma cromaticidade.** Céu azul, vegetação seca (que cai na faixa de
   matiz do amarelo), lanternas traseiras, veículos vermelhos e toldos compartilham matiz e
   saturação com as placas. É o componente dominante do erro: com precisão de 0,31, cerca de
   dois terços das detecções são falsos positivos. O teto de área ataca os maiores, mas no ponto
   de operação escolhido por F1 ainda deixa passar regiões grandes de céu, conforme a Seção 7.
2. **Placas verdes fora do escopo.** O verde foi deliberadamente excluído das faixas de matiz,
   porque a vegetação urbana ocupa exatamente a mesma faixa e cobre área ordens de grandeza
   maior em cenas de via.
3. **Círculo e octógono sob perspectiva.** Ficam indistinguíveis por descritor clássico, e o
   código sinaliza a ambiguidade em vez de arbitrar.
4. **Placas pequenas e distantes.** Abaixo da área mínima calibrada, são descartadas por
   construção. O compromisso é explícito e ajustável.
5. **Desbotamento severo e oclusão.** Reduzem a saturação abaixo da porta mínima ou fragmentam o
   contorno, o que produz falsos negativos.
6. **Sem identificação do significado da placa.** O pipeline entrega forma e posição, mas não a
   categoria, que é por definição a tarefa da 2ª Etapa.
7. **Classes pouco cromáticas ficam sem cobertura.** Das 75 classes anotadas, várias não têm
   assinatura de cor forte, como o marcador de alinhamento `-MA-` e a família `-MP-` de
   passagem, em preto e amarelo sobre fundo variável. Um pipeline que decide por matiz e
   saturação não tem como alcançá-las.
8. **A escolha de parâmetros carrega variância amostral.** A Seção 8 do notebook reporta o F1
   nas duas amostras. A diferença entre o resultado no conjunto de ajuste e no conjunto retido
   mede o otimismo de escolher entre dezenas de configurações, e é por existir essa diferença
   que o número publicado é o da amostra retida.

---

## 10. Planejamento da 2ª Etapa (N2)

| Limitação atual | Tratamento previsto |
|---|---|
| Falsos positivos cromáticos | Detector YOLO treinado em cena completa, capaz de aprender contexto e textura além da cor |
| Placas verdes e desbotadas | Aprendizado supervisionado sobre exemplos reais anotados |
| Ambiguidade entre círculo e octógono | Classificação por CNN sobre a ROI, no lugar do descritor geométrico |
| Significado da placa | CNN classificadora treinada no GTSRB, com 43 categorias |

O pipeline clássico permanece em uso na N2 em três papéis: correção de iluminação como
pré-processamento do detector, filtro por área mínima como pós-processamento das caixas
propostas e contagem clássica como linha de base comparativa. É contra os números da Seção 8
que o ganho do modelo treinado será medido, por mAP, IoU e acurácia.

---

## 11. Cronograma

| Data | Marco | Situação |
|---|---|---|
| 31/08/2026 | Definição de tema e equipe | Concluído |
| 06/09/2026 | Obtenção e estruturação do dataset | Concluído |
| 11/09/2026 | Checkpoint 1, pipeline de PDI executável | Este entregável |
| 29/09 e 02/10/2026 | Defesa técnica parcial | A preparar |
| 02/10/2026 | Entrega final N1 (PP1), relatório em PDF | A preparar |
