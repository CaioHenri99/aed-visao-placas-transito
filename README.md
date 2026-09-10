# Sistema Inteligente de Detecção e Classificação de Placas de Trânsito

AED / Projeto Integrador, 2ª Etapa, Checkpoint 1 (N1)

Pontifícia Universidade Católica de Goiás
Escola Politécnica e de Artes, Curso de Ciência de Dados e Inteligência Artificial
Disciplina CDI1021 (Visão Computacional, 2026/2)

- Prof. Welington Júlio Dias Rodrigues

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

A fonte primária é a base [Placas de Trânsito BR (`stefano-tommasini-coelho-euf67/placas-de-transito-br`,
versão 9)](https://universe.roboflow.com/stefano-tommasini-coelho-euf67/placas-de-transito-br/dataset/9),
publicada no Roboflow Universe.

É uma base de sinalização vertical brasileira, anotada em caixa delimitadora com os códigos do
CONTRAN (`A-1a` curva acentuada à esquerda, `R-1` parada obrigatória, `R-19` velocidade máxima,
`I-4` indicação, entre outros). A aderência normativa foi o motivo da escolha: cor, forma e
proporção das placas são definidas por resolução, o que permite fundamentar cada parâmetro do
pipeline em uma especificação verificável, em vez de tentativa e erro.

| Item | Valor |
|---|---|
| Imagens | 2.963 (train 2.073, valid 593, test 297) |
| Objetos anotados | 2.305 |
| Classes | 68, nomeadas pelos códigos do CONTRAN |
| Dimensão | 640 × 640 px, padronizada pelo próprio export do Roboflow |
| Escala do objeto | A placa mediana tem cerca de 28 px de lado equivalente, ou seja, são cenas completas e há de fato segmentação a ser realizada |
| Formato de anotação | Caixa delimitadora, exportada em YOLO (`classe cx cy w h`, normalizados) |
| Sem anotação | 990 imagens, excluídas das amostras de ajuste e validação |

**Sobre a padronização em 640 × 640.** O export aplicou redimensionamento por esticamento, sem
preservar a proporção original. Isso distorce a razão de aspecto dos objetos, e é uma das razões
para o filtro correspondente do pipeline ser largo. Em compensação, todas as imagens chegam na
mesma escala, o que torna os parâmetros em pixels diretamente comparáveis entre elas.

**Licença.** O `data.yaml` do export declara `license: Public Domain`. Ainda assim, este
repositório não redistribui as imagens: o notebook as baixa da fonte e o `.gitignore` mantém
`data/` fora do versionamento, para não inflar o histórico do Git com dados que já têm um
endereço estável. O que fica versionado são as figuras derivadas, usadas como evidência do
checkpoint.

**Sobre os números da tabela.** Eles são gerados pela Seção 2 do notebook, que percorre o
dataset e grava `outputs/inventario_dataset.csv` com uma linha por imagem. Esse arquivo é a
documentação da base efetivamente usada, e não a anotação manual reproduzida aqui.

**As cores da base não são assumidas, são medidas.** A Seção 5.0 do notebook percorre as caixas
anotadas, calcula a matiz dominante de cada classe e agrupa as classes em torno das âncoras
normativas do CONTRAN. É assim que se descobre, por exemplo, que `I-4` é verde, que `S-14` e
`LOC-6` são azuis e que os delineadores da classe `Del` puxam a faixa amarela para o âmbar. As
faixas do pipeline saem dessa medição, e não de valores fixados à mão. O resultado fica em
`figuras/00b_calibracao_cromatica.png` e nos recortes de `figuras/00c_recortes_por_faixa.png`.

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
| 3 | Ruído | Filtro gaussiano 3×3, com o kernel derivado da escala da placa | O ruído de alta frequência polui o histograma e desloca o limiar de Otsu. A ordem importa: suavizar vem antes de limiarizar. O kernel precisa ser mais estreito que a orla da menor placa, senão mistura orla e miolo e derruba a saturação |
| 4 | Evidência | Mapa escalar em HSV, com pesos gaussianos nas matizes vermelha e amarela multiplicados pela saturação, atrás de uma porta mínima de saturação por faixa | Converte a noção de "parece uma placa" em um único canal contínuo, apto à limiarização. Usa distância circular de matiz, porque o vermelho ocupa as duas pontas da escala `H`. A porta de saturação é o que separa placa de grama seca, solo e fachada, e é escolhida por métrica |
| 5 | Segmentação | Limiarização global, Otsu, Otsu restrito e adaptativa | A escolha entre elas é feita por métrica, conforme a Seção 6 |
| 6 | Morfologia | Abertura, fechamento e preenchimento | A abertura remove ruído. O fechamento é a operação essencial: a placa de regulamentação é uma orla vermelha em torno de um miolo branco e, sem fechá-la, o `findContours` devolveria um anel, com área e centroide errados |
| 7 | Contornos | `findContours(RETR_EXTERNAL)` sobre a máscara morfológica | Nunca sobre a saída do Canny, porque uma borda de um pixel tem dois lados e duplicaria a contagem |
| 8 | Filtros | Área mínima e máxima, razão de aspecto, extensão e solidez | O piso remove ruído. O teto é um filtro de escala contra fachadas, toldos e vegetação fotografados de perto, que a forma não separa de uma placa |
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
à mão. A Seção 5 do notebook os obtém em três etapas, porque eles têm naturezas diferentes.

**5.0. Medição cromática nas anotações.** Antes de qualquer ajuste, o notebook mede, dentro de
cada faixa de matiz, a saturação dos pixels que são placa (núcleo das caixas anotadas) contra a
dos pixels de fundo. A tabela resultante mostra, para cada candidato a porta de saturação, que
fração da placa sobrevive e que fração do fundo sobrevive. A mesma célula mede a matiz das
classes que o CTB define em azul, e foi essa medição que retirou a faixa azul do pipeline (ver
Seção 4).

**5.1. Kernels.** São consequência geométrica da escala do objeto.

| Parâmetro | Regra de derivação | Justificativa |
|---|---|---|
| `suavizacao_k` | `ímpar(0,10 × p10 do lado equivalente da placa)` | O borrão precisa ser mais estreito que a orla da menor placa detectável, senão mistura orla e miolo |
| `k_abertura` | `ímpar(0,10 × p10 do lado equivalente da placa)` | Precisa apagar ruído sem apagar a menor placa detectável, por isso ancora no percentil 10 e não na mediana |
| `k_fechamento` | `ímpar(0,25 × mediana do lado equivalente)` | Precisa vencer a espessura do miolo branco de uma placa típica, para que as margens da orla se toquem |

**5.2. Portas de saturação, área mínima e método de limiarização.** Estes não têm valor correto
derivável da geometria, porque governam um compromisso entre precisão e recall. O notebook mede
a curva de trade-off e escolhe um ponto segundo um critério declarado de antemão.

Cada candidato a área é expresso como "descartar o quantil `q` das placas anotadas", em vez de
um número solto de pixels. Assim o parâmetro carrega significado: `q = 0,25` no piso quer dizer
que o pipeline abre mão do quartil de placas mais distantes, cuja área é comparável à do ruído
cromático residual. O piso ainda multiplica por 0,45, fator que converte área da caixa em área
da figura inscrita.

A busca tem três estágios, no formato de busca por coordenadas. O estágio 0 percorre as portas
de saturação das faixas vermelha (60, 80, 100 e 120) e amarela (130, 150 e 170), com piso e método
provisórios. O estágio 1 percorre `q` de 0,05 a 0,60 combinado com os quatro métodos de
limiarização. O estágio 2, já com o vencedor fixado, varre o teto de área em `q` igual a 0,90,
0,95, 0,99 e sem teto.

**Protocolo anti-viés.** A medição e a grade rodam sobre uma amostra de ajuste, enquanto as
métricas reportadas na Seção 8 do notebook vêm de uma amostra de validação disjunta, que não
participa de nenhuma decisão. Sem essa separação, o desempenho publicado seria otimista por
construção, já que o parâmetro teria sido escolhido no mesmo conjunto em que é avaliado. A
ordenação dos arquivos é feita de modo idêntico no Windows e no Linux, para que a mesma semente
produza as mesmas amostras no Colab e na máquina local.

O limiar em si não é constante. Com Otsu, clássico ou restrito, ele é recalculado por imagem, e
o registro guarda a média e o desvio-padrão dos valores obtidos.

### Valores da última execução

O bloco abaixo é gerado pela Seção 10 do notebook em `outputs/parametros_adotados.md` e está
reproduzido aqui. Ao reexecutar, substitua por aquele arquivo.

Execução de 10/09/2026, semente 42, OpenCV 5.0.0, Python 3.14.4.

| Parâmetro | Valor | Origem |
|---|---|---|
| Largura de trabalho | 640 px | Padroniza a escala em pixels entre imagens |
| CLAHE (clip e grade) | 2,0 e 8×8 | Canal `L*` do LAB, preserva a matiz |
| Suavização | gaussiano 3×3 | `ímpar(0,10 × 13,4) = 3`, mais estreito que a orla da menor placa |
| Faixas de matiz | vermelho H 10 ± 6 e amarelo H 19 ± 6 | Centro e largura medidos nas anotações, a partir das âncoras do CONTRAN |
| Portas de saturação | vermelho 100 e amarelo 160 | Estágio 0 da busca em grade, por descida em coordenadas |
| Faixas descartadas | azul, verde | O verde não reuniu objetos anotados suficientes. O azul foi medido e descartado no estágio 0 |
| Kernel de abertura | 3×3 | `ímpar(0,10 × 13,4) = 3` |
| Kernel de fechamento | 7×7 | `ímpar(0,25 × 29,6) = 7` |
| Método de limiarização | `adaptativa` | Busca por coordenadas em 3 estágios, 50 configurações sobre 250 imagens de ajuste, maior F1 com IoU de no mínimo 0,30 |
| Área mínima de contorno | 199 px² | Descarta o quantil 0,30 inferior das placas anotadas, multiplicado por 0,45 de preenchimento |
| Área máxima de contorno | 8.699 px² | Descarta o quantil superior a 0,99, como filtro de escala contra fachadas e vegetação fotografadas de perto |
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
| Precisão | 0,082 |
| Recall | 0,076 |
| F1 | 0,079 |
| F1 na amostra de ajuste | 0,134 |
| Diferença entre ajuste e validação | 0,055 |
| Erro absoluto médio de contagem | 0,88 objeto por imagem |
| Contagem exata | 84 de 250 imagens |

Esses números são baixos e a Seção 9 explica por quê, ponto a ponto. Dois fatores pesam mais que
os demais. O primeiro é a densidade de anotação da base: são 2.305 objetos em 1.973 imagens
anotadas, cerca de um objeto por imagem, e 990 imagens sem anotação nenhuma. As figuras de
`outputs/figuras/05_mosaico_deteccoes.png` mostram cenas com várias placas visíveis e apenas uma
anotada, o que faz o protocolo contar como falso positivo uma detecção correta. O segundo é a
escala: um décimo das placas anotadas tem menos de 14 px de lado equivalente.

**Ressalva medida sobre o pré-processamento.** A ablação mostra que o CLAHE se paga quando há
suavização, com ganho de 0,031 no F1 com filtro gaussiano e 0,030 com mediana, e que sem
suavização a correção de iluminação passa a atrapalhar, com perda de 0,011. O gaussiano de 3×3
foi mantido por ser a melhor combinação medida e por ser etapa exigida pelo Checkpoint 1. A
tabela completa está em `outputs/ablacao_preprocessamento.csv`.

---

## 7. Como as decisões foram validadas

Cada escolha do pipeline tem uma medição por trás, e não uma afirmação.

| Seção | O que mede | Saída |
|---|---|---|
| 5.0 | Matiz dominante de cada classe anotada, formação das faixas e separação de placa contra fundo por saturação | `figuras/00b_calibracao_cromatica.png`, `figuras/00c_recortes_por_faixa.png` |
| 5.2 | Grade de portas de saturação, método de limiarização e área mínima, na amostra de ajuste | `busca_em_grade.csv`, `figuras/01_escolha_de_parametros.png` |
| 6 | Ablação com CLAHE ligado e desligado, combinado com gaussiano, mediana e nenhum filtro | `ablacao_preprocessamento.csv`, `figuras/02_ablacao_preprocessamento.png` |
| 6.1 | Histograma do mapa de evidência com o corte de cada método sobreposto | `figuras/03_histograma_limiares.png` |
| 7 | Pipeline etapa a etapa em várias imagens sorteadas com semente fixa | `figuras/04_pipeline_*.png`, `figuras/05_mosaico_deteccoes.png` |
| 8 | Precisão, recall, F1 e erro de contagem na amostra retida | `avaliacao_por_imagem.csv`, `figuras/07_avaliacao_quantitativa.png` |

A ablação existe por um motivo específico: responder com número ao erro mais comum apontado na
orientação da AED, que é "segmentar sem suavizar antes, e concluir que Otsu não funciona".

### 1° Tentativa

Este caso está registrado porque mudou o pipeline e porque a primeira explicação estava errada.

Uma versão anterior do pipeline trazia uma faixa azul fixa, para as placas de indicação. Em
imagens de rodovia ela devolvia objetos detectados que eram o céu. Numa delas o maior deles
ocupava 13,6% da imagem e passava pelo piso de área, pela razão de aspecto, pela extensão e pela
solidez. A leitura inicial foi que faltava um filtro de escala, e daí veio o teto de área. A
varredura do teto mostrou que o valor que maximiza o F1 ainda deixava esses blobs passarem, e
que apertá-lo custava recall nas placas fotografadas de perto.

A causa real estava um passo antes. Uma faixa de cor fixada à mão não tem como saber se existe
alvo para ela naquela base, e o céu é a maior região de matiz azul de qualquer cena externa. A
correção foi estrutural: as faixas passaram a ser medidas nas anotações, e o estágio 0 da busca
passou a poder descartar uma faixa inteira. Nesta base o azul chegou a virar faixa, porque as
classes `S-14`, `LOC-6` e `RQ` são realmente azuis, mas o estágio 0 mediu que ela rende menos do
que custa e a descartou. O verde nem chegou a virar faixa, por falta de objetos anotados.

O teto de área continua no pipeline como filtro de escala contra fachadas, toldos e vegetação
fotografados de perto, e o estágio 2 da busca mede se ele ainda ajuda.

Duas conclusões ficaram registradas:

1. Um filtro de forma ou de escala não corrige uma faixa de cor sem alvo. O lugar de resolver o
   problema é a medição das anotações, e não um remendo na saída.
2. As faixas de cor que o CTB define não podem ser assumidas como presentes numa base anotada
   por terceiros. Cada faixa precisa ser confrontada com os pixels das caixas antes de entrar no
   pipeline, e precisa provar que se paga.

O desempenho desta etapa é modesto, e isso também é informação. Um detector puramente cromático
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

1. **A anotação da base é esparsa, e isso deprime a precisão medida.** São 2.305 objetos em
   1.973 imagens anotadas, cerca de um por imagem, além de 990 imagens sem anotação nenhuma. As
   cenas costumam ter mais placas visíveis do que anotadas, e cada detecção correta de uma placa
   não anotada entra na conta como falso positivo. O mosaico de detecções mostra o efeito. A
   precisão de 0,082 é, portanto, um piso, e não uma medida limpa do pipeline.
2. **Falsos positivos de mesma cromaticidade.** Lanternas traseiras, veículos vermelhos, toldos,
   solo exposto e grama seca muito saturada compartilham matiz e saturação com as placas. Nesta
   base o efeito é forte: com porta de saturação em 100, quase metade dos pixels de fundo dentro
   da faixa vermelha sobrevive, contra menos de um décimo em bases de melhor qualidade
   fotográfica. São quadros de câmera veicular em estrada de terra e vegetação seca, com as
   mesmas matizes das placas.
3. **Placas pequenas são o teto do recall.** Um décimo das placas anotadas tem menos de 14 px de
   lado equivalente, e a orla delas tem 1 px de espessura. Nenhuma combinação de cor e morfologia
   as recupera, e aumentar a largura de trabalho não ajuda, porque elas já são pequenas na imagem
   original.
4. **Cada faixa de cor carrega um confundidor natural.** O verde disputa a cena com vegetação, o
   azul com o céu, o amarelo com solo exposto, o vermelho com veículos. Por isso a porta de
   saturação e o descarte de faixa são decididos por métrica, e não por hipótese. Nesta execução
   o azul e o verde ficaram de fora, e as placas dessas cores não têm cobertura.
5. **Círculo e octógono sob perspectiva.** Ficam indistinguíveis por descritor clássico, e o
   código sinaliza a ambiguidade em vez de arbitrar. Nesta execução 93,7% dos objetos saíram
   marcados como forma ambígua, o que é consequência direta do tamanho dos objetos.
6. **Placas abaixo da área mínima calibrada** são descartadas por construção. O compromisso é
   explícito e ajustável.
7. **Desbotamento severo e oclusão.** Reduzem a saturação abaixo da porta mínima ou fragmentam o
   contorno, o que produz falsos negativos.
8. **Sem identificação do significado da placa.** O pipeline entrega forma e posição, mas não a
   categoria, que é por definição a tarefa da 2ª Etapa.
9. **Classes pouco cromáticas ficam sem cobertura.** Das 68 classes anotadas, várias não têm
   assinatura de cor forte, com pictograma preto sobre fundo claro. Um pipeline que decide por
   matiz e saturação não tem como alcançá-las.
10. **A escolha de parâmetros carrega variância amostral.** A Seção 8 do notebook reporta o F1
   nas duas amostras. A diferença entre o resultado no conjunto de ajuste e no conjunto retido
   mede o otimismo de escolher entre dezenas de configurações, e é por existir essa diferença
   que o número publicado é o da amostra retida.

---

## 10. Planejamento da 2ª Etapa (N2)

| Limitação atual | Tratamento previsto |
|---|---|
| Falsos positivos cromáticos | Detector YOLO treinado em cena completa, capaz de aprender contexto e textura além da cor |
| Placas pequenas e distantes | Detector treinado em múltiplas escalas, com avaliação separada por tamanho de objeto |
| Placas verdes, azuis e desbotadas | Aprendizado supervisionado sobre exemplos reais anotados |
| Ambiguidade entre círculo e octógono | Classificação por CNN sobre a ROI, no lugar do descritor geométrico |
| Significado da placa | CNN classificadora treinada no GTSRB, com 43 categorias |

O pipeline clássico permanece em uso na N2 em três papéis: correção de iluminação como
pré-processamento do detector, filtro por área mínima como pós-processamento das caixas
propostas e contagem clássica como linha de base comparativa. É contra os números da Seção 8
que o ganho do modelo treinado será medido, por mAP, IoU e acurácia.
