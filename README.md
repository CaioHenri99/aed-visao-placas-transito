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
├── .github/
│   └── workflows/validar.yml       # CI: notebook íntegro, sem saídas pesadas e sem chave de API
├── notebooks/
│   └── checkpoint1_pipeline_pdi.ipynb   # pipeline completo, executável de ponta a ponta
├── docs/
│   ├── arquitetura.md                   # diagrama da solução em Mermaid
│   ├── arquitetura_pipeline.png         # diagrama gerado pelo notebook
│   └── referencias/
│       ├── orientacoes-aed-1a-etapa.pdf     # enunciado da disciplina
│       └── proposta-tema-e-equipe.docx      # documento da 1ª Etapa
├── data/                           # dataset baixado (não versionado)
│   └── placas-de-transito-br-9/
│       ├── data.yaml
│       ├── train/  images/  labels/
│       ├── valid/  images/  labels/
│       └── test/   images/  labels/
└── outputs/                        # gerado pelo notebook e versionado
    ├── inventario_dataset.csv           # 1 linha por imagem: split, dimensões, nº de objetos
    ├── parametros_adotados.json         # registro completo e reprodutível da execução
    ├── parametros_adotados.md           # tabela pronta para colar na Seção 6 deste README
    ├── busca_em_grade.csv               # portas de saturação, método e área na amostra de ajuste
    ├── comparacao_limiarizacao.csv      # melhor configuração de cada método
    ├── ablacao_preprocessamento.csv     # efeito medido de CLAHE e suavização
    ├── limiares_por_imagem.csv          # limiar T de Otsu e pixels de objeto, por imagem
    ├── avaliacao_por_imagem.csv         # VP, FP, FN, erro de contagem e T, imagem a imagem
    ├── descritores_objetos.csv          # saída numérica, um objeto por linha
    ├── distribuicao_formas.csv          # classes geométricas na amostra de validação
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

**Dois tratamentos aplicados aos dados.** Nenhum deles remove imagem ou anotação da base,
os dois corrigem a forma de ler e de dividir:

1. **Rótulos em polígono.** 5 linhas (em 3 arquivos) vêm como polígono, com a classe seguida
   de pares `x y`, em vez de caixa. Lidas como caixa, viravam placas com 75% da largura da
   imagem. O `ler_rotulos` agora converte o polígono na caixa que o envolve.
2. **Divisão por trecho de gravação.** A base são quadros de câmera veicular tirados em média
   a cada 3 segundos. Sorteando imagem por imagem, a mesma placa vista um segundo depois caía
   nas duas amostras: 55% das imagens de validação tinham um quadro de ajuste a menos de 30 s.
   Agora os quadros são agrupados em trechos (um trecho novo começa após 30 s sem captura) e
   cada trecho vai inteiro para um lado só. O quadro de validação mais próximo de um de ajuste
   passou a estar a 31 s.

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
| 5 | Segmentação | Limiarização global, Otsu, Otsu restrito e adaptativa, com a global adotada | A escolha é feita por métrica, conforme a Seção 6. Nesta base os quatro empatam dentro de 0,005 de F1, e o desempate declarado é pelo método mais simples. O limiar T de Otsu de cada imagem é registrado como diagnóstico de iluminação, como pede a receita da Apostila 02 |
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

**5.2. Portas de saturação, área mínima e máxima e método de limiarização.** Estes não têm valor correto derivável da
geometria, porque governam um compromisso entre precisão e recall. O notebook mede a curva de
trade-off e escolhe um ponto segundo um critério declarado de antemão.

**Método de limiarização.** O enunciado pede "global, Otsu ou adaptativa, com justificativa
técnica da escolha". A justificativa aqui é medida: os quatro métodos rodam na mesma grade, na
mesma amostra de ajuste. O resultado é um empate:

| Método | Melhor F1 no ajuste |
|---|---|
| global | 0,120 |
| adaptativa | 0,120 |
| otsu_restrito | 0,117 |
| otsu | 0,116 |

Os quatro cabem dentro de 0,005 de F1, que é menos do que o ruído de uma amostra de 250
imagens. Deixar o vencedor sair da ordem em que a tabela foi montada seria sorte, então o
desempate é declarado de antemão: **entre métodos empatados fica o mais simples**, e a lista
`METODOS_LIMIAR` está escrita do mais simples para o mais complexo. Isso adota a limiarização
global, com o corte escolhido no estágio 1b.

O custo dessa escolha fica registrado: um corte fixo funciona nesta base, mas não é
transferível. Nas fotos que a equipe vai tirar, com outra câmera e outra luz, o valor precisa
ser recalibrado, enquanto Otsu e adaptativa se ajustam sozinhos. É a primeira coisa a refazer
ao trocar de fonte de imagem.

**O T da Apostila.** A receita da Apostila 02 imprime o limiar T de cada imagem, porque esse
número entra no relatório e mostra quando a iluminação muda muito entre imagens. A adaptativa
não tem um T único, então o notebook calcula o T do Otsu restrito sobre o mesmo mapa de
evidência, só como diagnóstico. Ele não entra na segmentação.

Cada candidato a área é expresso como "descartar o quantil `q` das placas anotadas", em vez de
um número solto de pixels. Assim o parâmetro carrega significado: `q = 0,25` no piso quer dizer
que o pipeline abre mão do quartil de placas mais distantes, cuja área é comparável à do ruído
cromático residual. O piso ainda multiplica por 0,45, fator que converte área da caixa em área
da figura inscrita.

A busca tem três estágios, no formato de busca por coordenadas. O estágio 0 percorre as portas
de saturação das faixas vermelha (60, 80, 100 e 120) e amarela (130, 150 e 170), com piso e método
provisórios. O estágio 1 percorre `q` de 0,05 a 0,60 combinado com os quatro métodos de
limiarização, e a combinação de maior F1 define o método e o piso. O estágio 1b ajusta os
parâmetros internos do método vencedor: o corte (64, 96 e 128) no caso do global, ou a janela
(31, 51 e 71 px) e o `C` (−5, −10 e −15) no caso da adaptativa. O estágio 2, já com tudo
fixado, varre o teto de área em `q` igual a 0,90, 0,95, 0,99 e sem teto.

**Duas regras de desempate, declaradas antes de rodar.** Diferenças menores que 0,005 de F1
cabem no ruído de 250 imagens, e sem uma regra elas seriam decididas pela ordem das linhas.
Por isso: uma faixa de cor só permanece no pipeline se ganhar do descarte por mais de 0,005, e
entre métodos empatados nessa mesma margem fica o mais simples. A regra das faixas mudou o
resultado: o azul ganhava do descarte por 0,001 e colocava nuvem e céu entre as detecções.

**Protocolo anti-viés.** A medição e a grade rodam sobre uma amostra de ajuste, enquanto as
métricas reportadas na Seção 8 do notebook vêm de uma amostra de validação disjunta, que não
participa de nenhuma decisão. A divisão é por trecho de gravação, e não por imagem (ver Seção
4), e a escala usada para derivar kernels e áreas também é medida só no lado de ajuste. Sem essa separação, o desempenho publicado seria otimista por
construção, já que o parâmetro teria sido escolhido no mesmo conjunto em que é avaliado. A
ordenação dos arquivos é feita de modo idêntico no Windows e no Linux, para que a mesma semente
produza as mesmas amostras no Colab e na máquina local.

O limiar em si não é constante. Na adaptativa ele muda pixel a pixel, e o T de Otsu usado como
diagnóstico é recalculado por imagem, como na receita da Apostila 02. A Seção 7 do notebook imprime `Limiar de Otsu: T | pixels de objeto: N` para cada
imagem de evidência e salva em `outputs/limiares_por_imagem.csv`, e a Seção 8 guarda o T de cada
imagem de validação, com média, desvio, mínimo e máximo.

### Valores da última execução

O bloco abaixo é gerado pela Seção 10 do notebook em `outputs/parametros_adotados.md` e está
reproduzido aqui. Ao reexecutar, substitua por aquele arquivo.

Execução de 17/09/2026, semente 42, OpenCV 5.0.0, Python 3.14.4.

| Parâmetro | Valor | Origem |
|---|---|---|
| Largura de trabalho | 640 px | Deixa os parâmetros em pixel comparáveis entre imagens |
| CLAHE (clip e grade) | 2,0 e 8×8 | Canal `L*` do LAB, preserva a matiz |
| Suavização | gaussiano 3×3 | `ímpar(0,10 × 13,5) = 3`, mais estreito que a orla da menor placa |
| Faixas de matiz | vermelho H 9 ± 6 e amarelo H 16 ± 6 | Centro e largura medidos nas anotações, a partir das âncoras do CONTRAN |
| Portas de saturação | vermelho 80 e amarelo 100 | Estágio 0 da busca em grade, por descida em coordenadas |
| Faixas descartadas | azul, verde | O verde não reuniu objetos anotados suficientes. O azul virou faixa, mas no estágio 0 ganhou do descarte por só 0,001 de F1, abaixo da margem de 0,005, e saiu |
| Kernel de abertura | 3×3 | `ímpar(0,10 × 13,5) = 3` |
| Kernel de fechamento | 7×7 | `ímpar(0,25 × 27,1) = 7` |
| Método de limiarização | `global` | Busca por coordenadas em 4 estágios, 53 configurações sobre 250 imagens de ajuste. Empate técnico entre os quatro métodos (global 0,120, adaptativa 0,120, Otsu restrito 0,117, Otsu 0,116), resolvido pela regra do mais simples |
| Limiar do método global | 96 | Estágio 1b da busca em grade, entre 64, 96 e 128 |
| `blockSize` e `C` da adaptativa | 51 px e −10 | Só entram na comparação de métodos. Valores fixos, não calibrados |
| Limiar T de Otsu (diagnóstico) | recalculado por imagem: média 74,9, desvio 18,1, de 33 a 142 | Otsu restrito sobre o mapa de evidência das 250 imagens de validação. Não entra na segmentação, serve para mostrar o quanto a iluminação muda entre imagens |
| Área mínima de contorno | 183 px² | Descarta o quantil 0,30 inferior das placas anotadas, multiplicado por 0,45 de preenchimento |
| Área máxima de contorno | 6.750 px² | Descarta o quantil superior a 0,95, como filtro de escala contra fachadas e vegetação fotografadas de perto |
| Razão de aspecto aceita | 0,35 a 2,85 | Rejeita postes, faixas e meios-fios |
| Extensão mínima | 0,35 | Rejeita contornos rendilhados, como vegetação |
| Solidez mínima | 0,70 | Toda placa normativa é convexa |

**Protocolo de avaliação.** São 250 imagens de ajuste, usadas na escolha dos parâmetros, e 250
de validação, sorteadas com semente fixa entre trechos de gravação disjuntos: 82 trechos de um
lado e 97 do outro. O casamento entre detecção e anotação usa IoU de no mínimo 0,30.

**Desempenho da linha de base clássica**, medido na amostra de validação retida, com 250
imagens:

| Métrica | Valor |
|---|---|
| Precisão | 0,137 |
| Recall | 0,157 |
| F1 | 0,146 |
| F1 na amostra de ajuste | 0,138 |
| Diferença entre ajuste e validação | −0,008 |
| Erro absoluto médio de contagem | 0,92 objeto por imagem |
| Contagem exata | 81 de 250 imagens |

A diferença entre ajuste e validação ficou negativa, ou seja, o resultado na amostra retida
foi um pouco melhor. Não é erro: com a divisão por trecho, os dois lados têm conteúdo
diferente e nada garante que o lado retido seja o mais difícil.

**A contagem precisa de referência, e não passa nela.** Das 250 imagens avaliadas, 226 têm
exatamente uma placa anotada. Quem chutar "1 placa" em toda imagem erra bem menos que o
pipeline:

| Contagem | Erro absoluto médio | Contagens exatas |
|---|---|---|
| Pipeline | 0,92 | 81 de 250 |
| Chutar sempre 1 placa | **0,12** | **226 de 250** |

A conclusão está no notebook e vale repetir aqui: nesta base o erro de contagem não serve
para avaliar o pipeline. A leitura que vale é a de detecção, com precisão, recall e F1.

**Limiar T nas imagens de evidência (Seção 7).** Mesmo formato da receita da Apostila 02,
salvo em `outputs/limiares_por_imagem.csv`. O T é o do Otsu restrito, calculado só como
diagnóstico; os pixels de objeto são os da máscara final:

| Imagem | Limiar de Otsu (T) | Pixels de objeto |
|---|---|---|
| captura_2023-09-29_16-19-23 | 53 | 793 |
| captura_2023-10-10_17-49-29 | 104 | 88.804 |
| captura_2023-10-05_18-19-04 | 92 | 130.788 |
| captura_2023-09-29_16-19-55 | 50 | 1.064 |
| captura_2023-10-02_14-58-39 | 68 | 64.924 |
| captura_2023-09-28_15-41-02 | 86 | 106.068 |

O T vai de 50 a 104 nessas seis imagens (média 75,5, desvio 21,9). É uma variação grande, e
é exatamente o que a Apostila diz que esse número serve para mostrar: a iluminação e a
quantidade de cor na cena mudam muito entre os quadros da câmera veicular. Nas 250 imagens
de validação a variação é ainda maior, de 33 a 142.

Esses números são baixos e a Seção 9 explica por quê, ponto a ponto. Dois fatores pesam mais que
os demais. O primeiro é a densidade de anotação da base: são 2.305 objetos em 1.973 imagens
anotadas, cerca de um objeto por imagem, e 990 imagens sem anotação nenhuma. As figuras de
`outputs/figuras/05_mosaico_deteccoes.png` mostram cenas com várias placas visíveis e apenas uma
anotada, o que faz o protocolo contar como falso positivo uma detecção correta. O segundo é a
escala: um décimo das placas anotadas tem menos de 14 px de lado equivalente.

**Ressalva medida sobre o pré-processamento.** A ablação mostra que o CLAHE se paga quando há
suavização gaussiana, com ganho de 0,008 no F1, mas atrapalha junto com a mediana, com perda
de 0,013. O gaussiano de 3×3 foi mantido por ser a melhor combinação medida e por ser etapa
exigida pelo Checkpoint 1. A
tabela completa está em `outputs/ablacao_preprocessamento.csv`.

---

## 7. Como as decisões foram validadas

Cada escolha do pipeline tem uma medição por trás, e não uma afirmação.

| Seção | O que mede | Saída |
|---|---|---|
| 5.0 | Matiz dominante de cada classe anotada, formação das faixas e separação de placa contra fundo por saturação | `figuras/00b_calibracao_cromatica.png`, `figuras/00c_recortes_por_faixa.png` |
| 5.2 | Grade de portas de saturação, método de limiarização e área, na amostra de ajuste | `busca_em_grade.csv`, `figuras/01_escolha_de_parametros.png` |
| 6 | Ablação com CLAHE ligado e desligado, combinado com gaussiano, mediana e nenhum filtro | `ablacao_preprocessamento.csv`, `figuras/02_ablacao_preprocessamento.png` |
| 6.1 | Histograma do mapa de evidência com o corte de cada método sobreposto | `figuras/03_histograma_limiares.png` |
| 7 | Pipeline etapa a etapa em várias imagens sorteadas com semente fixa | `figuras/04_pipeline_*.png`, `figuras/05_mosaico_deteccoes.png` |
| 7 | Limiar T de Otsu (diagnóstico) e pixels de objeto por imagem, como na receita da Apostila 02 | `limiares_por_imagem.csv` |
| 8 | Precisão, recall, F1, erro de contagem e T por imagem na amostra retida | `avaliacao_por_imagem.csv`, `figuras/07_avaliacao_quantitativa.png` |

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
| 2 | Fernanda Andrade | 20241013700048 | Pré-processamento: conversão de espaços de cor, correção de iluminação, filtragem espacial e análise de histograma | 3.2 a 3.5 e 6.1 |
| 3 | Alisson Leonardo | 20241013700170 | Segmentação por cor e limiarização, operações morfológicas e extração de contornos | 3.6 a 3.8 e 6 |
| 4 | Vitor Manoel | 20241013700307 | Descritores geométricos, relatório técnico, figuras comparativas, README e documentação de reprodução | 3.8, 7 a 11 |

---

## 9. Limitações conhecidas

1. **A anotação da base é esparsa, e isso deprime a precisão medida.** São 2.305 objetos em
   1.973 imagens anotadas, cerca de um por imagem, além de 990 imagens sem anotação nenhuma. As
   cenas costumam ter mais placas visíveis do que anotadas, e cada detecção correta de uma placa
   não anotada entra na conta como falso positivo. O mosaico de detecções mostra o efeito. A
   precisão de 0,137 é, portanto, um piso, e não uma medida limpa do pipeline.
2. **Falsos positivos de mesma cromaticidade.** Lanternas traseiras, veículos vermelhos, toldos,
   solo exposto e grama seca muito saturada compartilham matiz e saturação com as placas. Nesta
   base o efeito é forte: com a porta de saturação em 80, mais da metade dos pixels de fundo
   dentro da faixa vermelha sobrevive, contra menos de um décimo em bases de melhor qualidade
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
   código sinaliza a ambiguidade em vez de arbitrar. Nesta execução 89,4% dos objetos saíram
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
