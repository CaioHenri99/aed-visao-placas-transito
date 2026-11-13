# Saídas da N2

Tudo aqui é gravado pelo notebook [`checkpoint2_modelo.ipynb`](../../notebooks/checkpoint2_modelo.ipynb)
e regravado quando a seção correspondente roda de novo. A exceção são os arquivos `inspecao_*.csv`
e `vazamento_confirmado.csv`: eles guardam julgamentos visuais, feitos uma vez, que o notebook só lê.

## Divisão e base de treino (Seções 2 a 4)

| Arquivo | Conteúdo |
|---|---|
| `divisao_por_imagem.csv` | Uma linha por imagem: dia, instante, trecho, conjunto do Roboflow e conjunto adotado. É a divisão que o treino e a linha de base usam |
| `candidatos_divisao.csv` | As distribuições dos 8 dias que passam nas condições da Seção 2.4, na ordem da regra; a primeira é a adotada |
| `contagem_por_classe.csv` | Objetos por classe original (67, sem o `Del`) em cada conjunto |
| `cortes_de_classe.csv` | Classes e objetos cobertos para cada mínimo de exemplos no treino |
| `classes_do_modelo.csv` | De cada classe original para a classe do modelo (26 placas e `outra_placa`) |
| `contagem_classes_do_modelo.csv` | Objetos por classe do modelo em cada conjunto |
| `base_yolo_resumo.csv` | Imagens, imagens de fundo e objetos de cada conjunto na base do YOLO |

## Linha de base e aumento de dados (Seções 5 e 6)

| Arquivo | Conteúdo |
|---|---|
| `linha_de_base.json` | Precisão, recall e F1 do pipeline da N1 no teste, com IoU ≥ 0,30 e ≥ 0,50 |
| `linha_de_base_por_imagem.csv` | Placas anotadas, detecções e acertos da linha de base, imagem a imagem |
| `linha_de_base_por_dia.csv` | A linha de base em cada um dos 8 dias |
| `aumento_de_dados.csv` | Cada transformação: valor adotado, padrão do Ultralytics e motivo |

## Treino e diagnóstico (Seções 7 e 8)

| Arquivo | Conteúdo |
|---|---|
| `treino/escolha_do_modelo.json` | Modelo escolhido, regra de escolha, classes, configuração do treino e métricas de validação dos dois tamanhos |
| `treino/comparacao_validacao.csv` | Nano e small na validação: mAP50, mAP50-95, precisão, recall e tempo por imagem |
| `treino/vao_treino_validacao.csv` | Perdas de treino e de validação nas épocas 5, 10 e 20, e o vão entre elas |
| `treino/yolo26n/`, `treino/yolo26s/` | O que o Ultralytics gravou em cada treino: `results.csv` (uma linha por época), curvas, configuração (`args.yaml`) e lotes de exemplo. As matrizes de confusão dessas pastas não são usadas (Seção 8.3) |
| `limiar_validacao.csv` | Precisão, recall e F1 na validação para cada limiar de confiança |
| `predicoes_validacao.csv` | Todas as detecções do modelo na validação com confiança ≥ 0,05 |
| `matriz_confusao_validacao.csv` | A matriz de confusão da Seção 8.3, em número de placas |
| `validacao_por_classe.csv` | Por classe: placas anotadas, acertos, perdidas, trocadas e alarmes falsos |
| `inspecao_alarmes_falsos_validacao.csv` | Julgamento visual de cada um dos 45 alarmes falsos da validação |

## Teste e vazamento (Seção 9)

| Arquivo | Conteúdo |
|---|---|
| `predicoes_teste.csv` | Todas as detecções do modelo no teste com confiança ≥ 0,05 |
| `teste_modelo_vs_linha_de_base.csv` | Precisão, recall e F1 do modelo e da linha de base, com o mesmo gabarito e o mesmo casamento |
| `teste_por_dia.csv` | A mesma comparação em cada dia de teste |
| `teste_por_classe.csv` | Recall com a classe certa e AP50 de cada classe no teste |
| `teste_resumo.json` | Resumo do teste: mAP50, mAP50-95, F1 e limiar usado |
| `inspecao_alarmes_falsos_teste_amostra.csv` | Julgamento visual de 54 alarmes falsos do teste, sorteados |
| `vazamento_pares.csv` | Para cada placa de validação e teste, a placa de treino com a cena mais parecida (SIFT e RANSAC) |
| `inspecao_vazamento.csv` | Julgamento visual dos 40 pares mais parecidos: mesmo lugar ou não |
| `vazamento_confirmado.csv` | As 5 placas de validação e teste que também estão no treino |
| `vazamento_impacto.csv` | Métricas com e sem as imagens dessas 5 placas |
| `custo_linha_de_base_vs_modelo.csv` | Acerto, tempo por imagem (CPU e GPU) e tamanho da linha de base e do modelo |

## Figuras (`figuras/`)

| Figura | Seção | O que mostra |
|---|---|---|
| `01_vazamento_roboflow.jpg` | 2.2 | Fotos quase iguais que a divisão do Roboflow pôs em conjuntos diferentes |
| `02_linha_do_tempo_divisao.png` | 2.5 | A divisão do Roboflow e a adotada, foto a foto, ao longo de cada dia |
| `03_objetos_por_classe.png` | 3 | Objetos por classe em cada conjunto, em escala log |
| `04_aumento_de_dados.jpg` | 6 | Por que não espelhar, e os extremos de brilho e saturação adotados |
| `05_curvas_de_treino.png` | 8.1 | Perdas de treino e validação e mAP50 por época, nano e small |
| `06_limiar_de_confianca.png` | 8.2 | Precisão, recall e F1 na validação em função do limiar |
| `07_matriz_de_confusao_validacao.png` | 8.3 | Matriz de confusão da validação |
| `08_erros_validacao.jpg` | 8.3 | Exemplos sorteados de cada tipo de erro na validação |
| `09_alarmes_falsos_validacao.jpg` | 8.3 | Os 45 alarmes falsos da validação |
| `10_erros_teste.jpg` | 9.3 | Exemplos sorteados de cada tipo de erro no teste |
| `11_alarmes_falsos_teste.jpg` | 9.3 | 54 alarmes falsos do teste, sorteados |
| `12_mesma_placa_em_dias_diferentes.jpg` | 9.4 | Os três pares confirmados de mesma placa em dias de conjuntos diferentes |
| `13_pipeline_de_inferencia.jpg` | 10 | Entrada, imagem pré-processada e saída do pipeline de inferência |
| `14_inferencia_exemplos.jpg` | 10 | Saída do pipeline em quatro imagens de teste sorteadas |
