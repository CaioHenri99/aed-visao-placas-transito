# Modelo da N2

`modelo_n2.pt` é o detector de placas de trânsito escolhido no Checkpoint 2, e `modelo_n2.json` guarda
a configuração de inferência dele. Os dois são gravados pelo notebook
[`checkpoint2_modelo.ipynb`](../notebooks/checkpoint2_modelo.ipynb): o peso na Seção 7, o limiar na
Seção 8.2. O [`inferir.py`](../inferir.py) lê os dois.

| Item | Valor |
|---|---|
| Arquitetura | YOLO26s (Ultralytics 8.4.182), cerca de 10 milhões de parâmetros, ponta a ponta (sem NMS) |
| Ponto de partida | Pesos pré-treinados no COCO, ajustados à base inteira (transferência de aprendizado) |
| Entrada | Imagem colorida, redimensionada para 1.280 px no lado maior, mantendo a proporção |
| Saída | Caixa, classe e confiança de cada placa |
| Classes | 27: 26 códigos de placa com pelo menos 10 exemplos no treino e `outra_placa`, que junta 41 classes raras. A lista está em `modelo_n2.json` |
| Dados | [Placas de Trânsito BR](https://universe.roboflow.com/caios-workspace-01wh5/placas-de-transito-br-wq5tp/dataset/1), câmera veicular, divisão por dia: treino com 5 dias (1.234 imagens na base do YOLO), validação 28/09 (399) e teste 10/10 e 11/10 (403) |
| Treino | 20 épocas no Google Colab (GPU T4), em 05/11/2026; configuração completa em `outputs/n2/treino/escolha_do_modelo.json` |
| Limiar de confiança | 0,41, o de maior F1 na validação exigindo a classe certa e IoU ≥ 0,50 |

## Desempenho

| Conjunto | mAP50 | mAP50-95 | F1 de detecção | F1 com a classe certa |
|---|---|---|---|---|
| Validação | 0,770 | 0,443 | 0,806 | 0,731 |
| Teste | 0,736 | 0,478 | 0,713 | 0,639 |

F1 com IoU ≥ 0,50 e limiar de 0,41. No mesmo teste, o pipeline clássico da N1 tem F1 de detecção
de 0,108.

## Limitações conhecidas

- **A precisão medida é um piso.** A base deixou muitas placas sem anotação, e o modelo é punido
  quando as encontra: 49 de 54 alarmes falsos sorteados no teste são placas reais.
- **Placas pequenas.** Abaixo de 25 px de lado, o acerto cai de 76% para 60% (validação).
- **Classes raras e parecidas.** Losangos amarelos de pista (`A-21e`, `A-22`, `A-42a`, `A-6`) se
  confundem entre si; cada um tem de 10 a 14 exemplos no treino.
- **Domínio.** Fotos diurnas de uma única câmera veicular, em estradas e cidades da serra gaúcha. À
  noite, com chuva ou com outra câmera, o desempenho não foi medido.
- **Vazamento pequeno.** 5 placas da validação e do teste também aparecem no treino (notebook,
  Seção 9.4), sem efeito mensurável nas métricas.

## Licença

O modelo parte de pesos do Ultralytics YOLO, distribuídos sob a licença AGPL-3.0, que também vale
para este peso derivado.
