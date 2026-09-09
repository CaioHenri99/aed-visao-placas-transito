## Parâmetros adotados

Gerado automaticamente pelo notebook em 2026-09-09 16:49:26 (semente 42, OpenCV 5.0.0, Python 3.14.4).

| Parâmetro | Valor | Origem |
|---|---|---|
| Largura de trabalho | 640 px | Padroniza a escala em pixels entre imagens |
| CLAHE (clip / grade) | 2.0 / 8×8 | Canal L* do LAB, preserva a matiz |
| Suavização | gaussiano 5×5 | Menor que a placa típica; precede a limiarização |
| **Kernel de abertura** | **3×3** | `impar(0.10 x 10.3) = 3` |
| **Kernel de fechamento** | **9×9** | `impar(0.25 x 34.5) = 9` |
| **Método de limiarização** | **otsu_restrito** | busca por coordenadas em 2 estagios, 32 configuracoes sobre 250 imagens de ajuste, maior F1 com IoU >= 0.30 |
| **Área mínima de contorno** | **536 px²** | Descarta o quantil 0.50 inferior das placas anotadas (× 0,45 de preenchimento) |
| **Área máxima de contorno** | **79520 px²** | Descarta o quantil superior a 0.99 — ataca céu, fachadas e vegetação (alcance medido na Seção 5.2) |
| Razão de aspecto aceita | 0.35–2.85 | Rejeita postes, faixas e meios-fios |
| Extensão mínima | 0.35 | Rejeita contornos rendilhados (vegetação) |
| Solidez mínima | 0.7 | Toda placa normativa é convexa |

### Dataset

- Fonte: Roboflow Universe / paic/placas-de-transito-1jw8i versao 2
- 5297 imagens · 8627 objetos anotados · 75 classes
- Dimensão mediana: 1280x720 px

### Protocolo de avaliação

- 250 imagens de **ajuste** (escolha dos parâmetros) e 250 de **validação**, disjuntas, sorteadas com semente fixa
- Casamento detecção ↔ anotação por IoU ≥ 0.3

### Desempenho da linha de base clássica (amostra de validação retida)

- Precisão 0.314 · Recall 0.221 · F1 0.259 (250 imagens)
- Erro absoluto médio de contagem: 1.25 objetos por imagem
- Contagem exata em 70 de 250 imagens