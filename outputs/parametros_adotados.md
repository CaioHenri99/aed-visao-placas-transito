## Parâmetros adotados

Gerado automaticamente pelo notebook em 2026-09-10 14:38:59 (semente 42, OpenCV 5.0.0, Python 3.14.4).

| Parâmetro | Valor | Origem |
|---|---|---|
| Largura de trabalho | 640 px | Padroniza a escala em pixels entre imagens |
| CLAHE (clip / grade) | 2.0 / 8×8 | Canal L* do LAB, preserva a matiz |
| Suavização | gaussiano 3×3 | `impar(0.10 x 13.4) = 3` (mais estreito que a orla da menor placa) |
| Faixas de matiz | vermelho (H 10 +/- 6), amarelo (H 19 +/- 6) | Centro e largura medidos nas anotações (Seção 5.0), a partir das âncoras do CONTRAN |
| **Portas de saturação** | **vermelho 100, amarelo 160** | Estágio 0 da busca em grade, por descida em coordenadas |
| Faixas descartadas | azul | O estágio 0 mediu que custam mais em falsos positivos do que rendem em detecções |
| **Kernel de abertura** | **3×3** | `impar(0.10 x 13.4) = 3` |
| **Kernel de fechamento** | **7×7** | `impar(0.25 x 29.6) = 7` |
| **Método de limiarização** | **adaptativa** | busca por coordenadas em 3 estagios, 50 configuracoes sobre 250 imagens de ajuste, maior F1 com IoU >= 0.30 |
| **Área mínima de contorno** | **199 px²** | Descarta o quantil 0.30 inferior das placas anotadas (× 0,45 de preenchimento) |
| **Área máxima de contorno** | **8699 px²** | Descarta o quantil superior a 0.99: filtro de escala contra fachadas e vegetação fotografadas de perto |
| Razão de aspecto aceita | 0.35–2.85 | Rejeita postes, faixas e meios-fios |
| Extensão mínima | 0.35 | Rejeita contornos rendilhados (vegetação) |
| Solidez mínima | 0.7 | Toda placa normativa é convexa |

### Dataset

- Fonte: Roboflow Universe / stefano-tommasini-coelho-euf67/placas-de-transito-br versao 9
- 2963 imagens · 2305 objetos anotados · 68 classes
- Dimensão mediana: 640x640 px

### Protocolo de avaliação

- 250 imagens de **ajuste** (escolha dos parâmetros) e 250 de **validação**, disjuntas, sorteadas com semente fixa
- Casamento detecção ↔ anotação por IoU ≥ 0.3

### Desempenho da linha de base clássica (amostra de validação retida)

- Precisão 0.082 · Recall 0.076 · F1 0.079 (250 imagens)
- Erro absoluto médio de contagem: 0.88 objetos por imagem
- Contagem exata em 84 de 250 imagens