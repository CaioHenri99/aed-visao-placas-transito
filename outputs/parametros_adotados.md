## Parâmetros adotados

Gerado automaticamente pelo notebook em 2026-09-22 16:49:34 (semente 42, OpenCV 5.0.0, Python 3.14.4).

| Parâmetro | Valor | Origem |
|---|---|---|
| Largura de trabalho | 1248 px | Deixa os parâmetros em pixel comparáveis entre imagens |
| CLAHE (clip / grade) | 2.0 / 8×8 | Canal L* do LAB, preserva a matiz |
| Suavização | gaussiano 3×3 | `impar(0.10 x 19.2) = 3` (mais estreito que a orla da placa menor) |
| Faixas de matiz | vermelho (H 8 +/- 6), amarelo (H 16 +/- 6) | Centro e largura medidos nas anotações da Seção 5.0, partindo das âncoras do CONTRAN |
| **Portas de saturação** | **vermelho 100, amarelo 120** | Estágio 0 da busca em grade, por descida em coordenadas |
| Faixas descartadas | azul | O estágio 0 mediu que elas custam mais em falso positivo do que rendem em detecção |
| **Kernel de abertura** | **3×3** | `impar(0.10 x 19.2) = 3` |
| **Kernel de fechamento** | **9×9** | `impar(0.25 x 37.5) = 9` |
| **Método de limiarização** | **global** | busca por coordenadas em 4 estagios (0, 1, 1b e 2), 53 configuracoes sobre 250 imagens de ajuste, maior F1 com IoU >= 0.30. Empate técnico entre 4 métodos (global, otsu, otsu_restrito, adaptativa), resolvido pelo mais simples |
| Limiar T de Otsu (diagnóstico) | recalculado por imagem: média 80.1, desvio 16.7, de 37.0 a 138.0 | Otsu restrito sobre o mapa de evidência, só como diagnóstico, não entra na segmentação. O desvio mostra o quanto a iluminação varia entre imagens. 1 imagem(ns) sem nenhum pixel com evidência de cor ficam fora da conta |
| Limiar do método global | 96 | Estágio 1b da busca em grade. Corte único pra imagem toda |
| `blockSize` da adaptativa | 51 px | Só entra na comparação de métodos. Valor fixo, não calibrado |
| `C` da adaptativa | -10 | Só entra na comparação de métodos. Valor fixo, não calibrado |
| **Área mínima de contorno** | **479 px²** | Descarta o quantil 0.40 inferior das placas anotadas (× 0,45 de preenchimento) |
| **Área máxima de contorno** | **12887 px²** | Descarta o quantil superior a 0.95: é um filtro de escala contra fachada e vegetação fotografadas de perto |
| Razão de aspecto aceita | de 0.35 a 2.85 | Rejeita postes, faixas e meios-fios |
| Extensão mínima | 0.35 | Rejeita contorno rendilhado, normalmente vegetação |
| Solidez mínima | 0.7 | Toda placa normativa é convexa |

### Dataset

- Fonte: Roboflow Universe / caios-workspace-01wh5/placas-de-transito-br-wq5tp versao 1
- 2963 imagens · 2305 objetos anotados · 68 classes
- Dimensão mediana: 1270x636 px

### Protocolo de avaliação

- 250 imagens de **ajuste** (escolha dos parâmetros) e 250 de **validação**, sorteadas com semente fixa
- Divisão por trecho de gravação (novo trecho apos 30 s sem captura): 82 trechos de ajuste e 97 de validação. O quadro de validação mais próximo de um de ajuste está a 31 s
- Casamento detecção ↔ anotação por IoU ≥ 0.3

### Desempenho da linha de base clássica (amostra de validação retida)

- Precisão 0.236 · Recall 0.175 · F1 0.201 (250 imagens)
- Erro absoluto médio de contagem: 0.87 objetos por imagem (chutar sempre 1 placa(s) dá 0.12)
- Contagem exata em 75 de 250 imagens