# -*- coding: utf-8 -*-
"""Pipeline de inferência da N2: imagem -> pré-processamento -> modelo -> placas encontradas.

    python inferir.py foto.jpg                     # uma imagem
    python inferir.py pasta/ --saida resultados    # todas as imagens de uma pasta

Para cada imagem, imprime as placas encontradas (código, descrição, confiança e caixa) e grava
na pasta de saída a imagem com as caixas desenhadas e um JSON com o mesmo conteúdo.

O modelo (modelos/modelo_n2.pt) e o limiar de confiança (modelos/modelo_n2.json) saem do
notebook notebooks/checkpoint2_modelo.ipynb: o modelo na Seção 7, o limiar na Seção 8,
escolhido na validação.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np
import torch
from ultralytics import YOLO
from ultralytics.data.augment import LetterBox
from ultralytics.utils import ops

RAIZ = Path(__file__).resolve().parent
ARQ_MODELO = RAIZ / "modelos" / "modelo_n2.pt"
ARQ_CONFIG = RAIZ / "modelos" / "modelo_n2.json"
EXTENSOES = (".jpg", ".jpeg", ".png", ".bmp", ".webp")

# Significado das placas pelo Manual Brasileiro de Sinalização de Trânsito (CONTRAN). Os outros
# códigos (I-4, S-14, LOC-6, MP, E-5, RQ, ESP-20, TUR-4) são da própria base e saem só com o código.
DESCRICAO = {
    "R-1": "Parada obrigatória",
    "R-2": "Dê a preferência",
    "R-5a": "Proibido retornar à esquerda",
    "R-7": "Proibido ultrapassar",
    "R-19": "Velocidade máxima permitida",
    "R-24b": "Passagem obrigatória",
    "R-27": "Ônibus, caminhões e veículos de grande porte mantenham-se à direita",
    "A-1b": "Curva acentuada à direita",
    "A-2a": "Curva à esquerda",
    "A-2b": "Curva à direita",
    "A-6": "Cruzamento de vias",
    "A-7b": "Via lateral à direita",
    "A-14": "Semáforo à frente",
    "A-21c": "Estreitamento de pista à direita",
    "A-21e": "Alargamento de pista à direita",
    "A-22": "Ponte estreita",
    "A-32b": "Passagem sinalizada de pedestres",
    "A-42a": "Início de pista dupla",
    "outra_placa": "Placa de uma das 41 classes raras, agrupadas numa só no treino",
}


@dataclass
class Placa:
    classe: str
    descricao: str
    confianca: float
    caixa: tuple[int, int, int, int]    # x0, y0, x1, y1, em pixels da imagem original


def carregar_modelo(arq_modelo: Path = ARQ_MODELO, arq_config: Path = ARQ_CONFIG) -> tuple[YOLO, dict]:
    """O modelo treinado e a configuração de inferência gravada pelo notebook."""
    if not arq_modelo.exists():
        sys.exit(f"Modelo não encontrado: {arq_modelo}. Ele sai da Seção 7 do notebook da N2.")
    config = json.loads(arq_config.read_text(encoding="utf-8")) if arq_config.exists() else {}
    modelo = YOLO(str(arq_modelo))
    config.setdefault("imgsz", int(modelo.ckpt.get("train_args", {}).get("imgsz", 1280)))
    config.setdefault("limiar_confianca", 0.25)
    return modelo, config


def ler_imagem(caminho: Path) -> np.ndarray:
    """Etapa 1, entrada: a imagem em BGR, com 3 canais (cinza e transparência são convertidos)."""
    dados = np.fromfile(str(caminho), dtype=np.uint8)    # aceita acento no caminho, no Windows
    imagem = cv2.imdecode(dados, cv2.IMREAD_COLOR)
    if imagem is None:
        raise ValueError(f"não é uma imagem legível: {caminho}")
    return imagem


def preprocessar(imagem_bgr: np.ndarray, imgsz: int, passo: int = 32) -> torch.Tensor:
    """Etapa 2, o mesmo pré-processamento que o modelo teve no treino e na validação.

    1. Letterbox: redimensiona mantendo a proporção, com o lado maior em `imgsz`, e completa as
       bordas com cinza (114) até um múltiplo de `passo`, o passo da rede.
    2. BGR para RGB e de altura x largura x canal para canal x altura x largura.
    3. Valores de 0-255 para 0-1, num lote de uma imagem.

    Nenhum filtro clássico entra aqui: o modelo aprendeu com as imagens cruas, e um CLAHE só na
    inferência mudaria a imagem que ele recebe.
    """
    ajustada = LetterBox(new_shape=(imgsz, imgsz), auto=True, stride=passo)(image=imagem_bgr)
    chw = ajustada[..., ::-1].transpose(2, 0, 1)
    return torch.from_numpy(np.ascontiguousarray(chw)).float().div(255).unsqueeze(0)


def detectar(modelo: YOLO, entrada: torch.Tensor, forma_original: tuple, limiar: float) -> list[Placa]:
    """Etapas 3 e 4, modelo e saída: placas acima do limiar, com as caixas na imagem original.

    O YOLO26 é ponta a ponta: a rede já entrega uma caixa por objeto, sem NMS depois.
    """
    resultado = modelo.predict(entrada, conf=limiar, verbose=False)[0]
    if len(resultado.boxes) == 0:
        return []
    caixas = ops.scale_boxes(entrada.shape[2:], resultado.boxes.xyxy.clone(), forma_original).cpu().numpy()
    placas = []
    for caixa, conf, cls in zip(caixas, resultado.boxes.conf.tolist(), resultado.boxes.cls.tolist()):
        nome = modelo.names[int(cls)]
        placas.append(Placa(nome, DESCRICAO.get(nome, ""), round(conf, 3), tuple(int(round(v)) for v in caixa)))
    return sorted(placas, key=lambda p: -p.confianca)


def inferir(modelo: YOLO, config: dict, caminho: Path, limiar: float | None = None) -> tuple[np.ndarray, list[Placa]]:
    """O pipeline inteiro para uma imagem: devolve a imagem lida e as placas encontradas."""
    imagem = ler_imagem(caminho)
    entrada = preprocessar(imagem, config["imgsz"], int(modelo.model.stride.max()))
    limiar = config["limiar_confianca"] if limiar is None else limiar
    return imagem, detectar(modelo, entrada, imagem.shape, limiar)


def cor_da_classe(nome: str) -> tuple[int, int, int]:
    """BGR: vermelho para regulamentação, âmbar para advertência, azul para os demais códigos."""
    if nome == "outra_placa":
        return (150, 150, 150)
    return {"R": (40, 40, 220), "A": (0, 180, 255)}.get(nome[0], (220, 120, 30))


def desenhar(imagem_bgr: np.ndarray, placas: list[Placa]) -> np.ndarray:
    """Cópia da imagem com cada placa na caixa, com código e confiança."""
    saida = imagem_bgr.copy()
    espessura = max(2, round(max(saida.shape[:2]) / 600))
    for p in placas:
        x0, y0, x1, y1 = p.caixa
        cor = cor_da_classe(p.classe)
        cv2.rectangle(saida, (x0, y0), (x1, y1), cor, espessura)
        texto = f"{p.classe} {p.confianca:.2f}"
        (larg, alt), base = cv2.getTextSize(texto, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        topo = y0 - alt - base - 4 if y0 - alt - base - 4 > 0 else y1 + 2
        cv2.rectangle(saida, (x0, topo), (x0 + larg + 6, topo + alt + base + 4), cor, -1)
        cv2.putText(saida, texto, (x0 + 3, topo + alt + 2), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (255, 255, 255), 2, cv2.LINE_AA)
    return saida


def listar_entradas(caminhos: list[str]) -> list[Path]:
    """Arquivos de imagem passados diretamente ou dentro das pastas passadas."""
    imagens = []
    for c in map(Path, caminhos):
        if c.is_dir():
            imagens += sorted(p for p in c.rglob("*") if p.suffix.lower() in EXTENSOES)
        elif c.exists():
            imagens.append(c)
        else:
            print(f"[aviso] não encontrado: {c}")
    return imagens


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="Detecta e classifica placas de trânsito numa imagem.")
    parser.add_argument("entradas", nargs="+", help="imagens ou pastas com imagens")
    parser.add_argument("--saida", default="saida_inferencia", help="pasta para as imagens anotadas e os JSON")
    parser.add_argument("--limiar", type=float, default=None,
                        help="confiança mínima (padrão: a escolhida na validação, em modelos/modelo_n2.json)")
    parser.add_argument("--modelo", type=Path, default=ARQ_MODELO, help="peso do modelo (.pt)")
    parser.add_argument("--sem-arquivos", action="store_true", help="só imprime, sem gravar imagem e JSON")
    args = parser.parse_args()

    modelo, config = carregar_modelo(args.modelo)
    limiar = config["limiar_confianca"] if args.limiar is None else args.limiar
    imagens = listar_entradas(args.entradas)
    if not imagens:
        sys.exit("Nenhuma imagem para processar.")
    pasta = Path(args.saida)
    if not args.sem_arquivos:
        pasta.mkdir(parents=True, exist_ok=True)
    print(f"Modelo {args.modelo.name} | entrada de {config['imgsz']} px | limiar de confiança {limiar:.2f}\n")

    for caminho in imagens:
        inicio = time.perf_counter()
        imagem, placas = inferir(modelo, config, caminho, limiar)
        ms = (time.perf_counter() - inicio) * 1000
        print(f"{caminho.name}: {len(placas)} placa(s) em {ms:.0f} ms")
        for p in placas:
            print(f"  {p.classe:<12} {p.confianca:.2f}  caixa {p.caixa}  {p.descricao}")
        if args.sem_arquivos:
            continue
        cv2.imencode(".jpg", desenhar(imagem, placas))[1].tofile(str(pasta / f"{caminho.stem}_placas.jpg"))
        contagem = {}
        for p in placas:
            contagem[p.classe] = contagem.get(p.classe, 0) + 1
        (pasta / f"{caminho.stem}_placas.json").write_text(json.dumps({
            "imagem": caminho.name, "largura": imagem.shape[1], "altura": imagem.shape[0],
            "modelo": args.modelo.name, "limiar_confianca": limiar, "contagem": contagem,
            "placas": [asdict(p) for p in placas],
        }, ensure_ascii=False, indent=2), encoding="utf-8")
    if not args.sem_arquivos:
        print(f"\nImagens anotadas e JSON em: {pasta}")


if __name__ == "__main__":
    main()
