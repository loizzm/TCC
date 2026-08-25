# Ambiente da máquina da triagem (Bloco 6)

Capturado em 2026-08-25 13:17 -03, no commit `b963f6e`.
Existe porque os Rulings 12 (OCR sensível à versão do tesseract) e 14
(trajetória de treino sensível à versão do torch) do `HANDOFF_P2_6.md`
tornam estes números parte do resultado, não detalhe de infraestrutura.

## Sistema

| item | valor |
|---|---|
| SO | Fedora Linux 43 (Workstation Edition) |
| kernel | 7.1.9-100.fc43.x86_64 |
| CPU | 13th Gen Intel(R) Core(TM) i7-13620H |
| núcleos / threads | 16 threads |
| RAM | 15 GiB |
| swap | 7 GiB |
| GPU | nenhuma (torch.cuda.is_available() == False) |
| Python | 3.11.15 |

## OCR (Ruling 12)

```
tesseract 5.5.3
 leptonica-1.87.0
  libgif 5.2.2 : libjpeg 6b (libjpeg-turbo 3.1.2) : libpng 1.6.58 : libtiff 4.7.2 : zlib 1.3.1.zlib-ng : libwebp 1.6.0
```

## Codecs de imagem (Ruling 11)

O dataset é escrito pelo backend Agg do matplotlib e lido pelo Pillow. A versão
de zlib abaixo é a que comprime os `image.png`/`mask.png` — é ela que faz os
sha256 diferirem entre máquinas com os MESMOS pixels.

```
zlib (runtime do Python): 1.3.1.zlib-ng
Pillow: 12.3.0
libpng (via Pillow): 1.3.1.zlib-ng
```

## Pacotes Python instalados (`pip freeze`)

```
contourpy==1.3.3
cycler==0.12.1
filelock==3.32.3
fonttools==4.63.0
fsspec==2026.7.0
ImageIO==2.37.4
iniconfig==2.3.0
Jinja2==3.1.6
joblib==1.5.3
kiwisolver==1.5.0
lazy-loader==0.5
MarkupSafe==3.0.3
matplotlib==3.11.1
mpmath==1.3.0
narwhals==2.24.0
networkx==3.6.1
numpy==2.4.6
opencv-python-headless==5.0.0.93
packaging==26.3
pillow==12.3.0
pluggy==1.6.0
Pygments==2.20.0
pyparsing==3.3.2
pytesseract==0.3.13
pytest==9.1.1
python-dateutil==2.9.0.post0
scikit-image==0.26.0
scikit-learn==1.9.0
scipy==1.17.1
six==1.17.0
sympy==1.14.0
threadpoolctl==3.6.0
tifffile==2026.3.3
torch==2.13.0+cpu
typing_extensions==4.16.0
```

## Configuração de treino usada nos pilotos

| item | valor |
|---|---|
| `OMP_NUM_THREADS` | 6 (ótimo medido; 10 é 6% e 16 é 17% mais lento) |
| `torch.get_num_threads()` padrão | 10 |
| seed | `torch.manual_seed(20260817)` |
| batch | 8 |
| `num_workers` treino / val | 4 / 2 |
