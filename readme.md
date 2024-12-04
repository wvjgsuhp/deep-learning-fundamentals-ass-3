## Prerequisites

1.  Python 3.12
2.  [CUDA](https://www.nvidia.com/en-au/data-center/gpu-accelerated-applications/tensorflow/)

## Dependencies

1.  Create environment.

    ```bash
    python3 -m venv env
    ```

2.  Install dependencies.

    ```bash
    source env/bin/activate
    pip install -r requirements.txt
    ```

## Usage

1.  Run experiment.

    ```bash
    ./main.py
    ```

2.  Render report.

    ```bash
    ./analyze.py

    cd latex
    pdflatex --shell-escape report.tex \
        && bibtex report.aux \
        && pdflatex --shell-escape report.tex \
        && pdflatex --shell-escape report.tex
    ```
