# mw-fdtd

Moving window FDTD implementation (work in progress). Window can follow arbitrary paths through a geometry.

## Installation

Requires Python 3.11. In the top level directory, install the python dependencies with,

```bash
pip install -r requirements.txt
```

## Usage

```python
import mw_fdtd
```

This script demonstrates a simple moving window example and shows a side by side comparison with a non-moving grid,

```bash
python scripts/mw_comparison.py
```