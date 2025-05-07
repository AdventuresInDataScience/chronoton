# chronoton
Fast Trading and Backtesting library in python/cython

chronoton/
│
├── src/
│   └── chronoton/
│       ├── __init__.py
│       ├── CFD/
│       │   ├── __init__.py
│       │   ├── SAST.py
│       │   ├── ABCD.py
│       │   └── _fast_cfd.pyx         # Cython file for CFD module
│       ├── ShareDealing/
│       │   ├── __init__.py
│       │   ├── MAMT.py
│       │   ├── XYZT.py
│       │   └── _fast_share.pyx       # Cython file for ShareDealing
│       ├── common/
│       │   ├── __init__.py
│       │   ├── utils.py              # shared Python logic
│       │   └── _cyutils.pyx          # shared Cython logic
│
├── tests/
│   ├── __init__.py
│   ├── test_SAST.py
│   ├── test_MAMT.py
│   └── test_common.py
│
├── pyproject.toml                   # or setup.py / setup.cfg
└── README.md



