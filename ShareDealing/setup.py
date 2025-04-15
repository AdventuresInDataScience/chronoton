from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy as np

setup(
    ext_modules=cythonize([
        Extension(
            "backtest_core",  # Name of the extension module
            ["_execute_backtest_loop_ShareDealing.py"],  # Cython source file
            include_dirs=[np.get_include()]  # Include NumPy headers
        )
    ]),
)


'''
2. Install required packages
Make sure you have the necessary packages installed:

```bash
pip install cython numpy
```

3. Compile the extension
Open a command prompt/terminal, navigate to your project directory, and run:
```bash
cd c:\Users\malha\Documents\Projects\chronoton\ShareDealing
python setup.py build_ext --inplace
```

This will generate a compiled extension (.pyd file on Windows or .so file on Unix-like systems) in your current directory.

4. Fix file extension
Note that your Cython file should have a .pyx extension instead of .py. Rename it:
```bash
ren _execute_backtest_loop_ShareDealing.py _execute_backtest_loop_ShareDealing.pyx
```

Then update your setup.py file to use the .pyx file and run the build command again.

5. Import in your main code
Now you can import the compiled module in your main Python code:

```python
# In SingleAssetSingleTimeFrame.py
from backtest_core import execute_backtest_loop

# When you need to call it:
equity_curve, trades = execute_backtest_loop(
    data, buy_signals, sell_signals, short_signals, cover_signals, 
    commission, commission_pct
)
```

Common issues and solutions
ImportError: If you get import errors, make sure the compiled .pyd/.so file is in your Python's import path (usually the current directory)
Compilation errors: Make sure you have a C compiler installed (Visual C++ on Windows, GCC on Linux/Mac)
Type errors: Ensure your NumPy arrays have the correct data types as specified in the Cython function signature
The build process should output a compiled module that you can import and use in your main Python code, which will run much faster than the equivalent Python implementation.
'''