[![Python testing](https://github.com/alexarntzen/numalg/workflows/Python%20testing/badge.svg)](https://github.com/alexarntzen/numalg/actions/workflows/python_test.yml)
[![Python linting](https://github.com/alexarntzen/numalg/workflows/Python%20linting/badge.svg)](https://github.com/alexarntzen/numalg/actions/workflows/python_lint.yml)
[![Latex compiled](https://github.com/alexarntzen/numalg/workflows/Compile%20latex/badge.svg)](https://github.com/alexarntzen/numalg/actions/workflows/compile_latex.yml)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

Useful methods are placed in the `linalg/` directory.

The tests are placed in `test/` directory.  

The two jupyter notebooks making up the report for the project are placed in the `report` directory.
If you want to run the report you have to have the packages in `requirements.txt` installed. 
Also, the notebooks must be run from within the project, or `linalg` must be installed.

To run all the test run: 
```
python3 -m unittest discover
```
from in the top level directory. 
