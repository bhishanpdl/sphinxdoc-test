Homework 1
============

The source codes are given in directory "code".
The report is given in "docs/html/index.html".

To run the source code:

cd code
python univariate.py
python multivariate.py
python polyfit.py

Alternatively:
python univariate.py -i ../data/univariate
python multivariate.py -i ../data/multivariate
python polyfit.py -i ../data/multivariate

Alternatively:
python univariate.py --input_data_dir ../data/univariate
python multivariate.py --input_data_dir ../data/multivariate
python polyfit.py --input_data_dir ../data/multivariate


Outputs
========
The output images from source codes are generated inside code/images folder.
They are also used in the index.html documentation.

Caveats
========
The source codes requires python >=3.5 to compile.
For example the matrix multiply operator @ needs python >= 3.5.
