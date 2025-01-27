This repository contains the source code for the numerical experiments considered
in [A Novel Deflation Approach for Topology Optimization and Application for Optimization of Bipolar Plates of Electrolysis Cells](https://doi.org/10.48550/arXiv.2406.17491) by Leon Baeck, Sebastian Blauth, Christian Leithäuser, René Pinnau and Kevin Sturm.

To run the code, you have to install [cashocs](https://cashocs.readthedocs.io/)
first, which includes all necessary prerequisites. The results presented in this
repository have been obtained with version 2.2.0-dev of cashocs (which uses FEniCS 2019.1) and numba version 0.55.2.

The repository consists of the following test cases:

- The five-holes double-pipe example (named `five_holes_double_pipe`) which is considered in Section 4 of the paper.

- The topology optimization of a bipolar plate model (named `bipolar_plate`) which is considered in Section 5 of the paper.

In each of the directories, there is a `main.py` file, which can be used to run the code. This file runs the entire benchmark, consisting of the application of the deflation approach for the respective problem, as presented in [A Novel Deflation Approach for Topology Optimization and Application for Optimization of Bipolar Plates of Electrolysis Cells](https://doi.org/10.48550/arXiv.2406.17491).

Further, each directory, contains a file `visualization.py`, which generates the plots used in the paper. The repository is already initialized with the solutions obtained for the numerical examples in the paper, so that this can be run directly.

This software is citeable under the following DOI: .

If you use the deflation approach for your work, please cite the paper

	A Novel Deflation Approach for Topology Optimization and Application for Optimization of Bipolar Plates of Electrolysis Cells
	Leon Baeck, Sebastian Blauth, Christian Leithäuser, René Pinnau and Kevin Sturm
	arXiv, 2024
	https://doi.org/10.48550/arXiv.2406.17491

If you are using BibTeX, you can use the following entry:

	@misc{Baeck2024Deflation,
	  author        = {Baeck, Leon and Blauth, Sebastian and Leith\"auser, Christian and Pinnau, Ren\'e and Sturm, Kevin},
	  title         = {A Novel Deflation Approach for Topology Optimization and Application for Optimization of Bipolar Plates of Electrolysis Cells},
	  year          = {2024},
	  doi           = {10.48550/arXiv.2406.17491},
      eprint        = {2406.17491},
	  archivePrefix = {arXiv},
	}

