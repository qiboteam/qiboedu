## Bell Inequalities

This folder contains a series of notebooks to explore the popular inequalities concerning quantum Bell states in the context of quantum computing.

The material is organized in three modules of increasing difficulty, and the relative implementation makes use of Qibo basic primitives to instantiate circuits and run them. For further details about how to work with Qibo, please refer to the other tutorials in this repository.

These notebooks have been demonstrated to work also on Google Colab. In that case, we recommend to insert a cell at the top of each notebook, where to execute `!pip install qibo` and incorporate the `plot_bell_inequalities()` function from `src/qiboedu/scripts/plotscripts.py` as opposed to import it, which would require mounting your GDrive.

All notebooks are part of the paper [Simulating Bell inequalities with Qibo](https://doi.org/10.1088/1361-6404/adcd13).
