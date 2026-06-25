
Lid driven cavity on a colocated grid solved with the pressure weighted interpolation method (PWIM) and the SIMPLE alogorithm.

Derivations and details here: https://app.notion.com/p/Lid-Driven-Cavity-with-SIMPLE-Algorithm-and-PWIM-Interpolation-346dcdf2638480c59515e26a845dc4dc

lid4a.py: explicit relaxation, dense matrix: not working yet

lid4b.py: implicit relaxation, dense matrix
comparison plots with Ghia (1982) for Re 100, Re 100 tested with 32x32 mesh.

lid4c.py: implicit relaxation, sparse matrix, comparison plots with Ghia (1982) for Re 100, 1000.
Re tested upto 2000 with 64x64 mesh. 

lidGS.py: implicit relaxation, sparse matrix, gauss-siedel solvers, plot with Ghia (1982) for Re 100.
Re 100 tested on a 32x32 mesh.  Slower after that.



<!-- lid4c1.py: implicit relaxation, sparse matrix, plot with Ghia (1982) for Re 100, 1000.
Tested with alternate numerical solvers to speed up the simulation

lid4d.py:   -->

