# 3D Cryo-EM Density Map Alignment
alignOT is a method for solving rigid body alignment of cryo-EM density maps.

## 1. Get Started
For getting started you should clone the project and install some prerequisite python packages.
```
git clone git@github.com:artajmir3/alignOT.git
pip install -r requirements.txt
```

## 2. Run alignOT
To run alignOT on your desired data/maps, you can use python code and import our files or use command line. The usage of python code is shown in the `examples/basic_tour.ipynb` file. For runing the code run it from command line and specify the parameters here is an example:
```
python main.py --src Data/emd_1717.map --dest Data/emd_1717.map --max_iter 500 --threshold 72 --num_points 500 --reg 10 --lr 0.000005 --theta 20 --ax 0 --ay 0 --az 1
```
