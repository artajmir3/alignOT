from utils import *
import argparse


def experiment10(fname, thresh, num_points, rotation_q, lr, reg, max_iter, num_experiments, resampling=True):
    diffs = []
    times = []
    global x, y, z
    for i in range(num_experiments):
#         print(i)
        x, y, z = sample(fname, thresh, num_points)
        xr, yr, zr = perform(rotation_q)
        if resampling:
            x, y, z = sample(fname, thresh, num_points)
        t = time.time()
        quartenions, costs = SGD(x, y, z, xr, yr, zr, lr=lr, max_iter=max_iter, reg=reg, num_samples=1, verbose=False)
        times.append(time.time() - t)
        
        mini = None
        minx = costs[0]
        for i in range(len(costs)):
            if costs[i] <= minx:
                minx = costs[i]
                mini = i
        print(mini)
        print(diff_quaternions(quartenions[mini], rotation_q))
            
        diffs.append(diff_quaternions(quartenions[mini], rotation_q))
        fig, ax = plt.subplots()
        plt.plot(costs)
        plt.show()
    return diffs, times

experiment10('emd_2472.map', 98, 500, lr=0.00001, max_iter=500, reg=10, rotation_q=get_quaternion_vals(math.pi/7, 0, 0, 1),
           num_experiments=1)

# Instantiate the parser
parser = argparse.ArgumentParser(description='3D Cryo-EM map alignment experiment #10')


parser.add_argument( '--fname', action = 'store', type = str, help = 'Identify the map' )
parser.add_argument('--threshold', type = float, help='Thresholding parameter of the maps')
parser.add_argument('--num_points', type = int, help='Number of sampled points')
parser.add_argument('--max_iter', type = int, help='Maximum number of iterations')
parser.add_argument('--lr', type = float, help='Learning rate')
parser.add_argument('--regs', nargs='+', help='<Required> Set flag', required=True)
parser.add_argument('--theta', type = float, help='Initial rotation angle')
parser.add_argument('--ax', type = float, help='Initial rotation axis x')
parser.add_argument('--ay', type = float, help='Initial rotation axis y')
parser.add_argument('--az', type = float, help='Initial rotation axis z')

args = parser.parse_args( )

for reg in args.regs:
    x, y, z = sample(args.src, args.threshold, args.num_points)
    xr, yr, zr = perform(x, y, z, get_quaternion_vals(args.theta * math.pi /180, args.ax, args.ay, args.az))
    x, y, z = sample(args.dest, args.threshold, args.num_points)
    t = time.time()
    quartenions, costs = SGD(x, y, z, xr, yr, zr, lr=args.lr, max_iter=args.max_iter, reg=args.reg, num_samples=1)
    print('Total time consumption is ' + str(time.time() - t) + ' second(s).')

    print('Our best rotation quaternion is: ' + str(quartenions[-1]))
    print('The ground truth was: ' + str(get_quaternion_vals(args.theta * math.pi /180, args.ax, args.ay, args.az)))
    print('So our result is ' + str(diff_quaternions(quartenions[-1], get_quaternion_vals(args.theta * math.pi /180, args.ax, args.ay, args.az))) + ' degrees off.')


    fig, ax = plt.subplots()
    plt.plot(costs)
    plt.show()

x_fin, y_fin, z_fin = perform(x, y, z, quartenions[-1])


fig = plt.figure()
ax = fig.gca(projection='3d', adjustable='box')
ax.scatter(x, y, z, c='C0',  marker='o')
ax.scatter(xr, yr, zr, c='C1', marker='o')
ax.scatter(x_fin, y_fin, z_fin, c='C2',  marker='o')
plt.show()
