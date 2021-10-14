from utils import *
import argparse

# Instantiate the parser
parser = argparse.ArgumentParser(description='3D Cryo-EM map alignment')


parser.add_argument( '--src', action = 'store', type = str, help = 'Identigy the source map' )
parser.add_argument( '--dest', action = 'store', type = str, help = 'Identigy the target map' )
parser.add_argument('--threshold', type = float, help='Thresholding parameter of the maps')
parser.add_argument('--num_points', type = int, help='Number of sampled points')
parser.add_argument('--max_iter', type = int, help='Maximum number of iterations')
parser.add_argument('--lr', type = float, help='Learning rate')
parser.add_argument('--reg', type = float, help='Regularization parameter')
parser.add_argument('--theta', type = float, help='Initial rotation angle')
parser.add_argument('--ax', type = float, help='Initial rotation axis x')
parser.add_argument('--ay', type = float, help='Initial rotation axis y')
parser.add_argument('--az', type = float, help='Initial rotation axis z')

args = parser.parse_args( )


t = time.time()
x, y, z = sample(args.src, args.threshold, args.num_points)
xr, yr, zr = perform(x, y, z, get_quaternion_vals(args.theta * math.pi /180, args.ax, args.ay, args.az))
x, y, z = sample(args.dest, args.threshold, args.num_points)
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

