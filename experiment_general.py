from utils import *
import argparse
import json



def experiment_genral(source, dest, thresh, num_points, rotation_q, lr, reg, max_iter, num_experiments, resampling=True, output_path="results.json"):
    diffs = []
    times = []
    q_logs = {}
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
        q_logs.append(quartenions)
        
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

    f = open(output_file, 'w')
    f.write(json.dumps({
        'diffs': diffs,
        'times': times,
        'q_logs': q_logs
        }))

#experiment10('emd_2472.map', 98, 500, lr=0.00001, max_iter=500, reg=10, rotation_q=get_quaternion_vals(math.pi/7, 0, 0, 1),
#           num_experiments=1)

# Instantiate the parser
parser = argparse.ArgumentParser(description='3D Cryo-EM map alignment experiment #10')


parser.add_argument( '--src', action = 'store', type = str, help = 'Identify the source map' )
parser.add_argument( '--dest', action = 'store', type = str, help = 'Identify the dest map' )
parser.add_argument('--threshold', type = float, help='Thresholding parameter of the maps')
parser.add_argument('--num_points_s', nargs='+', help='<Required> Set flag', required=True)
parser.add_argument('--max_iter', type = int, help='Maximum number of iterations')
parser.add_argument('--lr', type = float, help='Learning rate')
parser.add_argument('--reg', type = float, help='Regularization parameter')
parser.add_argument('--thetas', nargs='+', help='<Required> Set flag', required=True)
parser.add_argument( '--sphere', action = 'store', type = str, help = 'Identify the sphere file' )
parser.add_argument( '--output', action = 'store', type = str, help = 'Identify the output file' )
#parser.add_argument('--ax', type = float, help='Initial rotation axis x')
#parser.add_argument('--ay', type = float, help='Initial rotation axis y')
#parser.add_argument('--az', type = float, help='Initial rotation axis z')

args = parser.parse_args( )

data = np.genfromtxt(args.sphere)
# theta, phi, r = np.hsplit(data, 3) 
# theta = theta * pi / 180.0
# phi = phi * pi / 180.0
# xx = sin(phi)*cos(theta)
# yy = sin(phi)*sin(theta)
# zz = cos(phi)
xx, yy, zz = np.hsplit(data, 3)

xxx = []
yyy = []
zzz = []
for i in range(len(xx)):
    if xx[i] >= 0 and yy[i] <= 0 and zz[i] >= 0:
        xxx.append(xx[i])
        yyy.append(yy[i])
        zzz.append(zz[i])

for num_points in args.num_points_s:
    for i in range(len(xxx)):
        for theta in args.thetas:
            #x, y, z = sample(args.src, args.threshold, num_points)
            #xr, yr, zr = perform(x, y, z, get_quaternion_vals(theta * math.pi /180, xxx[i], yyy[i], zzz[i]))
            #x, y, z = sample(args.dest, args.threshold, args.num_points)
            #t = time.time()
            quartenions, costs = experiment_genral(args.src, args.dest, num=num_points, thresh=args.threshold, rotation_q=get_quaternion_vals(theta * math.pi /180, xxx[i], yyy[i], zzz[i]),
                                                   lr=args.lr, max_iter=args.max_iter, reg=args.reg, num_experiments=20, output_path=args.output + '_num=%d_%dthaxis_theta=%d.json'%(num_points, i, theta))
            #print('Total time consumption is ' + str(time.time() - t) + ' second(s).')

            #print('Our best rotation quaternion is: ' + str(quartenions[-1]))
            #print('The ground truth was: ' + str(get_quaternion_vals(theta * math.pi /180, xxx[i], yyy[i], zzz[i])))
            #print('So our result is ' + str(diff_quaternions(quartenions[-1], get_quaternion_vals(theta * math.pi /180, xxx[i], yyy[i], zzz[i]))) + ' degrees off.')


            #fig, ax = plt.subplots()
            #plt.plot(costs)
            #plt.show()

x_fin, y_fin, z_fin = perform(x, y, z, quartenions[-1])


fig = plt.figure()
ax = fig.gca(projection='3d', adjustable='box')
ax.scatter(x, y, z, c='C0',  marker='o')
ax.scatter(xr, yr, zr, c='C1', marker='o')
ax.scatter(x_fin, y_fin, z_fin, c='C2',  marker='o')
plt.show()
