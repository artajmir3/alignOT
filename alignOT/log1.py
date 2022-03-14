from utils import *
import json
import numpy as np
import argparse

def get_baseline(fname1, fname2, thresh, q1, q2, num = 20*89):
    costs = []
    for i in range(num):
        x, y, z = sample(fname1, thresh, 500)
        xr, yr, zr = perform(x, y, z, q1)
        x, y, z = sample(fname2, thresh, 500)
        x, y, z = perform(x, y, z, q2)
        T, c = OT(x, y, z, xr, yr, zr, method='emd')
        costs.append(c)
    return costs


parser = argparse.ArgumentParser(description='LOG GENERATOR')

parser.add_argument('--id', type = int, help='Experiment id')

args = parser.parse_args( )

fname1 = {
    1:'Data/MMAlign/1/1.mrc',
    2:'Data/MMAlign/2/1.mrc',
    3:'Data/MMAlign/3/3jbr.mrc',
    4:'Data/MMAlign/4/1.mrc',
    5:'Data/MMAlign/5/1.mrc',
    6:'Data/MMAlign/6/1.mrc',
    }

fname2 = {
    1:'Data/MMAlign/1/2.mrc',
    2:'Data/MMAlign/2/2.mrc',
    3:'Data/MMAlign/3/5gjw.mrc',
    4:'Data/MMAlign/4/2.mrc',
    5:'Data/MMAlign/5/2.mrc',
    6:'Data/MMAlign/6/2.mrc',
    }

threshold = {1:0.1, 2:0.22, 3:0.65, 4:0.1, 5:0.1, 6:0.43}

base = {
    args.id:get_baseline(fname1[args.id], fname2[args.id], threshold[args.id], get_quaternion_vals(0, 1, 1, 1), get_quaternion_vals(0, 1, 1, 1))
#    1:get_baseline('Data/MMAlign/1/1.mrc', 'Data/MMAlign/1/2.mrc', 0.1, get_quaternion_vals(0, 1, 1, 1), get_quaternion_vals(0, 1, 1, 1)),
#    2:get_baseline('Data/MMAlign/2/1.mrc', 'Data/MMAlign/2/2.mrc', 0.22, get_quaternion_vals(0, 1, 1, 1), get_quaternion_vals(0, 1, 1, 1)),
#    3:get_baseline('Data/MMAlign/3/3jbr.mrc', 'Data/MMAlign/3/5gjw.mrc', 0.65, get_quaternion_vals(0, 1, 1, 1), get_quaternion_vals(0, 1, 1, 1)),
#    4:get_baseline('Data/MMAlign/4/1.mrc', 'Data/MMAlign/4/2.mrc', 0.1, get_quaternion_vals(0, 1, 1, 1), get_quaternion_vals(0, 1, 1, 1)),
#    5:get_baseline('Data/MMAlign/5/1.mrc', 'Data/MMAlign/5/2.mrc', 0.1, get_quaternion_vals(0, 1, 1, 1), get_quaternion_vals(0, 1, 1, 1)),
#    6:get_baseline('Data/MMAlign/6/1.mrc', 'Data/MMAlign/6/2.mrc', 0.43, get_quaternion_vals(0, 1, 1, 1), get_quaternion_vals(0, 1, 1, 1))
}

f = open('RPE_github/RPE/q.txt', 'r')
best_qs = json.load(f)
f.close()


data = np.genfromtxt('Data/sphere_grid_icos2_f4.xyz')
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
    if zz[i] >= 0: #xx[i] >= 0 and yy[i] <= 0 and
        xxx.append(xx[i])
        yyy.append(yy[i])
        zzz.append(zz[i])


complete_log = {'base': base}
fl = open('/scratch/pr-kdd-1/AryanTajmirRiahi/logs/log_%d.txt'%(args.id,), 'w')
for exp_id in [args.id]:#[1]: #[1,2,3,4,5,6]:
    for num in [250, 500, 1000]:
        for theta in [45, 60, 90]:
            means = []
            d = []
            for i in range(89):
                for j in range(20):
                    q_base = get_quaternion_vals(float(theta) * math.pi /180, xxx[i][0], yyy[i][0], zzz[i][0])
                    q = best_qs["%d,%d,%d,%d,%d"%(exp_id, num, theta, i, j)]
                    d += get_baseline(fname1[exp_id], fname2[exp_id], threshold[exp_id], q_base, q, num=1)
                #    break
                #break

            complete_log["%d,%d,%d"%(exp_id, num, theta)] = d
            fl.write('| %d\t| %d\t| %d\t| %.2f+%.2f\t| %.2f+%.2f\t|\n'%(exp_id, num, theta, np.mean(base[exp_id]), np.std(base[exp_id]), np.mean(d), np.std(d)))
    #        break
    #    break
    #break
fl.close()

cl = open('/scratch/pr-kdd-1/AryanTajmirRiahi/logs/complete_log_%d.txt'%(args.id,), 'w')
cl.write(json.dumps(complete_log))
cl.close()
