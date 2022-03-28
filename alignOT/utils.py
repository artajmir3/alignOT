import matplotlib.pyplot as plt
import math
import numpy as np
import ot
from matplotlib import collections  as mc
from mpl_toolkits.mplot3d import Axes3D
import random
import time
import mrcfile
import pandas as pd
def doplt(arr): return(plt.imshow(arr,cmap='gray'))
import trn, coords, gauss_forward_model
import importlib


class Term:
    def __init__(self, coef=1, exps=[0,0,0,0]):
        self.coef = coef
        self.exps = exps
        while len(self.exps) < 8:
            self.exps.append(0)
    
    def derivative(self, index):
        new_exps = [0,0,0,0,0,0,0,0]
        for i in range(8):
            if i != index:
                new_exps[i] = self.exps[i]
            else:
                new_exps[i] = self.exps[i] - 1
        return Term(coef=self.coef*self.exps[index], exps=new_exps)
    
    def evaluate(self, vals):
        s = self.coef
        if s == 0:
            return 0
        for i in range(min(len(vals), len(self.exps))):
            if self.exps[i] == 0:
                continue
            elif self.exps[i] == 1:
                s *= vals[i]
            elif self.exps[i] == 2:
                s *= vals[i] * vals[i]
            elif self.exps[i] == 3:
                s *= vals[i] * vals[i] * vals[i]
            else:
                print(self.exps[i])
                s *= vals[i] ** self.exps[i]
        return s
    
    def bunch_evaluate(self, bunch_vals):
        s = self.coef
        for i in range(min(len(bunch_vals), len(self.exps))):
            for j in range(self.exps[i]):
                s = np.multiply(s, bunch_vals[i])
        return s
    
    def __mul__(self, other):
        coef = self.coef * other.coef
        exps = []
        for i in range(8):
            exps.append(self.exps[i] + other.exps[i])
        return Term(coef=coef, exps=exps)
    
    def __neg__(self):
        return Term(coef=-self.coef, exps=self.exps)
    
    def __str__(self):
        s = ""
        for i in range(8):
            if self.exps[i] != 0:
                s += "q" + str(i) + "^" + str(self.exps[i])
        return str(self.coef) + "*" + s
    
    def is_similar(self, other):
        for i in range(8):
            if self.exps[i] != other.exps[i]:
                return False
        return True
    
    def simplify(self, vals):
        new_exps = [0,0,0,0,0,0,0,0]
        coef = self.coef
        for i in range(8):
            if vals[i] != 'x':
                coef *= vals[i]**self.exps[i]
            else:
                new_exps[i] = self.exps[i]
        return Term(coef=self.coef*self.exps[index], exps=new_exps)
        
    
class Polynomial:
    def __init__(self, terms=None):
        if terms is None:
            terms = []
        self.terms = {}
        for term in terms:
            self.terms[tuple(term.exps)] = term
        
    def add_term(self, term):
        if term.coef == 0:
            return
        if tuple(term.exps) not in self.terms:
            self.terms[tuple(term.exps)] = term
        else:
            self.terms[tuple(term.exps)].coef += term.coef

    def get_terms(self):
        terms = []
        for exps in self.terms:
            terms.append(self.terms[exps])
        return terms
        
    def derivative(self, index):
        res = Polynomial()
        for term in self.get_terms():
            res.add_term(term.derivative(index))
        return res
    
    def simplify(self, vals):
        res = ()
        for term in self.get_terms():
            res.add_term(term.simplify(vals))
        return res
    
    def evaluate(self, vals):
        t = 0
        for term in self.get_terms():
            t += term.evaluate(vals)
        return t
    
    def bunch_evaluate(self, bunch_vals):
        t = 0
        for term in self.get_terms():
            t = np.add(t, term.bunch_evaluate(bunch_vals))
        return t
    
    def __add__(self, other):
        res = Polynomial()
        for term in self.get_terms():
            res.add_term(term)
        for term in other.get_terms():
            res.add_term(term)
        return res
    
    def __sub__(self, other):
        res = Polynomial()
        for term in self.get_terms():
            res.add_term(term)
        for term in other.get_terms():
            res.add_term(-term)
        return res
    
    def __neg__(self):
        res = Polynomial()
        for term in self.get_terms():
            res.add_term(-term)
        return res
    
    def __mul__(self, other):
        res = Polynomial()
        for term1 in self.get_terms():
            for term2 in other.get_terms():
                res.add_term(term1*term2)
        return res
    
    def __str__(self):
        s = ""
        terms = self.get_terms()
        for i in range(len(terms)):
            if i > 0:
                s += " + "
            s += str(terms[i])
        return s

class Quaternion:
    def __init__(self, real_pol, i_pol, j_pol, k_pol):
        self.real_pol = real_pol
        self.i_pol = i_pol
        self.j_pol = j_pol
        self.k_pol = k_pol
        
    def conjugate(self):
        return Quaternion(self.real_pol, -self.i_pol, -self.j_pol, -self.k_pol)
    
    def __mul__(self, other):
        return Quaternion(self.real_pol*other.real_pol - self.i_pol*other.i_pol - self.j_pol*other.j_pol - self.k_pol*other.k_pol,
                          self.real_pol*other.i_pol + self.i_pol*other.real_pol + self.j_pol*other.k_pol - self.k_pol*other.j_pol,
                          self.real_pol*other.j_pol + self.j_pol*other.real_pol + self.k_pol*other.i_pol - self.i_pol*other.k_pol,
                          self.real_pol*other.k_pol + self.k_pol*other.real_pol + self.i_pol*other.j_pol - self.j_pol*other.i_pol)
    
    def __str__(self):
        return str(self.real_pol) + " + (" + str(self.i_pol) + ")i + (" + str(self.j_pol) + ")j + (" + str(self.k_pol) + ")k"
    

q = Quaternion(Polynomial([Term(coef=1, exps=[1,0,0,0])]), Polynomial([Term(coef=1, exps=[0,1,0,0])]), 
               Polynomial([Term(coef=1, exps=[0,0,1,0])]), Polynomial([Term(coef=1, exps=[0,0,0,1])]))
qs = q.conjugate()
print(qs)
print(q)

p = Quaternion(Polynomial([Term(coef=0, exps=[0,0,0,0])]), Polynomial([Term(coef=1, exps=[0,0,0,0])]), 
               Polynomial([Term(coef=1, exps=[0,0,0,0])]), Polynomial([Term(coef=1, exps=[0,0,0,0])]))

b = q*p*qs
print(b)

def get_quaternion_vals(theta, ax, ay, az):
    """
    Compute the quaternion representation for a given rotation in angle-axis representation
    params:
        theta: the angle of the rotation in radians
        ax, ay, az: three floats in a way that *ax, ay, az) shows the 3d axis of the rotation

    retrun:
        q: is a list of length 4 that has values of the corresponding quaternion
    """
    
    n = math.sqrt(ax**2 + ay**2 + az**2)
    return [math.cos(theta/2), math.sin(theta/2)*ax/n, math.sin(theta/2)*ay/n, math.sin(theta/2)*az/n]

def convert_to_poly(vals):
    return Quaternion(Polynomial([Term(coef=vals[0], exps=[0,0,0,0])]), Polynomial([Term(coef=vals[1], exps=[0,0,0,0])]), 
                      Polynomial([Term(coef=vals[2], exps=[0,0,0,0])]), Polynomial([Term(coef=vals[3], exps=[0,0,0,0])]))
    

def perform(x, y, z, vals):
    """
    Apply a given rotation on a given point cloud and generate a new point cloud
    params:
        x, y, z: three lists with len(x)=len(y)=len(z) in a way that (x[i], y[i], z[i]) is the 3d coordinates of the i-th point
        vals: a list of length 4 that contains the values of the quaternion correponding the the ritation

    return:
        xr, yr, zr: three lists with len(xr)=len(yr)=len(zr) in a way that (xr[i], yr[i], zr[i]) is the 3d coordinates of the i-th point after the rotation
    """
    
    xr = []
    yr = []
    zr = []
    q = convert_to_poly(vals)
    qs = q.conjugate()
    p = Quaternion(Polynomial([Term(coef=1, exps=[1,0,0,0])]), Polynomial([Term(coef=1, exps=[0,1,0,0])]), 
                   Polynomial([Term(coef=1, exps=[0,0,1,0])]), Polynomial([Term(coef=1, exps=[0,0,0,1])]))
    t = time.time()
    b = q*p*qs
    t = time.time()
    
    bunch_vals = [np.zeros(len(x)), np.array(x), np.array(y), np.array(z)]
    xr = list(b.i_pol.bunch_evaluate(bunch_vals))
    yr = list(b.j_pol.bunch_evaluate(bunch_vals))
    zr = list(b.k_pol.bunch_evaluate(bunch_vals))
    
    return xr, yr, zr


def my_sinkhorn(a, b, M, reg, numItermax=5000, stopThr=1e-3, prev=None):

    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    M = np.asarray(M, dtype=np.float64)

    # init data
    dim_a = len(a)
    dim_b = len(b)
    
    if prev is None:
        u = np.ones(dim_a) / dim_a
        v = np.ones(dim_b) / dim_b
    else:
        u = prev[0]
        v = prev[1]
    
    # Next 3 lines equivalent to K= np.exp(-M/reg), but faster to compute
    K = np.empty(M.shape, dtype=M.dtype)
    np.divide(M, -reg, out=K)
    np.exp(K, out=K)

    tmp2 = np.empty(b.shape, dtype=M.dtype)

    Kp = (1 / a).reshape(-1, 1) * K
    cpt = 0
    err = 1
    while (err > stopThr and cpt < numItermax):
        uprev = u
        vprev = v

        KtransposeU = np.dot(K.T, u)
        v = np.divide(b, KtransposeU)
        u = 1. / np.dot(Kp, v)

        if (np.any(KtransposeU == 0)
                or np.any(np.isnan(u)) or np.any(np.isnan(v))
                or np.any(np.isinf(u)) or np.any(np.isinf(v))):
            # we have reached the machine precision
            # come back to previous solution and quit loop
            print('Warning: numerical errors at iteration', cpt)
            raise Exception('hi')
            u = uprev
            v = vprev
            reg *= 2
            K = np.empty(M.shape, dtype=M.dtype)
            np.divide(M, -reg, out=K)
            np.exp(K, out=K)
        if cpt % 10 == 0:
            # we can speed up the process by checking for the error only all
            # the 10th iterations
            # compute right marginal tmp2= (diag(u)Kdiag(v))^T1
            np.einsum('i,ij,j->j', u, K, v, out=tmp2)
            err = np.linalg.norm(tmp2 - b)  # violation of marginal
        cpt = cpt + 1
        
    return u.reshape((-1, 1)) * K * v.reshape((1, -1)), u, v

def compute_diff_mat(a,b):
    
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    c = a.reshape((1,-1))
    d = b.reshape((1,-1))
    cd = c*d.T
    c2 = np.repeat(c*c,len(a), axis=0)
    d2 = np.repeat(d*d,len(b), axis=0)
    return c2 + d2.T - 2 * cd


def compute_cost_mat(x,y,z,xr,yr,zr):
    return compute_diff_mat(x,xr) + compute_diff_mat(y,yr) + compute_diff_mat(z,zr)

def OT(x,y,z,xr,yr,zr, prev=None, reg=0.1, method='my_sinkh'):
    a = []
    b = []
    for i in range(len(x)):
        a.append(1/len(x))
        b.append(1/len(x))
    
    M = compute_cost_mat(x,y,z,xr,yr,zr)
            
    t = time.time()
          
    if method == 'emd':
        T = ot.emd(a, b, M)
    if method == 'my_sinkh':
        while True:
            try:
                T, u, v = my_sinkhorn(a, b, M, reg, prev=prev)
            except Exception as e:
                reg = reg + 0.1
                print('Reg is now ' + str(reg))
                print(str(e))
                continue
            break
    
    cost = np.sum(np.multiply(M, T))
    
    if method == 'emd':
        return T,cost
    if method == 'my_sinkh':
        return T,cost, u, v


    
def dot(a, b):
    s = 0
    for i in range(4):
        s += a[i] * b[i]
    return s

    
def SGD(x, y, z, xr, yr, zr, lr=0.005, max_iter=100, reg=0.1, num_samples=1, verbose=False):
    px = Quaternion(Polynomial([Term(coef=0, exps=[0,0,0,0,1,0,0,0])]), Polynomial([Term(coef=1, exps=[0,0,0,0,0,1,0,0])]), 
                Polynomial([Term(coef=1, exps=[0,0,0,0,0,0,1,0])]), Polynomial([Term(coef=1, exps=[0,0,0,0,0,0,0,1])]))

    bx = q*px*qs
    i_der = []
    j_der = []
    k_der = []
    for i in range(4):
        i_der.append(bx.i_pol.derivative(i))
        j_der.append(bx.j_pol.derivative(i))
        k_der.append(bx.k_pol.derivative(i))
    
    vals = get_quaternion_vals(0, 0, 0, 1)
    quaternions = []
    costs = []
    OT_time = 0
    grad_time = 0
    sample_time = 0
    rotate_time = 0
    prev = None
    u = None
    v = None
    for i in range(max_iter):
        t = time.time()
        if verbose:
            if i % 10 == 9:
                print('Iteration number %d, the wasserstein deistance is %.2f'%(i, costs[-1]))
        quaternions.append([])
        for j in range(4):
            quaternions[-1].append(vals[j])
        
        xx,yy,zz = perform(x, y, z, vals)
        rotate_time += time.time() - t
        
        t = time.time()
        if u is not None:
            prev = (u,v)
        T,cost,u,v = OT(xr,yr,zr,xx,yy,zz,reg=reg,prev=prev)
        costs.append(cost)
        OT_time += time.time() -t
        t = time.time()
        
        for s in range(num_samples):
            sample_point = int(random.random()* len(x))
            
            

            x1 = x[sample_point]
            y1 = y[sample_point]
            z1 = z[sample_point]
            dest = None
            maxx = 0
            for j in range(len(x)):
                if T[sample_point][j] * len(x) > maxx:
                    maxx = T[sample_point][j] * len(x)
                    dest = j
            x2 = xr[dest]
            y2 = yr[dest]
            z2 = zr[dest]
            
            
            norm_grad = 0
            sample_time += time.time() - t
            t = time.time()

            grad = [0, 0, 0, 0]

            new_vals = []
            for k in range(4):
                new_vals.append(vals[k])
            new_vals.append(0)
            new_vals.append(x1)
            new_vals.append(y1)
            new_vals.append(z1)

            i_vals = bx.i_pol.evaluate(new_vals)
            i_der_vals = []
            j_vals = bx.j_pol.evaluate(new_vals)
            j_der_vals = []
            k_vals = bx.k_pol.evaluate(new_vals)
            k_der_vals = []
            for k in range(4):
                i_der_vals.append(i_der[k].evaluate(new_vals))
                j_der_vals.append(j_der[k].evaluate(new_vals))
                k_der_vals.append(k_der[k].evaluate(new_vals))
            

            for j in range(len(x)):
                if j != dest:
                    continue
                x2 = xr[j]
                y2 = yr[j]
                z2 = zr[j]

                temp = [0, 0, 0, 0]
                for k in range(4):

                    temp[k] += 2 * i_der_vals[k] * (i_vals - x2)
                    temp[k] += 2 * j_der_vals[k] * (j_vals - y2)
                    temp[k] += 2 * k_der_vals[k] * (k_vals - z2)

                    grad[k] += temp[k] * T[sample_point][j] * len(x) /num_samples
                    
        d_prod = dot(grad, vals)
        for j in range(4):
            grad[j] -= d_prod * vals[j]
        
        for j in range(4):
            vals[j] -= lr * grad[j]
        
        norm = math.sqrt(vals[0]**2 + vals[1]**2 + vals[2]**2 + vals[3]**2)
        for j in range(4):
            vals[j] /= norm
        
        norm_grad = math.sqrt(grad[0]**2 + grad[1]**2 + grad[2]**2 + grad[3]**2)
        grad_time += time.time() -t

    if verbose:
        print('Time spent for optimal transport is ' + str(OT_time) + ' second(s).')
        print('Time spent for computing gradient is ' + str(grad_time) + ' second(s).')
        print('Time spent for rotating is ' + str(rotate_time) + ' second(s).')
        print('Time spent for sampling is ' + str(sample_time) + ' second(s).')
        print('Final cost: ' + str(costs[-1]))
    return quaternions, costs
            
def sample(fname, thresh, M, invalid=False):
    """
    Sample a given file using a topology representing network and return sampled points
    params:
        fname: the name and address of the mrc file for the input map
        thresh: the thresholding parameter, to be more robust to noise the values in the map with intensity < thresh
                    will be changed to 0
        M: number of point you want to sample

    return:
        x, y, z: the coordinated of the sampled points
        x, y, z are lists so we have len(x)=len(y)=len(z)=M and
        (x[i], y[i], z[i]) shows the 3d coordinates of the i-th point
    """
    
    if invalid:
        with mrcfile.open(fname, mode='r+', permissive=True) as mrc:     
            mrc.header.map = mrcfile.constants.MAP_ID
            mrc.update_header_from_data()
    map_mrc = mrcfile.open(fname)
    map_original = map_mrc.data
    N = map_original.shape[0]
    print(N)
    psize_original = map_mrc.voxel_size.item(0)[0]
    psize = psize_original

    map_th = map_original.copy()
    map_th[map_th < thresh] = 0

    rm0,arr_flat,arr_idx,xyz,coords_1d = trn.trn_rm0(map_th,M,random_seed=None)

    l0 = 0.005*M # larger tightens things up (far apart areas too much to much, pulls together). smaller spreads things out
    lf = 0.5
    tf = M*8
    e0 = 0.3
    ef = 0.05

    rms,rs,ts_save = trn.trn_iterate(rm0,arr_flat,arr_idx,xyz,n_save=10,e0=e0,ef=ef,l0=l0,lf=lf,tf=tf,do_log=True,log_n=10)

    x_res = []
    y_res = []
    z_res = []
    for p in rms[10]:
        x_res.append(p[0])
        y_res.append(p[1])
        z_res.append(p[2])
    return x_res,y_res,z_res

def diff_quaternions(q1, q2):
    diff = (convert_to_poly(q1) * convert_to_poly(q2).conjugate()).real_pol.evaluate((0,0,0,0))
    if diff > 1:
        diff = 1
    elif diff < -1:
        diff = -1
    deg = math.acos(diff) * 2 * 180 / math.pi
    if deg > 180:
        return 360 - deg
    else:
        return deg


