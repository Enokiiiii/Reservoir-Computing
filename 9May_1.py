# Noise Added to Reservoir 2ndOrder batch new
import matplotlib.pyplot as plt
import numpy as np
import random
from scipy import signal
import tqdm
import math

#2nd order of reservoir
def secondorder(r, N):
    r2 = []
    for i in range(N):
        for j in range(i, N):
            r2.append(r[i]*r[j])
    return np.array(r2).reshape([-1, 1])

#MSE
def mse(T, Y, V):
    MSE = 0
    for i in range(T):
        MSE += (Y[i] - V[i])**2
    MSE /= T
    return MSE 

#SCR
def scr(T, Y, V):
    return np.count_nonzero(np.sign(Y) == np.sign(V)) / T

#Syncronization
def run(T, X, W, Win, N, Nin, e):
    r = np.zeros((N, 1))
    for i in tqdm.tqdm(range(T)):
        np.random.seed()
        noise = np.random.normal(0, 1, size=(N, 1))
        r = np.tanh(W @ r + Win @ X[i].T.reshape((Nin, 1)) + e*noise)
    return r

#Traning
def train(T, r, X, V, W, Win, N, Nin, e, l, sec):
    R = np.empty((N, 0))
    N2 = np.arange(N+1).sum()
    R2 = np.empty((N2, 0))
    for i in tqdm.tqdm(range(T)):
        np.random.seed()
        noise = np.random.normal(0, 1, size=(N, 1))
        r = np.tanh(W @ r + Win @ X[i].T.reshape((Nin, 1)) + e*noise)
        R = np.hstack((R, r))
        if sec == 1:
            r2 = secondorder(r, N)
            R2 = np.hstack((R2, r2))
    if sec == 1:
        R = np.vstack((R, R2)) 
        Wout  = np.linalg.solve(R @ R.T + T_train*l*np.eye(N+N2), R @ V.T).T
    elif sec == 0:
        Wout  = np.linalg.solve(R @ R.T + T_train*l*np.eye(N), R @ V.T).T
    Y = Wout @ R
    return Wout, Y, r

#Testing
def pre(T, r, X, W, Win, Wout, N, Nin, e, sec):
    R = np.empty((N, 0))
    N2 = np.arange(N+1).sum()
    R2 = np.empty((N2, 0))
    for i in tqdm.tqdm(range(T)):
        np.random.seed()
        noise = np.random.normal(0, 1, size=(N, 1))
        r = np.tanh(W @ r +  Win @ X[i].T.reshape((Nin, 1))  + e*noise)
        R = np.hstack((R, r))
        if sec == 1:
            r2 = secondorder(r, N)
            R2 = np.hstack((R2, r2))
    if sec == 1:
        R = np.vstack((R, R2)) 
    Y = Wout @ R
    return Y

#Setting of RC
N = 20
delay = 0
Nin = 12 + delay*12
sr = 0.95
b = 0.1
e = 0
l = 1e-4
sec = 1 #Normal RC: sec = 0, 2nd Order RC: 1

N_seed = 1

#Setting of data
T_run = 25000
T_train = 200000
T_pre =  100000
T_ahead = 250
h = 0.2

#Generating data
data = np.loadtxt('RBC_raw.dat')

X_run = data[delay:delay+T_run, 1:]
for i in range(delay):
    X_run = np.hstack((X_run, data[i:i+T_run, 1:]))

X_train = data[delay+T_run:delay+T_run+T_train, 1:]
for i in range(delay):
    X_train = np.hstack((X_train, data[i+T_run:i+T_run+T_train, 1:]))

V_train = data[delay+T_run+T_ahead:delay+T_run+T_train+T_ahead, 0]

X_pre = data[delay+T_run+T_train:delay+T_run+T_train+T_pre, 1:]
for i in range(delay):
    X_pre = np.hstack((X_pre, data[i+T_run+T_train:i+T_run+T_train+T_pre, 1:]))

V_pre = data[delay+T_run+T_train+T_ahead:delay+T_run+T_train+T_pre+T_ahead, 0]


#Main
log = np.empty((0, 2))
for seed in range(N_seed):
    #Set Wfb, W
    np.random.seed(seed)
    Win = np.random.uniform(-1, 1, (N, Nin)) 
    Win *= b 
    np.random.seed(seed)
    W = np.random.uniform(-1, 1, (N, N))
    eigv_list = np.linalg.eig(W)[0]
    sp_radius = np.max(np.abs(eigv_list))
    W *= sr / sp_radius
    #Syncronization
    r = run(T_run, X_run, W, Win, N, Nin, e)
    #Traning
    Wout, Y_train, r = train(T_train, r, X_train, V_train, W, Win, N, Nin, e, l, sec)
    #Testing
    Y_pre = pre(T_pre, r, X_pre, W, Win, Wout, N, Nin, e, sec)
    MSE = mse(T_pre, Y_pre, V_pre.T)
    SCR = scr(T_pre, Y_pre, V_pre.T)
    log = np.vstack((log, np.array([MSE, SCR]).reshape((1, 2))))
    print('Completed %d' % int(seed+1))
    print('MSE = %.3e, SCR = %.3f' % (MSE, SCR))
    np.savetxt("data.csv", log, delimiter=',')
    #if sec == 0:
     #   np.savetxt("N%ddelay%d.csv" % (N, delay), log, delimiter=',')
    #if sec == 1:
     #   np.savetxt("S%ddelay%d.csv" % (N, delay), log, delimiter=',')

#Time series visualization of L
fig = plt.figure(figsize=(8, 3))
fig.subplots_adjust(top=0.97, bottom=0.16, right=0.99, left=0.12)
ax = fig.add_subplot()

T_fig = 10000
time = np.arange(0, int(T_fig*h), h)

ax.plot(time, V_pre[:T_fig], label="Target")
ax.plot(time, Y_pre[:T_fig], label="Prediction")
ax.set_xlabel(r"$t$", size=15)
ax.set_ylabel(r"$L$", size=15)
ax.legend(loc=1)

plt.show()