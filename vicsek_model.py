import os
import math
import pprint
import numpy as np
import random as nr
import csv
from optparse import OptionParser

import matplotlib
import matplotlib.pyplot as plt

import multiprocessing
from joblib import Parallel, delayed

parser = OptionParser()
parser.add_option("-j", "--job", dest="jid", action = "store", type = "string", help = "run job", metavar = "JID")

(options, args) = parser.parse_args()
if options.jid:
    job_id = options.jid
else:
    print("Error: Job ID not specified")
    exit()

config_file = os.getcwd() + os.sep + "job_files" + os.sep + "JobID_" + job_id + ".txt"
print "Running Job: " + config_file

# simulation params
initN = None
timePoints = None

# model params
r = None
attraction = None

# seed rng
np.random.seed(1337)

periodicBounds = True

# cell division params
cellDiv = True
cellDivRate = None
divisionTimeMu = 15
divisionTimeStd = 3

with open(config_file) as currJob:

    for line in currJob:
    
        val = line.split(": ")
        if val[0] in ("# cells"):
            initN = int(val[len(val)-1])
        elif val[0] in ("# timepoints"):
            timePoints = int(val[len(val)-1])
        elif val[0] in ("r"):
            r = int(val[len(val)-1])
        elif val[0] in ("attraction"):
            attraction = int(val[len(val)-1])
        elif val[0] in ("Rate of cell division"):
            cellDivRate = int(val[len(val)-1])
        elif val[0] in ("Periodic Boundary Conditions"):
            if val[len(val)-1] == "True\n":
                periodicBounds = True
            else:
                periodicBounds = False
        else:
            if val[len(val)-1] in "True\n":
                cellDiv = True
            else:
                cellDiv = False

jobname = "JobID_" + job_id

print jobname
print (initN, timePoints, r, attraction, cellDivRate, cellDiv, periodicBounds)

if os.path.isdir(os.getcwd() + os.sep + jobname + '_output'):
    print "ERROR: Output directory already exists"
    exit()
else:
    os.makedirs(os.getcwd() + os.sep + jobname + '_output')

def findAngle(events, cid):
    return pos[events][cid]['a']

def getColor(events, cid):
    return pos[events][cid]['c']

def averageTheta(event, cell):
    global N
    global pos
    theta = pos[event][cell]['a']
    count = 1.0
    for cid in range(N):
        if(pos[event][cell]['x'] + attraction > 1000):
            if((pos[event][cid]['x'] < (-1000+(attraction-(1000-pos[event][cell]['x'])))) and 
               (abs(pos[event][cid]['y'] - pos[event][cell]['y']) <= attraction)):
                thetha=theta+pos[event][cid]['a']
                count+=count+1.0
        if(pos[event][cell]['x'] - attraction < -1000):
            if((pos[event][cid]['x'] > (1000-(attraction+(-1000-pos[event][cell]['x'])))) and 
               (abs(pos[event][cid]['y'] - pos[event][cell]['y']) <= attraction)):
                theta=theta+pos[event][cid]['a']
                count=count+1.0
        if(pos[event][cell]['y'] + attraction > 1000):
            if((pos[event][cid]['y'] < (-1000+(attraction-(1000-pos[event][cell]['y'])))) and 
               (abs(pos[event][cid]['x'] - pos[event][cell]['x']) <= attraction)):
                thetha=theta+pos[event][cid]['a']
                count=count+1.0
        if(pos[event][cell]['y'] - attraction < -1000):
            if((pos[event][cid]['y']>(1000-(attraction+(-1000-pos[event][cell]['y'])))) and 
               (abs(pos[event][cid]['x'] - pos[event][cell]['x']) <= attraction)):
                theta=theta+pos[event][cid]['a']
                count=count+1.0
        if((abs(pos[event][cid]['x'] - pos[event][cell]['x']) <= attraction) and 
           (abs(pos[event][cid]['y'] - pos[event][cell]['y']) <= attraction)):
            theta=theta + pos[event][cid]['a']
            count=count+1.0
    return (theta/count)
            
def getXPosition(cid, t):
    x_data = list()
    if cid in pos[t].keys():
        x_data.append(pos[t][cid]['x'])
    return x_data

def getYPosition(cid, t):
    y_data = list()
    if cid in pos[t].keys():
        y_data.append(pos[t][cid]['y'])
    return y_data

def getName(num):
    numZeros = 5 - len(repr(num))
    name = ""
    for i in range(numZeros):
        name += '0'
    name += repr(num)
    name += ".png"
    return name

def areInteracting(cell, event):
    global pos
    frame_N = np.max(pos[event].keys())
    for cid in range(frame_N):
        if ((abs(pos[event][cid]['x'] - pos[event][cell]['x']) <= attraction) and 
            (abs(pos[event][cid]['y'] - pos[event][cell]['y']) <= attraction)):
                plt.plot([pos[event][cid]['x'], pos[event][cell]['x']], 
                         [pos[event][cid]['y'], pos[event][cell]['y']], 
                         '-', linewidth=0.5, color='darkgray')
            
def showTrail(cell, event):
    if (event <= 10):
        for i in range(event):
            plt.plot([pos[i][cell]['x'], pos[i+1][cell]['x']], [pos[i][cell]['y'], pos[i+1][cell]['y']], 
                     'k--', linewidth=0.5)
    else:
        for i in range(10):
            plt.plot([pos[event-i-1][cell]['x'], pos[event-i][cell]['x']], 
                     [pos[event-i-1][cell]['y'], pos[event-i][cell]['y']], 
                     'g--', linewidth=0.5)

def setInitialCircle():
    x = list()
    y = list()
    for n in range(N):
        angle = nr.uniform(0,1)*(math.pi*2)
        x.append(800*math.cos(angle));
        y.append(800*math.sin(angle));
    
    for i in range(N):
        firstPos[i] = dict()
        for j in ['x', 'y', 'a', 'c']:
            if (j == 'c'):
                firstPos[i][j] = 'b'
            if (j == 'x'):
                firstPos[i][j] = x[i];
            if (j == 'y'):
                firstPos[i][j] = y[i];
            if (j == 'a'):
                if(firstPos[i]['x'] > 0):
                    firstPos[i][j] = math.pi+math.atan((firstPos[i]['y'])/(firstPos[i]['x']))
                else:
                    firstPos[i][j] = math.atan((firstPos[i]['y'])/(firstPos[i]['x']))
                        
def setOrbitingCircle():
    angle = 0
    delta_theta = (2*math.pi)/N
    pos = list()
    for n in range(N):
        pos.append([800*math.cos(angle), 800*math.sin(angle)]);
        angle = angle+delta_theta
    
    for i in range(N-1):
        firstPos[i] = dict()
        for j in ['x', 'y', 'a', 'c']:
            if (j == 'c'):
                firstPos[i][j] = 'b'
            if (j == 'x'):
                firstPos[i][j] = pos[i][0];
            if (j == 'y'):
                firstPos[i][j] = pos[i][1];
            if (j == 'a'):
                if(firstPos[i]['y'] > 0 and firstPos[i]['x'] > 0):
                    firstPos[i][j] = math.atan((pos[i+1][1]-pos[i][1])/(pos[i+1][0]-pos[i][0]))
                elif(firstPos[i]['y'] > 0 and firstPos[i]['x'] < 0):
                    firstPos[i][j] = (math.pi/2)+math.atan((pos[i+1][0]-pos[i][0])/(pos[i+1][1]-pos[i][1]))
                elif(firstPos[i]['y'] < 0 and firstPos[i]['x'] > 0):
                    firstPos[i][j] = math.atan((pos[i+1][0]-pos[i][0])/(pos[i+1][1]-pos[i][1]))-(math.pi)
                else:
                    firstPos[i][j] = math.pi+math.atan((pos[i+1][0]-pos[i][0])/(pos[i+1][1]-pos[i][1]))

def setDiagonal():
    x = np.linspace(-800, 800)
    y = np.linspace(-800, 800)
    for i in range(N):
        firstPos[i] = dict()
        for j in ['x', 'y', 'a', 'c']:
            if (j == 'c'):
                firstPos[i][j] = 'b'
            if (j == 'x'):
                firstPos[i][j] = x[i];
            if (j == 'y'):
                firstPos[i][j] = y[i];
            if (j == 'a'):
                if(firstPos[i]['x'] > 0):
                    firstPos[i][j] = math.pi+math.atan((firstPos[i]['y'])/(firstPos[i]['x']))
                else:
                    firstPos[i][j] = math.atan((firstPos[i]['y'])/(firstPos[i]['x']))

def setGrid():
    halfN = N/2
    x = np.linspace(-800, 800, num=halfN)
    y = np.linspace(-800, 800, num=halfN)
    c = 0
    for row in range(y.size):
        for col in range(x.size):
            firstPos[c] = dict()
            firstPos[c]['x'] = x[col];
            firstPos[c]['y'] = y[row];
            if(firstPos[c]['x'] >= 0):
                firstPos[c]['a'] = math.pi+math.atan((firstPos[c]['y'])/(firstPos[c]['x']))
            else:
                firstPos[c]['a'] = math.atan((firstPos[c]['y'])/(firstPos[c]['x']))
            firstPos[c]['t'] = np.floor(np.random.normal(divisionTimeMu, divisionTimeStd))
            c = c + 1

    with open('Time-Step Data.csv', mode='w') as csv_file:
        fieldnames = ['Time-Step', 'CellID', 'XPosition', 'YPosition', 'Angle', 'CellDivisionTime']
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)

        writer.writeheader()
        for i in range(c):
            writer.writerow({'Time-Step': 0, 'CellID': c, 'XPosition': firstPos[c]['x'], 'YPosition': firstPos[c]['y'], 'Angle': firstPos[c]['a'], 'CellDivisionTime': firstPos[c]['t']})

            
def cellDivide(event, cell):
    global nextPos
    global pos
    global N
    if event == 0:
        return((False, pos[event][cell]['t']-1))
    elif pos[event][cell]['t'] < 0:
        # return updated timer for parent cell
        new_timer_val = np.floor(np.random.normal(divisionTimeMu, divisionTimeStd))
        return((True, new_timer_val))
    elif event > 0:
        return((False, pos[event][cell]['t']-1))
        
        
pp = pprint.PrettyPrinter(indent=4)
N = initN

pos = list()
numCells = list()

firstPos = dict()
pos.append(firstPos)

setGrid();
#setInitialCircle();
#setDiagonal();
#setOrbitingCircle();

#print("\nSTATE (ITR = INITIAL) :")
#pp.pprint(pos)

for events in range(timePoints):
    
    nextPos = dict()
    divided_cells = list()

    for i in range(N):
        nextPos[i] = dict()
        for j in ['c', 'x', 'y', 'a', 't']:
            if(j=='x'):
                if(pos[events][i][j] > 1000):
                    pos[events][i][j] = -1000
                elif(pos[events][i][j] < -1000):
                    pos[events][i][j] = 1000
                nextPos[i][j] = pos[events][i][j]+(math.cos(pos[events][i]['a'])*r) 
            elif(j=='y'):
                if(pos[events][i][j] > 1000):
                    pos[events][i][j] = -1000
                elif(pos[events][i][j] < -1000):
                    pos[events][i][j] = 1000
                nextPos[i][j] = pos[events][i][j]+(math.sin(pos[events][i]['a'])*r)
            elif (j=='c'):
                nextPos[i][j] = 'b'
            elif (j=='a'):
                nextPos[i][j] = averageTheta(events, i)
            else:
                if cellDiv is True:
                    (did_divide, nextPos[i][j]) = cellDivide(events, i)
                    if did_divide == True:
                        divided_cells.append(i)
    
    if cellDiv is True:
        # cell division
        if len(divided_cells) > 0:
            for cid in divided_cells:
                nextPos[N] = dict({'c': pos[events][cid]['c'], 
                                    'x' : pos[events][cid]['x'] + np.random.normal(2,1), 
                                    'y' : pos[events][cid]['y'] + np.random.normal(2,1), 
                                    'a' : pos[events][cid]['a'] + math.pi, 
                                    't' : np.floor(np.random.normal(divisionTimeMu, divisionTimeStd))})
                N = N + 1
        
        numCells.append(N)
    with open('Time-Step Data.csv', mode='w') as csv_file:
        fieldnames = ['Time-Step', 'CellID', 'XPosition', 'YPosition', 'Angle', 'CellDivisionTime']
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        for i in range(N):
            writer.writerow({'Time-Step': events, 'CellID': i, 'XPosition': nextPos[i]['x'], 'YPosition': nextPos[i]['y'], 'Angle': nextPos[i]['a'], 'CellDivisionTime': nextPos[i]['t']})
    pos.append(nextPos)
    
#print("\nSTATE (ITR = " + repr(events) + ") :")
#pp.pprint(pos)

if cellDiv is True:
    plt.switch_backend('Agg')
    plt.figure(figsize=(3,2), dpi=200)
    plt.plot(range(timePoints), numCells)
    plt.xlabel("Time", size=7)
    plt.ylabel("Number of Cells", size=7)
    plt.savefig(jobname + '_output' + os.sep + "CellDivision.png")

tpoints = range(timePoints) 

def createPlot(frame):
    plt.switch_backend('Agg')
    matplotlib.rc('xtick', labelsize=4) 
    matplotlib.rc('ytick', labelsize=4) 
    matplotlib.rcParams.update({'font.size':6})
    
    plt.figure(figsize=(4,3), dpi=400)
    plt.xlim([-1000,1000])
    plt.ylim([-1000,1000])
    
    numcellsinframe = np.max(pos[frame].keys()) + 1
    
    for cid in range(numcellsinframe):
        xloc = getXPosition(cid, frame)[0]
        yloc = getYPosition(cid, frame)[0]
        theta_val = findAngle(frame, cid)
        plt.plot(xloc, yloc, marker='o', color='#2222ee', alpha=0.7, linestyle='None', markersize=4)
        areInteracting(cid, frame)
        #showTrail(cid, frame)
        #plt.quiver(xloc, yloc, 0.5*math.cos(theta_val), 0.5*math.sin(theta_val), angles='xy', 
        #           width=0.003, color='black')
        plt.plot([xloc, xloc+50*math.cos(theta_val)], [yloc, yloc+50*math.sin(theta_val)], 'k-', linewidth=0.5)
        
    plt.tight_layout()
    plt.savefig(jobname + '_output' + os.sep + getName(frame))
    plt.close()
    
    return(getName(frame))
    
num_cores = multiprocessing.cpu_count()
print(num_cores)
results = Parallel(n_jobs=num_cores)(delayed(createPlot)(frame) for frame in tpoints)

