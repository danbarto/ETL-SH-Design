#!/usr/bin/env python3
import math

def fbk_w19_5fc(r):
    return 307.12*math.log(r) - 1468.9

def fbk_w15_5fc(r):
    return 497.95*math.log(r) - 2630.6


# FBK production like sensors
def fbk_w13_2p5fc(r):
    return 499.43*math.log(r)-2618.8

def fbk_w13_5fc(r):
    return 435.93*math.log(r) - 2251.2

def fbk_w13_10fc(r):
    return 363.91*math.log(r)-1834.6

def fbk_w13_15fc(r):
    return 333.15*math.log(r)-1664.5


# HPK production like sensors
def hpk_split4_2p5fc(r):
    return 578.03*math.log(r)-3103.6

def hpk_split4_5fc(r):
    return 526.14*math.log(r)-2792

def hpk_split4_10fc(r):
    return 481.91*math.log(r)-2544.6

def hpk_split4_15fc(r):
    return 446.32*math.log(r)-2338.6

def irradiation(r):
    '''
    r in mm, returns equivalent fluence in 1e14
    '''
    return  -9.053e+00 + 7.420e+03/r + 3.664e-03*r

def sensor_current(fluence, gain=20, alpha=3.70235e-19, pixels=256):
    '''
    fluence in 1e14 equivalent fluence
    returns current in mA
    '''
    volume = pixels * 0.13**2 * 0.005  # entire 16x16 sensor in cm**3
    return gain*fluence*1e14*volume*alpha*1e3

def sensor_occupancy(r):
    '''
    r in [mm]
    returns occupancy wrt to 1 at 320mm
    '''
    return 0.11 + 91297/r**2  # this is pretty conservative
