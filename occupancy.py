#!/usr/bin/env python3

'''
data rate = l1a_rate (750 khz) * (40 + 40 + 40*(occupancy in # of pixels / l1a))

: 320 Mbps Elinks  -> 3.39% Max Occupancy  (1 lpgbt long RB)
: 640 Mbps Elinks  -> 7.55% Max Occupancy  (2 lpgbt long RB) *baseline*
: 1280 Mbps Elinks -> 15.89% Max Occupancy (2 lpgbt short RB) *baseline*
: 2048 Mbps Elinks -> 25.89% Max Occupancy (4 lpgbt short RB)
:
: 3.390000% Avg Occupancy -> 320 Mbps    (1 lpgbt long RB)
: 7.550000% Avg Occupancy -> 639 Mbps    (2 lpgbt long RB) *baseline*
: 15.890000% Avg Occupancy -> 1280 Mbps  (2 lpgbt short RB) *baseline*
: 25.890000% Avg Occupancy -> 2048 Mbps  (4 lpgbt short RB)
'''
import numpy as np
import colorsys
from yahist import Hist1D

from ETL import *
from sensors import *

from yaml import load, dump
try:
    from yaml import CLoader as Loader, CDumper as Dumper
except ImportError:
    from yaml import Loader, Dumper

import matplotlib.pyplot as plt
import mplhep as hep
plt.style.use(hep.style.CMS)

def green_to_red(x):
    assert 0<=x<=1
    return colorsys.hsv_to_rgb((1 - x) / 3., 1.0, 1.0)

if __name__ == '__main__':

    r_outer = 1185
    modify_partitions = False

    with open('layouts/database_new.yaml') as f:
        database_new = load(f, Loader=Loader)

    disk_new = database_new['new']

    new_layout = {}

    s = Sensor(42.5, 22)
    m = Module(43.10, 56.50, n_sensor_x=1, n_sensor_y=2, sensor_distance_y=22.5, sensor_distance_x=42.5+0.1)
    m.populate(s)

    rb = ReadoutBoard(10,56.5, color='green')
    pb = ReadoutBoard(10,29.5, color='red')

    SM = SuperModule(m, pb, rb, n_modules=3, orientation='above')


    new_layout['D1'] = Dee(315, 1185)
    new_layout['D1'].fromCenters(disk_new['disk1']['front'], s)

    new_layout['D2'] = Dee(315, 1185)
    new_layout['D2'].fromCenters(disk_new['disk1']['back'], s)

    new_layout['D3'] = Dee(315, 1185)
    new_layout['D3'].fromCenters(disk_new['disk2']['front'], s)

    new_layout['D4'] = Dee(315, 1185)
    new_layout['D4'].fromCenters(disk_new['disk2']['back'], s)

    d = new_layout['D2']
    centers = [ (sens.x, sens.y) for sens in d.sensors if sens.x>0 ]
    centers.sort(key=lambda y: y[1], reverse=True)

    y_centers = []
    for i, sens in enumerate(d.sensors):
        if sens.x>0:
            y_centers.append(sens.y)

    y_centers.sort(reverse=True)

    y_positions_individual = set(y_centers)
    y_positions_individual = list(y_positions_individual)
    y_positions_individual.sort(reverse=True)

    y_centers_module = (np.array(y_positions_individual[0::2]) + np.array(y_positions_individual[1::2]))/2

    x_centers_module = {y:[] for y in y_centers_module}
    for x, y in centers:
        if y in y_positions_individual[0::2]:
            i = list(y_positions_individual[0::2]).index(y)
            x_centers_module[y_centers_module[i]].append(x)

    mod_partitions = [
        (1039.65, 1, [7]),
        (953.95, 2, [7, 6]),
        (868.25, 3, [7, 6, 3]),
        (782.55, 3, [6, 6, 6]),
        (696.85, 3, [7, 7, 6]),
        (611.15, 4, [7, 6, 6, 3]),
        (525.45, 4, [7, 7, 6, 3]),
        (439.75, 4, [7, 6, 6, 3]),
        (354.05, 3, [7, 6, 3]),
        (268.35, 3, [7, 6, 6]),
        (182.65, 3, [6, 6, 6]),
        (96.95, 3, [7, 7, 3]),
        (11.25, 3, [6, 6, 3]),
        (-74.45, 3, [6, 6, 3]),
        (-160.15, 3, [6, 6, 6]),
        (-245.85000000000002, 3, [7, 6, 6]),
        (-331.55, 3, [7, 6, 6]),
        (-417.25, 4, [3, 7, 7, 3]),
        (-502.95000000000005, 4, [7, 7, 6, 3]),
        (-588.65, 4, [7, 6, 6, 3]),
        (-674.35, 3, [7, 7, 7]),
        (-760.05, 3, [7, 6, 6]),
        (-845.75, 3, [7, 7, 3]),
        (-931.45, 3, [6, 6, 3]),
        (-1017.1500000000001, 2, [7, 3]),
        (-1102.85, 1, [3]),
    ]

    supermodules = []
    for i, y in enumerate(x_centers_module.keys()):
        _, __, partition = mod_partitions[i]
        partition = reversed(partition) if modify_partitions else partition
        start = 0
        for n_mod in partition:
            tmp = copy.deepcopy(SuperModule.fromSuperModule(SM, n_modules=n_mod, module_gap=SM.module_gap, orientation=SM.orientation, color=colors[n_mod]))
            x = sum(x_centers_module[y][start:start+n_mod])/n_mod
            tmp.move_by(x, y+pb.width/2) ## switch between + and - for D1/D2
            start += n_mod
            supermodules.append(tmp)

    # Split into rings
    centers = set([ sm.y for sm in supermodules ])
    supermodules_dict = {x:[] for x in centers}
    for y in centers:
        for sm in supermodules:
            if sm.y == y:
                supermodules_dict[y].append(sm)

    supermodules_inner = []
    supermodules_middle = []
    supermodules_outer = []

    for y in supermodules_dict.keys():
        if abs(y) < 600:
            supermodules_inner += supermodules_dict[y][:-2]
            supermodules_middle += supermodules_dict[y][-2:-1]
            supermodules_outer += supermodules_dict[y][-1:]
        else:
            supermodules_inner += supermodules_dict[y][:-3]
            supermodules_middle += supermodules_dict[y][-3:-1]
            supermodules_outer += supermodules_dict[y][-1:]

    assert len(supermodules_inner+supermodules_middle+supermodules_outer) == len(supermodules), "Splitting into rings changed the number of supermodules"

    # just printing the "inner" service hybrids for debugging
    inner = plt.Circle((0, 0), 315,fill=None, edgecolor='r')
    rad_boarder = plt.Circle((0, 0), 520,fill=None, edgecolor='black', linewidth=2)
    outer = plt.Circle((0, 0), 1185,fill=None, edgecolor='r')

    fig, ax  = plt.subplots(1,1,figsize=(10,20) )

    plt.gca().add_patch(inner)
    plt.gca().add_patch(rad_boarder)
    plt.gca().add_patch(outer)

    for SM in supermodules_inner:
        fig.gca().add_patch(SM.getPolygon(alpha=0.1))
        fig.gca().add_patch(SM.getPolygon(fill=False))
        for mod in SM.modules:
            fig.gca().add_patch(mod.getPolygon(linewidth=1))
            for sen in mod.sensors:
                fig.gca().add_patch(sen.getPolygon())

    ax.set_xlim(-10, 1300)
    ax.set_ylim(-1300, 1300)

    fig.savefig(f"./figures/inner_SHs.pdf")
    fig.clear()


    inner = plt.Circle((0, 0), 315,fill=None, edgecolor='r')
    rad_boarder = plt.Circle((0, 0), 520,fill=None, edgecolor='black', linewidth=2)
    outer = plt.Circle((0, 0), 1185,fill=None, edgecolor='r')

    fig, ax  = plt.subplots(1,1,figsize=(10,20) )

    plt.gca().add_patch(inner)
    plt.gca().add_patch(rad_boarder)
    plt.gca().add_patch(outer)


    for SM in supermodules:
        fig.gca().add_patch(SM.getPolygon(alpha=0.1))
        SM.color = green_to_red(SM.get_occupancy())
        fig.gca().add_patch(SM.getPolygon(fill=True))
        #for mod in SM.modules:
        #    fig.gca().add_patch(mod.getPolygon(linewidth=1))
        #    for sen in mod.sensors:
        #        fig.gca().add_patch(sen.getPolygon())


    ax.set_xlim(-10, 1300)
    ax.set_ylim(-1300, 1300)

    fig.savefig(f"./figures/occupancy_SH.pdf")

    inner = plt.Circle((0, 0), 315,fill=None, edgecolor='r')
    rad_boarder = plt.Circle((0, 0), 520,fill=None, edgecolor='black', linewidth=2)
    outer = plt.Circle((0, 0), 1185,fill=None, edgecolor='r')

    fig, ax  = plt.subplots(1,1,figsize=(10,20) )

    plt.gca().add_patch(inner)
    plt.gca().add_patch(rad_boarder)
    plt.gca().add_patch(outer)


    occupancies = []
    for SM in supermodules:
        #fig.gca().add_patch(SM.getPolygon(alpha=0.1))
        #SM.color = green_to_red(SM.get_occupancy())
        #fig.gca().add_patch(SM.getPolygon(fill=False))
        for mod in SM.modules:
            mod.color = green_to_red(mod.get_occupancy())
            fig.gca().add_patch(mod.getPolygon(linewidth=1, fill=True))
            #for sen in mod.sensors:
            #    fig.gca().add_patch(sen.getPolygon())
            for s in mod.sensors:
                # do it twice because we still have 16x32 sensors here...
                occupancies.append(s.get_occupancy())
                occupancies.append(s.get_occupancy())


    ax.set_xlim(-10, 1300)
    ax.set_ylim(-1300, 1300)

    fig.savefig(f"./figures/occupancy_module.pdf")

    occupancies = []
    for SM in supermodules:
        for mod in SM.modules:
            for s in mod.sensors:
                occupancies += s.get_occupancy(per_etroc=True)

    import hist
    oc_axis = hist.axis.Regular(20, 0.0, 1, name="oc", label=r"occupancy")
    oc_hist = hist.Hist(oc_axis)
    oc_hist.fill(occupancies)

    l1a_rate = 750000
    n_etrocs = 28600

    bandwidth_header = l1a_rate * n_etrocs * 80 / 1e12
    bandwidth_miniheader = l1a_rate * n_etrocs * 16 / 1e12

    bandwidth_total = 80*n_etrocs*l1a_rate/1e12 + sum(oc_hist.axes[0].centers*40*256/100 * oc_hist.values()*16*l1a_rate/1e12)
    bandwidth_total_mh = 16*n_etrocs*l1a_rate/1e12 + sum(oc_hist.axes[0].centers*40*256/100 * oc_hist.values()*16*l1a_rate/1e12)
    bandwidth_total_mh_nocal = 16*n_etrocs*l1a_rate/1e12 + sum(oc_hist.axes[0].centers*32*256/100 * oc_hist.values()*16*l1a_rate/1e12)
    bandwidth_total_noheader_nocal = sum(oc_hist.axes[0].centers*32*256/100 * oc_hist.values()*16*l1a_rate/1e12)

    print(f"- TDR like bandwidth {bandwidth_total}Tbps")
    print(f"- Mini headers bandwidth {bandwidth_total_mh}Tbps")
    print(f"- Mini headers, strip cal code bandwidth {bandwidth_total_mh_nocal}Tbps")
    print(f"- No headers, strip cal code bandwidth {bandwidth_total_noheader_nocal}Tbps")
