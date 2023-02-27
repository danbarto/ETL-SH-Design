#!/usr/bin/env python3
import numpy as np
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


    # Simple default/single sensor configurations
    configs = {
        "HPK_split4_2p5fC": hpk_split4_2p5fc,
        "HPK_split4_5fC":   hpk_split4_5fc,
        "HPK_split4_10fC":  hpk_split4_10fc,
        "HPK_split4_15fC":  hpk_split4_15fc,

        "FBK_w13_2p5fC": fbk_w13_2p5fc,
        "FBK_w13_5fC":   fbk_w13_5fc,
        "FBK_w13_10fC":  fbk_w13_10fc,
        "FBK_w13_15fC":  fbk_w13_15fc,
    }

    for name in configs:

        print (f"\n Working on {name}")

        fig, ax  = plt.subplots(1,1,figsize=(10,20) )

        inner = plt.Circle((0, 0), 315,fill=None, edgecolor='r')
        outer = plt.Circle((0, 0), 1185,fill=None, edgecolor='r')

        fig.gca().add_patch(inner)
        fig.gca().add_patch(outer)

        BV_lines = 0
        currents = []

        for i, SM in enumerate(supermodules):
            SM.find_BV_config(configs[name], verbose=False, min_split=1)
            fig.gca().add_patch(SM.getPolygon(fill=False))
            BV_lines += SM.BV_lines
            currents += SM.currents
            for mod in SM.modules:
                if mod.problematic:
                    fig.gca().add_patch(mod.getPolygon(edgecolor='red'))
                else:
                    fig.gca().add_patch(mod.getPolygon(linewidth=1))
                for sen in mod.sensors:
                    fig.gca().add_patch(sen.getPolygon())

        print (f"BV leads needed: {BV_lines}")
        print (f"BV channels needed: {len(currents)}")

        ax.set_xlim(-10, 1300)
        ax.set_ylim(-1300, 1300)

        fig.savefig(f"./figures/{name}.pdf")

        fig.clear()
        del fig

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

    inner = plt.Circle((0, 0), 315,fill=None, edgecolor='r')
    rad_boarder = plt.Circle((0, 0), 520,fill=None, edgecolor='black', linewidth=2)
    outer = plt.Circle((0, 0), 1185,fill=None, edgecolor='r')

    fig, ax  = plt.subplots(1,1,figsize=(10,20) )

    plt.gca().add_patch(inner)
    plt.gca().add_patch(rad_boarder)
    plt.gca().add_patch(outer)

    BV_lines = 0
    currents = []
    n_modules = {'FBK': 0, 'HPK': 0}

    for SM in supermodules_inner + supermodules_middle:
        SM.find_BV_config(fbk_w13_5fc, verbose=False, min_split=2)
        fig.gca().add_patch(SM.getPolygon(alpha=0.1))
        fig.gca().add_patch(SM.getPolygon(fill=False))
        BV_lines += SM.BV_lines
        currents += SM.currents
        n_modules['FBK'] += SM.n_modules
        for mod in SM.modules:
            if mod.problematic:
                fig.gca().add_patch(mod.getPolygon(edgecolor='red'))
            else:
                fig.gca().add_patch(mod.getPolygon(linewidth=1))
            for sen in mod.sensors:
                fig.gca().add_patch(sen.getPolygon())

    for SM in supermodules_outer:
        SM.find_BV_config(hpk_split4_5fc, verbose=False, min_split=2)
        fig.gca().add_patch(SM.getPolygon(alpha=0.1))
        fig.gca().add_patch(SM.getPolygon(fill=False))
        BV_lines += SM.BV_lines
        currents += SM.currents
        n_modules['HPK'] += SM.n_modules
        for mod in SM.modules:
            if mod.problematic:
                fig.gca().add_patch(mod.getPolygon(edgecolor='red'))
            else:
                fig.gca().add_patch(mod.getPolygon(linewidth=1))
            for sen in mod.sensors:
                fig.gca().add_patch(sen.getPolygon())

    ax.set_xlim(-10, 1300)
    ax.set_ylim(-1300, 1300)

    print (f"BV lines needed: {BV_lines}")
    print (f"BV channels needed: {len(currents)}")
    print (f"FBK: {n_modules['FBK']}, HPK: {n_modules['HPK']}")

    fig.savefig(f"./figures/realistic_5fC.pdf")

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

    fig.savefig(f"./figures/inner_ring.pdf")



    bins = "10,0,1"
    h_curr = Hist1D(currents, bins=bins)

    fig, ax = plt.subplots(1,1,figsize=(7,7))

    h_curr.plot()

    ax.set_ylabel('channel count')
    ax.set_xlabel('I (mA)')

    fig.savefig(f"figures/currents_realistic_5fC.pdf")
    fig.clear()


    inner = plt.Circle((0, 0), 315,fill=None, edgecolor='r')
    rad_boarder = plt.Circle((0, 0), 520,fill=None, edgecolor='black', linewidth=2)
    outer = plt.Circle((0, 0), 1185,fill=None, edgecolor='r')

    fig, ax  = plt.subplots(1,1,figsize=(10,20) )

    plt.gca().add_patch(inner)
    plt.gca().add_patch(rad_boarder)
    plt.gca().add_patch(outer)

    BV_lines = 0
    currents = []
    n_modules = {'FBK': 0, 'HPK': 0}

    for SM in supermodules_inner:
        SM.find_BV_config(fbk_w13_10fc, verbose=False, min_split=3)
        fig.gca().add_patch(SM.getPolygon(alpha=0.1))
        fig.gca().add_patch(SM.getPolygon(fill=False))
        BV_lines += SM.BV_lines
        currents += SM.currents
        n_modules['FBK'] += SM.n_modules
        for mod in SM.modules:
            if mod.problematic:
                fig.gca().add_patch(mod.getPolygon(edgecolor='red'))
            else:
                fig.gca().add_patch(mod.getPolygon(linewidth=1))
            for sen in mod.sensors:
                fig.gca().add_patch(sen.getPolygon())

    for SM in supermodules_middle + supermodules_outer:
        SM.find_BV_config(hpk_split4_10fc, verbose=False, min_split=3)
        fig.gca().add_patch(SM.getPolygon(alpha=0.1))
        fig.gca().add_patch(SM.getPolygon(fill=False))
        BV_lines += SM.BV_lines
        currents += SM.currents
        n_modules['HPK'] += SM.n_modules
        for mod in SM.modules:
            if mod.problematic:
                fig.gca().add_patch(mod.getPolygon(edgecolor='red'))
            else:
                fig.gca().add_patch(mod.getPolygon(linewidth=1))
            for sen in mod.sensors:
                fig.gca().add_patch(sen.getPolygon())

    ax.set_xlim(-10, 1300)
    ax.set_ylim(-1300, 1300)

    print (f"BV lines needed: {BV_lines}")
    print (f"BV channels needed: {len(currents)}")
    print (f"FBK: {n_modules['FBK']}, HPK: {n_modules['HPK']}")

    fig.savefig(f"./figures/realistic_10fC.pdf")
    fig.clear()

    bins = "10,0,1"
    h_curr = Hist1D(currents, bins=bins)

    fig, ax = plt.subplots(1,1,figsize=(7,7))

    h_curr.plot()

    ax.set_ylabel('channel count')
    ax.set_xlabel('I (mA)')

    fig.savefig(f"figures/currents_realistic_10fC.pdf")
    fig.clear()



    inner = plt.Circle((0, 0), 315,fill=None, edgecolor='r')
    rad_boarder = plt.Circle((0, 0), 520,fill=None, edgecolor='black', linewidth=2)
    outer = plt.Circle((0, 0), 1185,fill=None, edgecolor='r')

    fig, ax  = plt.subplots(1,1,figsize=(10,20) )

    plt.gca().add_patch(inner)
    plt.gca().add_patch(rad_boarder)
    plt.gca().add_patch(outer)

    BV_lines = 0
    currents = []
    n_modules = {'FBK': 0, 'HPK': 0}

    for SM in supermodules_inner:
        SM.find_BV_config(fbk_w13_10fc, verbose=False, min_split=1)
        fig.gca().add_patch(SM.getPolygon(alpha=0.1))
        fig.gca().add_patch(SM.getPolygon(fill=False))
        BV_lines += SM.BV_lines
        currents += SM.currents
        n_modules['FBK'] += SM.n_modules
        for mod in SM.modules:
            if mod.problematic:
                fig.gca().add_patch(mod.getPolygon(edgecolor='red'))
            else:
                fig.gca().add_patch(mod.getPolygon(linewidth=1))
            for sen in mod.sensors:
                fig.gca().add_patch(sen.getPolygon())

    for SM in supermodules_middle + supermodules_outer:
        SM.find_BV_config(hpk_split4_10fc, verbose=False, min_split=1)
        fig.gca().add_patch(SM.getPolygon(alpha=0.1))
        fig.gca().add_patch(SM.getPolygon(fill=False))
        BV_lines += SM.BV_lines
        currents += SM.currents
        n_modules['HPK'] += SM.n_modules
        for mod in SM.modules:
            if mod.problematic:
                fig.gca().add_patch(mod.getPolygon(edgecolor='red'))
            else:
                fig.gca().add_patch(mod.getPolygon(linewidth=1))
            for sen in mod.sensors:
                fig.gca().add_patch(sen.getPolygon())

    ax.set_xlim(-10, 1300)
    ax.set_ylim(-1300, 1300)

    print (f"BV lines needed: {BV_lines}")
    print (f"BV channels needed: {len(currents)}")
    print (f"FBK: {n_modules['FBK']}, HPK: {n_modules['HPK']}")

    fig.savefig(f"./figures/realistic_10fC_no_min_split.pdf")
    fig.clear()

    bins = "10,0,1"
    h_curr = Hist1D(currents, bins=bins)

    fig, ax = plt.subplots(1,1,figsize=(7,7))

    h_curr.plot()

    ax.set_ylabel('channel count')
    ax.set_xlabel('I (mA)')

    fig.savefig(f"figures/currents_realistic_10fC_no_min_split.pdf")
    fig.clear()


    # Other configurations
    modules = []
    for sm in supermodules:
        modules += sm.modules

    for m in modules:
        m.r()

    modules.sort(key=lambda x: x._r, reverse=False)
    #modules.sort(key=lambda x: x.y, reverse=True)

    # find all modules that have a sensor in r<520mm
    fbk = []
    hpk = []
    for m in modules:
        rmin, rmax = get_sensors_r_min_max([m])
        if rmin<520:
            fbk.append(m)
        else:
            hpk.append(m)


    # sort from large r to small r
    fbk.sort(key=lambda x: x._r, reverse=True)
    hpk.sort(key=lambda x: x._r, reverse=True)

    # now find all modules that can be grouped together, starting from largest r
    # ignoring the currents for now because no one gives us any fucking information on power supplies, so keep on shooting in the dark. hurray!
    groupings = []
    first = True
    for m in hpk:
        rmin, rmax = get_sensors_r_min_max([m])
        if first:
            rmin_for_real = hpk_split4_10fc(rmax)  # being optimistic here with 15fC, because no one knows anything anyway
            groupings.append([])
            first = False
        if rmin > rmin_for_real:
            #print (rmin, rmin_for_real)
            groupings[-1].append(m)
        else:
            rmin_for_real = hpk_split4_10fc(rmax)  # being optimistic here with 15fC, because no one knows anything anyway
            groupings.append([m])

    first = True
    for m in fbk:
        rmin, rmax = get_sensors_r_min_max([m])
        if first:
            rmin_for_real = fbk_w13_10fc(rmax)  # being optimistic here with 15fC, because no one knows anything anyway
            groupings.append([])
            first = False
        if rmin > rmin_for_real:
            #print (rmin, rmin_for_real)
            groupings[-1].append(m)
        else:
            rmin_for_real = fbk_w13_10fc(rmax)  # being optimistic here with 15fC, because no one knows anything anyway
            groupings.append([m])

    # make a plot of the above groupings.
    inner = plt.Circle((0, 0), 315,fill=None, edgecolor='r')
    rad_boarder = plt.Circle((0, 0), 520,fill=None, edgecolor='black', linewidth=2)
    outer = plt.Circle((0, 0), 1185,fill=None, edgecolor='r')
    fig, ax  = plt.subplots(1,1,figsize=(10,20) )

    plt.gca().add_patch(inner)
    plt.gca().add_patch(rad_boarder)
    plt.gca().add_patch(outer)

    for i, group in enumerate(groupings):
        # define a color per group
        print (color_list[i])
        for m in group:
            print (m._r)
            fig.gca().add_patch(m.getPolygon(edgecolor='black'))
            for s in m.sensors:
                fig.gca().add_patch(s.getPolygon(color=color_list[i], active=True))
    ax.set_xlim(-10, 1300)
    ax.set_ylim(-1300, 1300)

    fig.savefig(f"./figures/sergey_10fC.pdf")
    fig.clear()

    # FIXME I don't know what I started below here.
    current = 0
    rmin_orig, rmax_orig = get_sensors_r_min_max([modules[0]])
    pop = []
    #for m in modules[1:]:
    #    rmin, rmax = get_sensors_r_min_max([m])
    #    if rmin > fbk_w13_10fc(rmax_orig) and current<20:
    #        current += m.get_current()
    #        pop.append(m)
    #        modules.pop(modules.index(m))
