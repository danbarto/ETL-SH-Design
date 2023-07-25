#!/usr/bin/env python3
'''
ETL Geometry with shapely, WIP
'''
import time
import random
import copy
import numpy as np
import os
import pickle
import matplotlib.pyplot as plt
from shapely.geometry import Polygon, Point
from shapely.plotting import plot_polygon
from shapely.affinity import rotate, translate
from shapely.ops import unary_union
from ETL import *
from sensors import *

import matplotlib.pyplot as plt
import mplhep as hep
plt.style.use(hep.style.CMS)


class three_vector:
    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z
        self.r     = np.sqrt(x**2+y**2)
        self.theta = np.arctan2(self.r, z)
        self.eta   = -np.log(np.tan(self.theta/2))
        self.phi   = np.arctan2(y, x)

    @classmethod
    def fromEtaPhi(cls, eta, phi, z):
        cls.eta = eta
        cls.phi = phi
        cls.z = z
        cls.theta = 2*np.arctan(np.exp(cls.eta*(-1)))
        cls.r = z*np.tan(cls.theta)
        cls.x = cls.r*np.cos(cls.phi)
        cls.y = cls.r*np.sin(cls.phi)

        return cls

z_ref = 2.99825
z = [
    2.99825,
    3.0055,
    3.02075,
    3.0285,
]

def time_res(n):
    return 50/np.sqrt(n)  # NOTE assumes a 50ps single hit time resolution

def rad(deg):
    return deg * np.pi / 180

def make_inner_attachment(angle=0, height=84, width=88.5, taper=5.4, inner=336):
    '''
    angle - in degrees, 0 is vertical upright, 90 is horizontal
    '''
    taper_rad = rad(taper)
    angle_rad = rad(angle)

    coords = [
        (-width/2, -height),
        (-width/2 + height*np.tan(taper_rad), 0),
        (width/2 - height*np.tan(taper_rad), 0),
        (width/2, -height),
        ]
    tmp = Polygon(coords)
    tmp = rotate(tmp, -angle, origin=(0,0))
    tmp = translate(tmp, np.sin(angle_rad)*(inner+height), np.cos(angle_rad)*(inner+height))

    return tmp

def make_outer_attachment(angle=0, height=107.5, width=68):
    angle_rad = rad(angle)
    coords = [
        (-width/2, -height),
        (-width/2, 0),
        (width/2, 0),
        (width/2, -height)
    ]

    circ = Point(0, -height).buffer(width/2)
    tmp = Polygon(coords)
    tmp = unary_union([circ, tmp])
    tmp = rotate(tmp, -angle, origin=(0,0))
    tmp = translate(tmp, np.sin(angle_rad)*(etl_outer), np.cos(angle_rad)*(etl_outer))

    return tmp

def make_inner(r_out=302, r_in=336):
    outer = Point(0, 0).buffer(r_in)
    inner = Point(0, 0).buffer(r_out)

    exter = list(zip(outer.exterior.xy[0], outer.exterior.xy[1]))
    inter = list(zip(inner.exterior.xy[0], inner.exterior.xy[1]))[::-1]

    return Polygon(exter, [inter])

def make_outer(etl_outer, feedthrough):
    outer = Point(0, 0).buffer(etl_outer+100)
    inner = Point(0, 0).buffer(etl_outer)

    inter = list(zip(inner.exterior.xy[0], inner.exterior.xy[1]))[::-1]
    exter = list(zip(outer.exterior.xy[0], outer.exterior.xy[1]))

    feed1 = Polygon(
        [
            (feedthrough, np.sqrt(etl_outer**2 - feedthrough**2)),
            (etl_outer, np.sqrt(etl_outer**2 - feedthrough**2)),
            (etl_outer, -np.sqrt(etl_outer**2 - feedthrough**2)),
            (feedthrough, -np.sqrt(etl_outer**2 - feedthrough**2))
        ]
    )
    feed2 = Polygon(
        [
            (-feedthrough, np.sqrt(etl_outer**2 - feedthrough**2)),
            (-etl_outer, np.sqrt(etl_outer**2 - feedthrough**2)),
            (-etl_outer, -np.sqrt(etl_outer**2 - feedthrough**2)),
            (-feedthrough, -np.sqrt(etl_outer**2 - feedthrough**2))
        ]
    )

    disc = Polygon(exter, [inter])
    disc = unary_union([feed1, feed2, disc])

    return disc

def within(module, dee):
    return np.any(np.array([module.within(x) for x in dee]))

def overlaps(module, dee):
    return np.any(np.array([module.overlaps(x) for x in dee]))

def update_pkl(out):
    pkl = 'layouts/attachment_studies.pkl'
    if os.path.isfile(pkl):
        with open(pkl, 'rb') as f:
            out_old = pickle.load(f)
        out_old.update(out)
    else:
        out_old = out
    with open(pkl, 'wb') as f:
        pickle.dump(out_old, f)

if __name__ == '__main__':

    # some default values
    etl_outer   = 1185  # outer radius (same for all studied layouts)
    etl_inner1  = 336  # larges inner radius we had
    etl_inner2  = 302  # smallest inner radius we had
    feedthrough = 1120  # feedthrough distance from center


    import argparse

    argParser = argparse.ArgumentParser(description = "Argument parser")
    argParser.add_argument(
        '--skip_acceptance',
        action = 'store_true',
        default = False,
        help = "Don't run the acceptance studies",
    )
    argParser.add_argument(
        '--comparison',
        action = 'store_true',
        default = False,
        help = "Make a comparison plot of the different scenarios",
    )
    argParser.add_argument(
        '--modules',
        action = 'store',
        default = "S",
        choices = ["S", "M", "L"],
        help = 'Module size',
    )
    argParser.add_argument(
        '--dee_layout',
        action = 'store',
        default = "updated",
        choices = ["baseline", "updated", "plain", "updatedV2"],
        help = 'Select which dee layout to use',
    )
    argParser.add_argument(
        '--seal',
        action = 'store_true',
        default = False,
        help = 'Add space for seal',
    )
    argParser.add_argument(
        '--no_feedthrough',
        action = 'store_true',
        default = False,
        help = "Don't put the geometry for feedthrough",
    )


    args = argParser.parse_args()

    module_size = args.modules
    layout      = args.dee_layout

    nose = Point(0, 0).buffer(etl_inner2)  # minimum size nose

    if args.no_feedthrough:
        outer = make_outer(etl_outer, feedthrough=etl_outer)  # add no feedthrough
    else:
        outer = make_outer(etl_outer, feedthrough)

    attachments = []

    # Make the different layouts
    run_name = f'{layout}_{module_size}'
    if args.seal:
        run_name += '_with_seal'
    if args.no_feedthrough:
        run_name += '_no_feedthrough'

    plot_dir = f"./figures/{run_name}/"
    if not os.path.isdir(plot_dir):
        os.makedirs(plot_dir)

    if layout in ['plain', 'baseline']:
        r_inner = 336
        r_inner_first = 336
        inner = make_inner(r_in=300, r_out=r_inner)

    if layout in ['baseline']:
        # baseline attachments from natalia
        attachments += [make_inner_attachment(angle=x) for x in [30,90,150,-30,-90,-150]]
        attachments += [make_outer_attachment(angle=x) for x in [2.5, 68, 112, 177.5, -2.5, -68, -112, -177.5]]
        attachments_first = copy.deepcopy(attachments)

    if layout in ['updated']:
        # NOTE the outer attachments remain unchanged, as per https://indico.cern.ch/event/1286311/contributions/5405295/attachments/2647635/4583271/20230515_update%20on%20mechanics%20-%20inner%20supports.pdf
        attachments += [make_outer_attachment(angle=x) for x in [2.5, 68, 112, 177.5, -2.5, -68, -112, -177.5]]
        attachments_first = copy.deepcopy(attachments)
        # new inner attachment from natalia
        r_inner = 302  # 302 --> 322 because of missing seal
        inner = make_inner(r_in=300, r_out=r_inner)
        attachments += [make_inner_attachment(
            angle=x,
            width=170,
            height=(796/2)-r_inner,
            taper=0,
            inner = r_inner,
        ) for x in [0,180]]
        attachments += [make_inner_attachment(
            angle=x,
            width=110,
            height=395-r_inner,
            taper=0,
            inner = r_inner,
        ) for x in [90,270]]

        #### Different inner radius restriction for FIRST layer only
        r_inner_first = 322
        inner_first = make_inner(r_in=300, r_out=r_inner_first)
        attachments_first += [make_inner_attachment(
            angle=x,
            width=170,
            height=(796/2)-r_inner_first,
            taper=0,
            inner = r_inner,
        ) for x in [0,180]]
        attachments_first += [make_inner_attachment(
            angle=x,
            width=110,
            height=395-r_inner_first,
            taper=0,
            inner = r_inner,
        ) for x in [90,270]]



    fig, ax = plt.subplots(figsize=(8, 8))
    plot_polygon(nose, ax=ax, add_points=False, color='red')
    plot_polygon(inner, ax=ax, add_points=False)
    plot_polygon(outer, ax=ax, add_points=False)
    for at in attachments:
        plot_polygon(at, ax=ax, add_points=False)

    modules = []
    modules.append(Module(56, 43, x=340, y=0))
    modules.append(Module(56, 43, x=500, y=0))
    modules.append(Module(56, 43, x=1115, y=0))

    dee_geo = attachments + [nose, inner, outer]
    if layout in ['updated']:
        dee_geo_first = attachments_first + [nose, inner_first, outer]
    if layout in ['baseline', 'plain']:
        dee_geo_first = dee_geo

    # NOTE: this is just a test of the collision detection
    for mod in modules:
        collision = overlaps(mod.getPolygon(), dee_geo)
        print (collision, mod.getPolygon())
        plot_polygon(mod.getPolygon(), ax=ax, add_points=False, color='green' if not collision else 'red')

    fig.savefig(plot_dir+'test2.png')

    s = Sensor(20.8, 20.8)  # realistic size of the active area. 21.6 x 21.4 are outer dimensions

    ## NOTE outdated module definitions, kept here for reference
    ## sensor_x = (42.5/2 + 0.5)
    #m = Module(44.20, 56.50, n_sensor_x=2, n_sensor_y=2, sensor_distance_y=22.5, sensor_distance_x=sensor_x+0.1)
    #m = Module(43.10, 56.50, n_sensor_x=2, n_sensor_y=2, sensor_distance_y=22.5, sensor_distance_x=sensor_x+0.1)
    #m = Module(44.20, 57.50, n_sensor_x=2, n_sensor_y=2, sensor_distance_y=22.5, sensor_distance_x=sensor_x+0.1)

    # NOTE modules that should be used
    if module_size in ['S']:
        m = Module(43.10, 56.50, n_sensor_x=2, n_sensor_y=2, sensor_distance_y = 22.4, sensor_distance_x = 22.6)
        module_gap = 0.5
    elif module_size in ['M']:
        m = Module(44.10, 57.50, n_sensor_x=2, n_sensor_y=2, sensor_distance_y = 22.4, sensor_distance_x = 22.6)
        module_gap = 0.5
    elif module_size in ['L']:
        m = Module(44.10, 57.50, n_sensor_x=2, n_sensor_y=2, sensor_distance_y = 22.4, sensor_distance_x = 22.6)
        module_gap = 0.6
    else:
        print(f"Don't know what to do with a module of size {module_size}. Get in shape!")
        raise("NotImplementedError")

    m.populate(s)

    if module_size in ['M', 'L']:
        rb = ReadoutBoard(10, 57.5, color='green')
        pb = ReadoutBoard(10, 28.5, color='red')
    elif module_size in ['S']:
        rb = ReadoutBoard(10, 56.5, color='green')
        pb = ReadoutBoard(10, 29.5, color='red')
    else:
        print(f"Don't know what to do with a module of size {module_size}. Get in shape!")
        raise("NotImplementedError")

    # make the configurations
    print ("Populating DEEs:")
    configs = [
        ('above', True, False, 'face1'),
        ('above', False, True, 'face2'),
        ('below', True, False, 'face3'),
        ('below', False, True, 'face4')
    ]

    dees = {}
    counts = {3:0, 6:0, 7:0}
    total_modules = 0
    for i, cfg in enumerate(configs):
        print(f"- {cfg[3]}")

        SM = SuperModule(m, pb, rb, n_modules=3, orientation=cfg[0], module_gap=module_gap)
        if i==0:
            dees[cfg[3]] = Dee(
                r_inner_first,  # 315, # old number is 315
                etl_outer,
            )
            print("Adding special first layer geometry")
            dees[cfg[3]].add_geometries(dee_geo_first)
        else:
            dees[cfg[3]] = Dee(
                r_inner,  # 315, # old number is 315
                etl_outer,
            )
            print("Adding geometry")
            dees[cfg[3]].add_geometries(dee_geo)

        dees[cfg[3]].populate(
            SM,
            center_RB=cfg[1],
            center_PB=cfg[2],
            edge_x = 6 if not args.seal else 25,
        )

        total_modules += dees[cfg[3]].n_modules

        for sm in dees[cfg[3]].supermodules:
            counts[sm.n_modules] += 1

        print(f"Layer {cfg[3]} has {dees[cfg[3]].n_modules} modules")
        print(f"With a supermodule count: {len(dees[cfg[3]].supermodules)}")
    print(f"This configuration results in {total_modules} modules for 1/4th of ETL.")
    print(f"This configuration results in {total_modules*4} modules for ETL.")

    print(f"Small RBs {counts[3]*4})")
    print(f"Medium RBs {counts[6]*4})")
    print(f"Large RBs {counts[7]*4})")

    print("Done. Plotting now.")

    fig, ax = plt.subplots(figsize=(8, 8))
    print("- Plotting geometry")
    for geo in dee_geo:
        plot_polygon(
            geo,
            ax=ax,
            add_points=False,
        )

    print("- Plotting active area of LGADs:")
    for cfg in configs:
        print(f"  - {cfg[3]}")

        for slot in dees[cfg[3]].slots_flat:
            if slot.covered:
                for mod in slot.modules:
                    for sen in mod.sensors:
                        ax.add_patch(
                            sen.getPolygon(active=True, simple=True),
                        )
                        # this MOFO down there is incredibly slow!
                        #plot_polygon(
                        #    sen.getPolygon(active=True, simple=False),
                        #    ax=ax,
                        #    add_points=False
                        #)
    fig.savefig(plot_dir+'sensors_all_layers.pdf')

    fig, ax = plt.subplots(figsize=(8, 8))
    print("- Plotting geometry of first face")
    for geo in dee_geo:
        plot_polygon(
            geo,
            ax=ax,
            add_points=False,
        )

    print("- Plotting active area of LGADs:")
    for cfg in configs[:1]:
        print(f"  - {cfg[3]}")

        for slot in dees[cfg[3]].slots_flat:
            if slot.covered:
                for mod in slot.modules:
                    for sen in mod.sensors:
                        ax.add_patch(
                            sen.getPolygon(active=True, simple=True),
                        )

        for sm in dees['face1'].supermodules:
            plot_polygon(
                sm.getPolygon(),
                ax=ax,
                add_points=False,
                color = sm.color,
            )

    fig.savefig(plot_dir+'config_face1.pdf')

    with open('centers_face1.txt', 'w') as f_out:
        for sm in dees['face1'].supermodules:
            for m in sm.modules:
                f_out.write(f'{m.x}, {m.y}\n')
                f_out.write(f'-{m.x}, {m.y}\n')


    # latest bias voltage studies with Natalia's latest attachment design
    # will assume a 10fC signal range
    # assuming 20mA HV supply channels
    # 0.75mA surface currents per sensor (conservative)
    # leakage current is radiation dependent, and assumed here at the end of life
    run_bias = True
    if run_bias:
        modules = []
        for sm in dees['face1'].supermodules:
            modules += sm.modules

        for m in modules:
            m.r()

        modules.sort(key=lambda x: x._r, reverse=False)

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
        groupings = []
        first = True
        current = 0
        new_group = False
        for m in hpk:
            rmin, rmax = get_sensors_r_min_max([m])
            current += m.get_current()
            if current > 20:
                new_group = True
            if first:
                rmin_for_real = hpk_split4_10fc(rmax)
                groupings.append([])
                first = False
            if rmin > rmin_for_real and not new_group:
                #print (rmin, rmin_for_real)
                groupings[-1].append(m)
            else:
                new_group = False
                current = m.get_current()
                rmin_for_real = hpk_split4_10fc(rmax)
                groupings.append([m])

        first = True
        current = 0
        new_group = False
        for m in fbk:
            rmin, rmax = get_sensors_r_min_max([m])
            if current + m.get_current() > 20:
                new_group = True
            else:
                current += m.get_current()
            if first:
                rmin_for_real = fbk_w13_10fc(rmax)
                groupings.append([])
                first = False
            if rmin > rmin_for_real and not new_group:
                groupings[-1].append(m)
            else:
                new_group = False
                current = m.get_current()
                rmin_for_real = fbk_w13_10fc(rmax)
                groupings.append([m])

        for group in groupings:
            current = 0
            for m in group:
                current += m.get_current()
            if current > 20:
                print("Found too large current in one group")


    run_acceptance = not args.skip_acceptance
    if run_acceptance:
        print ("Running acceptance study")
        from tqdm import tqdm
        import hist
        starttime = time.time()

        eta_max = 3.000  # was 2.950 for TAMALES so far
        eta_min = 1.659
        eta_range = eta_max-eta_min

        nEvents = int(1e6)

        eta = np.random.rand(nEvents)*eta_range + eta_min
        phi = np.random.rand(nEvents)*np.pi - np.pi/2

        vec = three_vector.fromEtaPhi(eta, phi, np.ones(nEvents)*3000)

        vec_list = []
        for x,y in zip(vec.x, vec.y):
            vec_list.append(three_vector(x,y,3000))

        # We also want to get the number of hits.
        for v in tqdm(vec_list):

            n = 0
            iLayer = 0

            for layer in ['face1', 'face2', 'face3', 'face4']:
                x_shift = 1000*(z[iLayer]-z_ref)*np.tan(v.theta)*np.cos(v.phi)
                y_shift = 1000*(z[iLayer]-z_ref)*np.tan(v.theta)*np.sin(v.phi)
                x,y = ((v.x + x_shift), (v.y + y_shift))

                if dees[layer].intersect(x, y):
                    n += 1
                iLayer += 1

            v.n = n
            v.t = 45 / np.sqrt(n)

        nHits = list(map(lambda x: x.n, vec_list))

        eta_phi_n = np.array(
            list(map(lambda x: [x.eta, x.phi, x.n], vec_list))
        )

        h_eta_phi_n = hist.Hist(
            hist.axis.Regular(40, eta_min, eta_max, name="eta", label=r'$\eta$'),
            hist.axis.Regular(40, -np.pi/2, np.pi/2, name="phi", label=r'$\phi$'),
            hist.axis.Integer(0,4, name="n")
        ).fill(*eta_phi_n.T)

        # profile number of hits
        h_eta_phi = h_eta_phi_n.profile("n")

        fig, ax = plt.subplots(1,1,figsize=(10,10))

        h_eta_phi.plot2d(ax=ax)

        fig.savefig(plot_dir+"nhits_profile_eta_phi.png")


        # Average time resolution
        eta_phi_t = np.array(
            list(map(lambda x: [x.eta, x.phi, x.t], vec_list))
        )

        h_eta_phi_t = hist.Hist(
            hist.axis.Regular(40, eta_min, eta_max, name="eta", label=r'$\eta$'),
            hist.axis.Regular(40, -np.pi/2, np.pi/2, name="phi", label=r'$\phi$'),
            hist.axis.Regular(40, 0, 100, name="t"),
            #hist.axis.Integer(0,4, name="n")
        ).fill(*eta_phi_t.T)

        # profile number of hits
        h_eta_phi2 = h_eta_phi_t.profile("t")

        fig, ax = plt.subplots(1,1,figsize=(10,10))

        h_eta_phi2.plot2d(ax=ax)

        fig.savefig(plot_dir+"time_profile_eta_phi.png")

        fig, ax = plt.subplots(1,1,figsize=(10,10))
        plt.scatter(
            list(map(lambda x: x.x, vec_list)),
            list(map(lambda x: x.y, vec_list)),
            c=list(map(lambda x: x.n, vec_list)),
            vmin=0, vmax=4, s=0.2,
        )
        cbar = plt.colorbar(ax=ax)
        cbar.set_label('ETL hits')

        fig.savefig(plot_dir+"scatter.png")


        r_phi_n = np.array(
            list(map(lambda x: [x.r, x.phi, x.n], vec_list))
        )
        h_r_phi_n = hist.Hist(
            hist.axis.Regular(40, 300, 1185, name="r", label=r'$r$'),
            hist.axis.Regular(40, -np.pi/2, np.pi/2, name="phi", label=r'$\phi$'),
            hist.axis.Integer(0,4, name="n")
        ).fill(*r_phi_n.T)

        # profile number of hits
        h_r_phi = h_r_phi_n.profile("n")


        fig, ax = plt.subplots(1,1,figsize=(10,10))

        h_r_phi.plot2d(ax=ax)

        fig.savefig(plot_dir+"nhits_profile_r_phi.png")



        fig, ax = plt.subplots(1,1,figsize=(10,10))
        h_nhits_phi = h_eta_phi_n[{'eta':sum}].profile('n')
        h_nhits_phi.plot1d(ax=ax)
        fig.savefig(plot_dir+"nhits_profile_phi.png")

        fig, ax = plt.subplots(1,1,figsize=(10,10))
        h_nhits_eta = h_eta_phi_n[{'phi':sum}].profile('n')
        h_nhits_eta.plot1d(ax=ax)
        fig.savefig(plot_dir+"nhits_profile_eta.png")

        fig, ax = plt.subplots(1,1,figsize=(10,10))
        h_nhits_eta2 = h_eta_phi_t[{'phi':sum}].profile('t')
        h_nhits_eta2.plot1d(ax=ax)
        fig.savefig(plot_dir+"time_profile_eta.png")

        endtime = time.time()

        print (endtime-starttime)

        res_to_store = {
            'time': h_eta_phi_t,
            'hits': h_eta_phi_n,
        }
        update_pkl({run_name: res_to_store})

        # NOTE run some sanity check for n hits
        # there are events with 3 hits, and we should make sure that those are real
        # just pick the first event where this happens and look at the hit locations
        x_shifts = []
        y_shifts = []
        for v in vec_list:

            iLayer = 0
            if v.n>2:
                print(f'{v.x=}')
                print(f'{v.y=}')
                print(f'{v.eta=}')
                print(f'{v.theta=}')
                print(f'{v.phi=}')
                for layer in ['face1', 'face2', 'face3', 'face4']:
                    x_shift = 1000*(z[iLayer]-z_ref)*np.tan(v.theta)*np.cos(v.phi)
                    y_shift = 1000*(z[iLayer]-z_ref)*np.tan(v.theta)*np.sin(v.phi)
                    x,y = ((v.x + x_shift), (v.y + y_shift))
                    print(f'{x_shift=}')
                    print(f'{y_shift=}')
                    x_shifts.append(x_shift)
                    y_shifts.append(y_shift)
                    iLayer += 1

                break

        fig, ax = plt.subplots(figsize=(8, 8))
        for cfg in configs[:1]:
            print(f"  - {cfg[3]}")

            for slot in dees[cfg[3]].slots_flat:
                if slot.covered:
                    for mod in slot.modules:
                        for sen in mod.sensors:
                            ax.add_patch(
                                sen.getPolygon(active=True, simple=True),
                            )

        plt.gca().add_patch(plt.Circle((v.x, v.y), 1,fill='full', facecolor='red', edgecolor='red'))
        ax.set_xlim(v.x-100, v.x+100)
        ax.set_ylim(v.y-100, v.y+100)
        fig.savefig('figures/intersect_face1.pdf')


        fig, ax = plt.subplots(figsize=(8, 8))
        for cfg in configs[1:2]:
            print(f"  - {cfg[3]}")

            for slot in dees[cfg[3]].slots_flat:
                if slot.covered:
                    for mod in slot.modules:
                        for sen in mod.sensors:
                            ax.add_patch(
                                sen.getPolygon(active=True, simple=True),
                            )

        plt.gca().add_patch(plt.Circle((v.x+x_shifts[1], v.y+y_shifts[1]), 1,fill='full', facecolor='red', edgecolor='red'))
        ax.set_xlim(v.x-100, v.x+100)
        ax.set_ylim(v.y-100, v.y+100)
        fig.savefig('figures/intersect_face2.pdf')



        fig, ax = plt.subplots(figsize=(8, 8))
        for cfg in configs[2:3]:
            print(f"  - {cfg[3]}")

            for slot in dees[cfg[3]].slots_flat:
                if slot.covered:
                    for mod in slot.modules:
                        for sen in mod.sensors:
                            ax.add_patch(
                                sen.getPolygon(active=True, simple=True),
                            )

        plt.gca().add_patch(plt.Circle((v.x+x_shifts[2], v.y+y_shifts[2]), 1,fill='full', facecolor='red', edgecolor='red'))
        ax.set_xlim(v.x-100, v.x+100)
        ax.set_ylim(v.y-100, v.y+100)
        fig.savefig('figures/intersect_face3.pdf')


        fig, ax = plt.subplots(figsize=(8, 8))

        for cfg in configs[3:4]:
            print(f"  - {cfg[3]}")

            for slot in dees[cfg[3]].slots_flat:
                if slot.covered:
                    for mod in slot.modules:
                        for sen in mod.sensors:
                            ax.add_patch(
                                sen.getPolygon(active=True, simple=True),
                            )

        plt.gca().add_patch(plt.Circle((v.x+x_shifts[3], v.y+y_shifts[3]), 1,fill='full', facecolor='red', edgecolor='red'))
        ax.set_xlim(v.x-100, v.x+100)
        ax.set_ylim(v.y-100, v.y+100)
        fig.savefig('figures/intersect_face4.pdf')



    if args.comparison:
        print("Plotting 1D comparison histograms")

        with open("layouts/attachment_studies.pkl", "rb") as f:
            results = pickle.load(f)

        fig, ax = plt.subplots(1,1,figsize=(10,10))

        results['plain']['hits'][{'phi':sum}].profile('n').plot1d(ax=ax, color='black', label='No attachment', linewidth=3)
        results['baseline']['hits'][{'phi':sum}].profile('n').plot1d(ax=ax, color='red', label='Initial attachment', linewidth=3)
        results['updated_S']['hits'][{'phi':sum}].profile('n').plot1d(ax=ax, color='blue', label='Updated attachment', linewidth=3)
        results['updated_M']['hits'][{'phi':sum}].profile('n').plot1d(ax=ax, color='orange', label='Updated attach + med modules', linewidth=3)
        results['updated_L']['hits'][{'phi':sum}].profile('n').plot1d(ax=ax, color='green', label='Updated attach + large modules', linewidth=3)
        results['updated_L_with_seal']['hits'][{'phi':sum}].profile('n').plot1d(ax=ax, color='magenta', label='Updated attach + LM + seal', linewidth=3)

        ax.legend()
        ax.set_xlabel(r'|$\eta$|')
        ax.set_ylabel(r'<$N_{hits}$>')

        fig.savefig("figures/comparison_nhits_eta.png")

        fig, ax = plt.subplots(1,1,figsize=(10,10))

        results['no_attachment_r336']['hits'][{'eta':sum}].profile('n').plot1d(ax=ax, color='black', label='No attachment', linewidth=3)
        results['realistic_baseline']['hits'][{'eta':sum}].profile('n').plot1d(ax=ax, color='red', label='Initial attachment', linewidth=3)
        results['realistic_update']['hits'][{'eta':sum}].profile('n').plot1d(ax=ax, color='blue', label='Updated attachment', linewidth=3)
        results['realistic_update_medium_module']['hits'][{'eta':sum}].profile('n').plot1d(ax=ax, color='orange', label='Updated attach + med modules', linewidth=3)
        results['realistic_update_large_module']['hits'][{'eta':sum}].profile('n').plot1d(ax=ax, color='green', label='Updated attach + large modules', linewidth=3)
        results['realistic_update_large_module_with_seal']['hits'][{'eta':sum}].profile('n').plot1d(ax=ax, color='magenta', label='Updated attach + LM + seal', linewidth=3)

        ax.legend()
        ax.set_xlabel(r'$\phi$')
        ax.set_ylabel(r'<$N_{hits}$>')

        fig.savefig("figures/comparison_nhits_phi.png")


        fig, ax = plt.subplots(1,1,figsize=(10,10))

        results['no_attachment_r336']['time'][{'phi':sum}].profile('t').plot1d(ax=ax, color='black', label='No attachment', linewidth=3)
        results['realistic_baseline']['time'][{'phi':sum}].profile('t').plot1d(ax=ax, color='red', label='Initial attachment', linewidth=3)
        results['realistic_update']['time'][{'phi':sum}].profile('t').plot1d(ax=ax, color='blue', label='Updated attachment', linewidth=3)
        results['realistic_update_medium_module']['time'][{'phi':sum}].profile('t').plot1d(ax=ax, color='orange', label='Updated attach + med modules', linewidth=3)
        results['realistic_update_large_module']['time'][{'phi':sum}].profile('t').plot1d(ax=ax, color='green', label='Updated attach + large modules', linewidth=3)
        results['realistic_update_large_module_with_seal']['time'][{'phi':sum}].profile('t').plot1d(ax=ax, color='magenta', label='Updated attach + LM + seal', linewidth=3)

        ax.legend()
        ax.set_xlabel(r'|$\eta$|')
        ax.set_ylabel(r'$\sigma_{t}$')

        fig.savefig("figures/comparison_time_res_eta.png")
