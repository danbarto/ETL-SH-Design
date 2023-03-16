#!/usr/bin/env python3
'''
ETL Geometry with shapely, WIP
'''
import time
import random
import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry import Polygon, Point
from shapely.plotting import plot_polygon
from shapely.affinity import rotate, translate
from shapely.ops import unary_union
from ETL import *
from sensors import *

etl_outer = 1185
etl_inner1 = 336
etl_inner2 = 300
feedthrough = 1120

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

def rad(deg):
    return deg * np.pi / 180

def make_inner_attachment(angle=0, height=84, width=88.5, taper=5.4):
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
    tmp = translate(tmp, np.sin(angle_rad)*(etl_inner1+height), np.cos(angle_rad)*(etl_inner1+height))

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

def make_inner():
    outer = Point(0, 0).buffer(etl_inner1)
    inner = Point(0, 0).buffer(etl_inner2)

    exter = list(zip(outer.exterior.xy[0], outer.exterior.xy[1]))
    inter = list(zip(inner.exterior.xy[0], inner.exterior.xy[1]))[::-1]

    return Polygon(exter, [inter])

def make_outer():
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

if __name__ == '__main__':

    nose = Point(0, 0).buffer(etl_inner2)
    inner = make_inner()
    outer = make_outer()

    attachments = []

    # baseline attachments from natalia
    attachments += [make_inner_attachment(angle=x) for x in [30,90,150,-30,-90,-150]]
    attachments += [make_outer_attachment(angle=x) for x in [2.5, 68, 112, 177.5, -2.5, -68, -112, -177.5]]

    ## "optimized placement" 1
    #attachments += [make_inner_attachment(angle=x) for x in [15,90,165,-15,-90,-165]]
    #attachments += [make_outer_attachment(angle=x) for x in [2.5, 68, 112, 177.5, -2.5, -68, -112, -177.5]]

    ## "optimized placement" 2
    #attachments += [make_inner_attachment(angle=x) for x in [40,90,140,-40,-90,-140]]
    #attachments += [make_outer_attachment(angle=x) for x in [2.5, 68, 112, 177.5, -2.5, -68, -112, -177.5]]

    ## "optimized placement" 3  --> 5292 modules. small improvement
    #attachments += [make_inner_attachment(angle=x) for x in [30,90,150,-30,-90,-150]]
    #attachments += [make_outer_attachment(angle=x) for x in [2.5, 70, 110, 177.5, -2.5, -70, -110, -177.5]]

    ## "optimized placement" 3  --> 5292 modules. small improvement
    #attachments += [make_inner_attachment(angle=x, height=50) for x in [15,25,85,95,155,165,-30,-90,-150]]
    #attachments += [make_outer_attachment(angle=x) for x in [2.5, 68, 112, 177.5, -2.5, -68, -112, -177.5]]

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
    #dee_geo += [Point(500,500).buffer(50)]
    #dee_geo += [Point(550,450).buffer(50)]
    for mod in modules:
        collision = overlaps(mod.getPolygon(), dee_geo)
        print (collision, mod.getPolygon())
        plot_polygon(mod.getPolygon(), ax=ax, add_points=False, color='green' if not collision else 'red')

    fig.savefig('test2.png')
    # realistic layouts:

    sensor_x = (42.5/2 + 0.5)
    s = Sensor(sensor_x, 22)
    #m = Module(44.20, 56.50, n_sensor_x=2, n_sensor_y=2, sensor_distance_y=22.5, sensor_distance_x=sensor_x+0.1)
    m = Module(43.10, 56.50, n_sensor_x=2, n_sensor_y=2, sensor_distance_y=22.5, sensor_distance_x=sensor_x+0.1)

    m.populate(s)

    rb = ReadoutBoard(10,56.5, color='green')
    pb = ReadoutBoard(10,29.5, color='red')

    # make the configurations
    print ("Populating DEEs:")
    configs = [
        ('above', True, False, 'face1'),
        #('above', False, True, 'face2'),
        #('below', True, False, 'face3'),
        #('below', False, True, 'face4')
    ]

    dees = {}
    for cfg in configs:
        print(f"- {cfg[3]}")

        SM = SuperModule(m, pb, rb, n_modules=3, orientation=cfg[0])
        dees[cfg[3]] = Dee(
            336,  # 315, # old number is 315
            1185,
        )
        dees[cfg[3]].add_geometries(dee_geo)

        dees[cfg[3]].populate(
            SM,
            center_RB=cfg[1],
            center_PB=cfg[2],
            edge_x = 6,
        )
    print("Done. Plotting now.")

    total_modules = 0
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
                    total_modules += 1
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

        for sm in dees['face1'].supermodules:
            plot_polygon(
                sm.getPolygon(),
                ax=ax,
                add_points=False,
                color = sm.color,
            )

    print(f"This configuration results in {total_modules} for 1/4th of ETL.")
    fig.savefig('test8.pdf')


    run_bias = True
    if run_bias:
        modules = []
        for sm in dees['face1'].supermodules:
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
        current = 0
        new_group = False
        for m in hpk:
            rmin, rmax = get_sensors_r_min_max([m])
            current += m.get_current()
            if current > 20:
                new_group = True
            if first:
                rmin_for_real = hpk_split4_10fc(rmax)  # being optimistic here with 15fC, because no one knows anything anyway
                groupings.append([])
                first = False
            if rmin > rmin_for_real and not new_group:
                #print (rmin, rmin_for_real)
                groupings[-1].append(m)
            else:
                new_group = False
                current = m.get_current()
                rmin_for_real = hpk_split4_10fc(rmax)  # being optimistic here with 15fC, because no one knows anything anyway
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
                rmin_for_real = fbk_w13_10fc(rmax)  # being optimistic here with 15fC, because no one knows anything anyway
                groupings.append([])
                first = False
            if rmin > rmin_for_real and not new_group:
                #print (rmin, rmin_for_real)
                groupings[-1].append(m)
            else:
                new_group = False
                current = m.get_current()
                rmin_for_real = fbk_w13_10fc(rmax)  # being optimistic here with 15fC, because no one knows anything anyway
                groupings.append([m])

        for group in groupings:
            current = 0
            for m in group:
                current += m.get_current()
            if current > 20:
                print("Found too large current in one group")


    run_acceptance = False
    if run_acceptance:
        starttime = time.time()

        eta_max = 2.950
        eta_min = 1.659
        eta_range = eta_max-eta_min

        nEvents = int(1e5)

        eta = np.random.rand(nEvents)*eta_range + eta_min
        phi = np.random.rand(nEvents)*np.pi - np.pi/2

        vec = three_vector.fromEtaPhi(eta, phi, np.ones(nEvents)*3000)

        vec_list = []
        for x,y in zip(vec.x, vec.y):
            vec_list.append(three_vector(x,y,3000))

        hits = []
        nHits = []

        delta_z = 20.5 # approximated distance between similar layers (e.g. front disk1 - front disk 2)

        # We also want to get the number of hits.
        for v in vec_list:

            n = 0
            iLayer = 0

            for layer in ['face1', 'face2', 'face3', 'face4']:
                x_shift = 1000*(z[iLayer]-z_ref)*np.tan(v.theta)*np.cos(v.phi)
                y_shift = 1000*(z[iLayer]-z_ref)*np.tan(v.theta)*np.sin(v.phi)
                x,y = ((v.x + x_shift), (v.y + y_shift))

                if dees[layer].intersect(x, y):
                    n += 1
                iLayer += 1

            if n>0: hits.append(v)
            nHits.append(n)



        endtime = time.time()

        print (endtime-starttime)
