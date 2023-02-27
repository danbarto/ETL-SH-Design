import matplotlib.pyplot as plt
import math
import copy

from itertools import combinations

import numpy as np
import pandas as pd

from partition import *
from sensors import sensor_current, irradiation, sensor_occupancy

colors = {
    3: 'cyan',
    6: 'magenta',
    7: 'yellow',
    11: 'orange',
    12: 'blue',
    14: 'green'
}

color_list = ['red', 'green', 'blue', 'yellow', 'cyan', 'orange', 'pink', 'gray', 'magenta', 'white', 'brown'] *10

def get_sensors_r_min_max(modules):
    r_min = 10000
    r_max = 0
    for module in modules:
        for sensor in module.sensors:
            for corner in sensor.activeArea:
                r = math.sqrt(corner[0]**2 + corner[1]**2)
                if r < r_min: r_min = r
                if r > r_max: r_max = r
    return r_min, r_max

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

class Sensor(object):
    def __init__(self, height, width, x=0, y=0, deadspace=0.5, color='orange'):
        '''
        Create a sensor object with height (in x) and width (in y). x and y define the position of the center
        '''
        self.height = height
        self.width = width
        self.x = x
        self.y = y
        self.color = color
        self.deadspace = deadspace

        self.setOutline()

    def setOutline(self):
        '''
        set the outline for the polygon, but also the coordinates of the corners in an easily accessible way.
        x1 < x2,
        y1 < y2
        so (x1, y1) is the lower left corner.
        '''
        self.x1 = self.x - self.height/2.
        self.x2 = self.x + self.height/2.
        self.y1 = self.y - self.width/2.
        self.y2 = self.y + self.width/2.

        self.outline = [
            [self.x1, self.y2],
            [self.x2, self.y2],
            [self.x2, self.y1],
            [self.x1, self.y1]
        ]

    def setActiveArea(self):
        self.ax1 = self.x - self.height/2. + self.deadspace
        self.ax2 = self.x + self.height/2. - self.deadspace
        self.ay1 = self.y - self.width/2. + self.deadspace
        self.ay2 = self.y + self.width/2. - self.deadspace

        self.activeArea = [
            [self.ax1, self.ay2],
            [self.ax2, self.ay2],
            [self.ax2, self.ay1],
            [self.ax1, self.ay1]
        ]

    def getActiveArea(self):
        return abs((self.ax2-self.ax1)*(self.ay2-self.ay1))
        
    def move_to(self, x, y):
        self.x = x
        self.y = y
        self.setOutline()
        self.setActiveArea()

    def move_by(self, x, y):
        self.x = self.x + x
        self.y = self.y + y
        self.setOutline()
        self.setActiveArea()

    def getPolygon(self, fill=True, active=False, alpha=0.5, color='gray'):
        '''
        Returns a polygon that can be drawn with matplotlib
        '''
        return plt.Polygon(self.outline if not active else self.activeArea, fill=fill, closed=True, edgecolor='black', facecolor=self.color if not active else color, alpha=alpha )

    def get_current(self):
        '''
        use the center
        '''
        return sensor_current(irradiation(math.sqrt(self.x**2+self.y**2)))

    def get_occupancy(self, per_etroc=False):
        if per_etroc:
            return [
                sensor_occupancy(math.sqrt((self.x-self.height/2.)**2+self.y**2)),
                sensor_occupancy(math.sqrt((self.x+self.height/2.)**2+self.y**2)),
            ]
        else:
            return sensor_occupancy(math.sqrt(self.x**2 + self.y**2))

class ReadoutBoard(Sensor):
    def __init__(self, height, width, x=0, y=0, color='green'):
        '''
        Create a readout board object with height (in x) and width (in y). x and y define the position of the center
        This inherits from Sensor - it's just a rectangle after all
        '''
        self.height = height
        self.width = width
        self.x = x
        self.y = y
        self.color = color
        self.deadspace = 0

        self.setOutline()

class PowerBoard(Sensor):
    def __init__(self, height, width, x=0, y=0, color='red'):
        '''
        Create a power board object with height (in x) and width (in y). x and y define the position of the center.
        This inherits from Sensor - it's just a rectangle after all
        '''
        self.height = height
        self.width = width
        self.x = x
        self.y = y
        self.color = color
        self.deadspace = 0

        self.setOutline()

class VTRX(Sensor):
    def __init__(self, rb, color='orange', cm=False):
        scaler = 10 if cm else 1
        self.height = 20/scaler
        self.width = 10/scaler
        self.x = rb.x2-32.5/scaler  # 49 is the old value
        self.y = rb.y + 6/scaler  # 6 is new
        self.color = color
        self.deadspace = 0

        self.setOutline()


class Module(object):
    def __init__(self, height, width, x=0, y=0, n_sensor_x=1, n_sensor_y=2, sensor_distance_y=22.5, sensor_distance_x=42.6):
        '''
        nSensor can be an even number (or 1).
        Sensor_distance is a measure from center of sensors.
        Symmetry is assumed.
        '''

        self.height = height
        self.width = width
        self.x = x
        self.y = y
        self.n_sensor_x = n_sensor_x
        self.n_sensor_y = n_sensor_y
        self.sensor_distance_x = 0 if n_sensor_x == 1 else sensor_distance_x 
        self.sensor_distance_y = 0 if n_sensor_y == 1 else sensor_distance_y
        self.sensors = []

        self.color = None

        self.setOutline()

    def r(self):
        self._r = math.sqrt(self.x**2 + self.y**2)
        return self.r

    def setOutline(self):
        '''
        set the outline for the polygon, but also the coordinates of the corners in an easily accessible way.
        x1 < x2,
        y1 < y2
        so (x1, y1) is the lower left corner.
        '''
        self.x1 = self.x - self.height/2.
        self.x2 = self.x + self.height/2.
        self.y1 = self.y - self.width/2.
        self.y2 = self.y + self.width/2.

        self.outline = [
            [self.x1, self.y2],
            [self.x2, self.y2],
            [self.x2, self.y1],
            [self.x1, self.y1]
        ]

        self.getActiveCorners()

    def getPolygon(self, edgecolor='black', linewidth=2, fill=False):
        '''
        Returns a polygon that can be drawn with matplotlib
        '''
        return plt.Polygon(self.outline, fill=fill, closed=True, edgecolor=edgecolor, linewidth=linewidth, facecolor=self.color)

    def populate(self, sensor):
        for ix in range(self.n_sensor_x):
            for iy in range(self.n_sensor_y):
                s_temp = copy.deepcopy(sensor)
                s_temp.move_to( (2*ix-1)*self.sensor_distance_x/2 + self.x, (2*iy-1)*self.sensor_distance_y/2 + self.y )
                self.sensors.append(s_temp)

    def move_to(self, x, y):
        self.x = x
        self.y = y
        self.setOutline()
        for s in self.sensors:
            s.move_to(s.x+x,s.y+y)

    def move_by(self, x, y):
        self.x = self.x + x
        self.y = self.y + y
        self.setOutline()
        for s in self.sensors:
            s.move_by(x, y)

    def getActiveArea(self):
        return sum( [ s.getActiveArea() for s in self.sensors ] )
    
    def getActiveCorners(self):
        '''
        this gets numpy arrays of the corners.
        in the end, we can check if a particle with (x,y) intersects with the active area of a sensor by doing
        ((m.vax1 < x) & (x < m.vax2) & (m.vay1 < y) & (y < m.vay2)).any()
        '''
        self.vax1 = [ s.ax1 for s in self.sensors ]
        self.vax2 = [ s.ax2 for s in self.sensors ]
        self.vay1 = [ s.ay1 for s in self.sensors ]
        self.vay2 = [ s.ay2 for s in self.sensors ]

    def get_current(self):
        self.current = 0
        for s in self.sensors:
            self.current += s.get_current()
        return self.current

    def get_occupancy(self):
        self.occupancy = 0
        for s in self.sensors:
            self.occupancy += s.get_occupancy()
        self.occupancy = self.occupancy / len(self.sensors)
        return self.occupancy

class SuperModule(object):
    def __init__(self, module, powerboard, readoutboard, x=0, y=0, n_modules=3, module_gap=0.5, orientation='above', color='b', cm=False):
        '''
        This consists of N modules together with a readout board and a power board.
        '''

        self.height = module.height * n_modules + module_gap * (n_modules-1)
        self.width = module.width+powerboard.width
        self.x = x
        self.y = y
        self.orientation = orientation
        self.module_gap = module_gap
        self.n_modules = n_modules
        self.color = color

        self.setOutline()

        # make copies of the components
        self.PB = copy.deepcopy(powerboard)
        self.RB = copy.deepcopy(readoutboard)
        # also keep the originals
        self._PB = copy.deepcopy(powerboard)
        self._RB = copy.deepcopy(readoutboard)
        self._module = copy.deepcopy(module) # not really used, but can be useful


        # place the modules
        self.modules = []
        for im in range(n_modules):
            m_temp = copy.deepcopy(module)
            m_temp.move_by( ( -(n_modules-1)/2 + im )*(module.height+module_gap),  (-1)*self.PB.width/2 if orientation=='above' else self.PB.width/2 )
            m_temp.i = im
            self.modules.append(m_temp)

        # update the dimensions of the RB and PB
        self.PB.height = self.height
        self.PB.setOutline()
        self.RB.height = self.height
        self.RB.setOutline()

        # move the components in place
        #print ("Not moving anything")
        self.PB.move_by(0, self.RB.width/2 if orientation=='above' else (-1)*self.RB.width/2)
        self.RB.move_by(0, (-1)*self.PB.width/2 if orientation=='above' else self.PB.width/2)

        self.VTRX = VTRX(self.RB, cm=cm)
    
    @classmethod
    def fromSuperModule(cls, supermodule, x=0, y=0, n_modules=3, module_gap=0.5, orientation='above', color='b', cm=False):
        
        return cls(supermodule._module, supermodule._PB, supermodule._RB, x=x, y=y, n_modules=n_modules, module_gap=module_gap, orientation=orientation, color=color, cm=cm)

    def setOutline(self):
        '''
        set the outline for the polygon, but also the coordinates of the corners in an easily accessible way.
        x1 < x2,
        y1 < y2
        so (x1, y1) is the lower left corner.
        '''
        self.x1 = self.x - self.height/2.
        self.x2 = self.x + self.height/2.
        self.y1 = self.y - self.width/2.
        self.y2 = self.y + self.width/2.

        self.outline = [
            [self.x1, self.y2],
            [self.x2, self.y2],
            [self.x2, self.y1],
            [self.x1, self.y1]
        ]

    def move_by(self, x, y):
        self.x = self.x + x
        self.y = self.y + y
        self.setOutline()
        self.RB.move_by(x,y)
        self.PB.move_by(x,y)
        self.VTRX.move_by(x,y)
        for s in self.modules:
            s.move_by(x, y)
        

    def getPolygon(self, alpha=0.5, fill=True):
        '''
        Returns a polygon that can be drawn with matplotlib
        '''
        return plt.Polygon(self.outline, fill=fill, closed=True, linewidth=3, edgecolor='black', facecolor=self.color, alpha=alpha)

    def getActiveArea(self):
        '''
        This will be important
        '''
        return sum( [ sum( [ s.getActiveArea() for s in m.sensors ] ) for m in self.modules ] )

    def centerModule(self):
        '''
        Move the whole thing so that the modules are centered around y
        '''
        self.move_by(-self.x1, self.width/2-self.RB.width/2 if self.orientation=='above' else self.width/2-self.PB.width-self.RB.width/2)

    def centerPB(self):
        '''
        Move the whole thing so that the modules are centered around y
        '''
        self.move_by(-self.x1, -self.width/2+self.PB.width/2 if self.orientation=='above' else self.width/2-self.PB.width/2)

    def find_BV_config(self, sensor_fun, verbose=False, min_split=1):
        combs = list(combinations(self.modules, 2))
        combs += [(self.modules[i], self.modules[i]) for i in range(self.n_modules)]
        working_combs = []
        problematic_modules = []
        for comb in combs:
            mi, ma = get_sensors_r_min_max(comb)
            if mi>sensor_fun(ma):
                working_combs.append((comb[0].i, comb[1].i))
            elif mi<sensor_fun(ma) and comb[0] == comb[1]:
                #if verbose:
                problematic_modules.append(comb[0].i)
                if verbose:
                    print ("Need more than 1 BV per module, but will proceed anyway!")
                working_combs.append((comb[0].i, comb[1].i))

        if len(problematic_modules) > 0 and verbose:
            print ("WARNING: Module found that needs more than 1 BV!")
            #print (problematic_modules)

        if verbose:
            print (f"RB {self.n_modules} sitting at {self.x}, {self.y}")
            print (f"I found {len(working_combs)} viable configurations.")
        if len(working_combs) <2:
            print ("!!! Warning !!!")

        new_cfgs = [[c] for c in working_combs if c[1]==self.n_modules-1]

        for it in range(self.n_modules):
            cfgs = copy.deepcopy(new_cfgs)
            new_cfgs = []
            for l in cfgs:
                for c in working_combs:
                    if c[1] == l[-1][0]-1:
                        new_cfgs.append(l + [c])
                if l[-1][0] == 0:
                    new_cfgs.append(l)

        min_length = min([len(x) for x in new_cfgs])
        min_length = max(min_split, min_length) if min_split > 1 else min_length
        avg_bvs = self.n_modules/min_length


        # picking the best config based on averaging the BV levels
        best_cfg = []
        best_avg = 10.
        for cfg in cfgs:
            if len(cfg) == min_length:
                dist = 0
                for pair in cfg:
                    dist += abs(pair[1]+1 - pair[0] - avg_bvs)
                dist = dist/min_length
                if dist < best_avg:
                    best_avg = dist
                    best_cfg = cfg

        if verbose:
            print ("Best cfg:", best_cfg)

        self.currents = []
        for color_index, (mod_0, mod_1) in enumerate(best_cfg):
            tmp = 0
            for i in range(mod_0, mod_1+1):
                self.modules[i].problematic=True if i in problematic_modules else False
                tmp += self.modules[i].get_current()
                for s in self.modules[i].sensors:
                    s.color = color_list[color_index]
            self.currents.append(tmp)
        self.BV_cfg = best_cfg
        self.BV_lines = len(best_cfg) + 1  # includes return
        return best_cfg

    def get_occupancy(self, scale=1):
        self.occupancy = 0
        for m in self.modules:
            self.occupancy += m.get_occupancy()
        self.occupancy = scale*self.occupancy / len(self.modules)
        return self.occupancy

    def get_lpgbt_mult(self):
        pass
        #if self.occupancy >
    
class Dee(object):
    def __init__(self, r_inner, r_outer, z=0, color='red'):
        self.r_inner = r_inner
        self.r_outer = r_outer
        self.area    = (r_outer**2 - r_inner**2)*np.pi/2
        self.z       = z
        self.color   = color
        self.supermodules = []

    def populate(self, supermodule, edge_x=6, shift_x=0, shift_y=0, flavors=[3,6,7], center_RB=False, center_PB=False):
        '''
        takes a supermodule, puts them wherever there's space.
        shift_y = 0 will make the _modules_ symmetric around the y-axis.
        shift_y = module.width/2 would then be the second Dee, for example.
        '''
        smallest = SuperModule.fromSuperModule(supermodule, n_modules=1, module_gap=supermodule.module_gap, orientation=supermodule.orientation)
        if center_RB:
            smallest.centerModule()
        if center_PB:
            smallest.centerPB()
        smallest.move_by(edge_x,0)

        self.n_rows    = int(2*self.r_outer/smallest.width)+2
        self.n_columns = int(self.r_outer/(smallest.height+smallest.module_gap))+2

        #self.slot_matrix = np.zeros((self.n_rows,self.n_columns))
        self.slot_matrix = [[ 0 for x in range(self.n_columns)] for y in range(self.n_rows)]

        self.slots = [ []  for y in range(self.n_rows)]
        
        for row in range(self.n_rows):
            for column in range(self.n_columns):
                tmp = copy.deepcopy(smallest) #SuperModule.fromSuperModule(smallest, x=smallest.x, y=smallest.y, n_modules=1, module_gap=smallest.module_gap, orientation=smallest.orientation)
                tmp.move_by(column*(smallest.height+smallest.module_gap), (math.floor(self.n_rows/2)-row)*smallest.width )
                if (tmp.x1**2 + tmp.y1**2)>self.r_inner**2 and \
                   (tmp.x2**2 + tmp.y2**2)>self.r_inner**2 and \
                   (tmp.x1**2 + tmp.y2**2)>self.r_inner**2 and \
                   (tmp.x2**2 + tmp.y1**2)>self.r_inner**2 and \
                   (tmp.x1**2 + tmp.y1**2)<self.r_outer**2 and \
                   (tmp.x2**2 + tmp.y2**2)<self.r_outer**2 and \
                   (tmp.x1**2 + tmp.y2**2)<self.r_outer**2 and \
                   (tmp.x2**2 + tmp.y1**2)<self.r_outer**2:

                    ## NOW apply the shift, otherwise we mess up the "on-disk" requirement. This can cause weirdness in the plots.
                    tmp.move_by(shift_x, shift_y)

                    self.slots[self.n_rows-row-1].append(tmp)

                    self.slot_matrix[self.n_rows-row-1][column] = 1
                else:
                    self.slot_matrix[self.n_rows-row-1][column] = 0

        # now let's go through the matrix again and see which slots we can actually populate
        self.module_matrix = []
        self.slots_flat = []
        for i, row in enumerate(self.slot_matrix):
            # maximum length
            length = sum(row)

            # use the partition function to get the composition of RB flavors
            partition = getPartition(length, flavors=flavors)
            covered = sum(partition)

            x_shift = 0
            for k, n_mod in enumerate(partition):
                tmp = copy.deepcopy(SuperModule.fromSuperModule(supermodule, n_modules=n_mod, module_gap=supermodule.module_gap, orientation=supermodule.orientation, color=colors[n_mod]))
                tmp.move_by(self.slots[i][0].x1-tmp.x1 + x_shift, self.slots[i][0].y1-tmp.y1)
                x_shift += tmp.height + tmp.module_gap
                self.supermodules.append(tmp)
                #break
                #self.slots[i][0].x1

            for j in range(length):
                self.slots[i][j].covered = True if j<covered else False

            self.slots_flat += self.slots[i] if length>0 else []

            if length == covered:
                self.module_matrix.append(row)
            else:
                self.module_matrix.append( [1]*covered + [-1]*(length-covered) + [0]*(len(row)-length) )

        self.getAllCorners()

        return

    def fromCenters(self, centers, sensor, cm=False):
        '''
        this is useful for old layouts / tilings
        
        '''
        scaler = 10 if cm else 1

        # loop over centers
        self.sensors = []
        for x,y in centers:
            tmp = copy.deepcopy(sensor)
            tmp.move_to(x/scaler,y/scaler)
            self.sensors.append(tmp)

        # manually get the corners
        self.vax1 = []
        self.vax2 = []
        self.vay1 = []
        self.vay2 = []

        for sen in self.sensors:
            self.vax1 += [sen.ax1]
            self.vax2 += [sen.ax2]
            self.vay1 += [sen.ay1]
            self.vay2 += [sen.ay2]

        self.vax1 = np.array(self.vax1)
        self.vax2 = np.array(self.vax2)
        self.vay1 = np.array(self.vay1)
        self.vay2 = np.array(self.vay2)

    def attach_supermodules(self, supermodules):
        self.supermodules = supermodules


    def getAllCorners(self):
        self.vax1 = []
        self.vax2 = []
        self.vay1 = []
        self.vay2 = []

        for slot in self.slots_flat:
            if slot.covered:
                for mod in slot.modules:
                    for sen in mod.sensors:
                        self.vax1 += [sen.ax1]
                        self.vax2 += [sen.ax2]
                        self.vay1 += [sen.ay1]
                        self.vay2 += [sen.ay2]

        self.vax1 = np.array(self.vax1)
        self.vax2 = np.array(self.vax2)
        self.vay1 = np.array(self.vay1)
        self.vay2 = np.array(self.vay2)

    def intersect(self, x, y):
        '''
        ((m.vax1 < x) & (x < m.vax2) & (m.vay1 < y) & (y < m.vay2)).any()
        '''
        return ((self.vax1 < x) & (x < self.vax2) & (self.vay1 < y) & (y < self.vay2)).any()


if __name__ == "__main__":

    # run an example
    ## current TAMALES
    s = Sensor(42.5, 22)
    m = Module(43.10, 56.50, n_sensor_x=1, n_sensor_y=2, sensor_distance_y=22.5, sensor_distance_x=42.5+0.1)
    
    ## the other possibility
    #sensor_x = (42.5/2 + 0.5)
    #s = Sensor(sensor_x, 22)
    #m = Module(44.20, 56.50, n_sensor_x=2, n_sensor_y=2, sensor_distance_y=22.5, sensor_distance_x=sensor_x+0.1)

    m.populate(s)
    
    rb = ReadoutBoard(10,56.5, color='green')
    pb = ReadoutBoard(10,29.5, color='red')
    
    SM = SuperModule(m, pb, rb, n_modules=3, orientation='above')
    

    D = Dee(315, 1185)
    D.populate(SM, flavors=[3,6,7], center_RB=True, edge_x = 6, shift_x = 2, shift_y = 2)


    #for row in D.slot_matrix:
    #    print ((' '.join([ str(x) for x in row])).replace('1','X').replace('0', '.'))

    for row in D.module_matrix:
        print ((' '.join([ str(x) for x in row])).replace('-1','O').replace('0', '.').replace('1', 'X'))


    covered_area = sum( [ slot.getActiveArea() for slot in D.slots_flat ] )
    available_slots = sum( [ sum(row) for row in D.slot_matrix ])
    filled_slots = sum( [ sum([ x for x in row if x==1]) for row in D.module_matrix ])

    print ("Number of available slots:", available_slots )
    print ("Number of used slots (= number of modules):", filled_slots )
    print ("The maximum fill factor is:", round(covered_area/D.area, 3))


    print("Testing if a particle at 10,10 intersects any of the sensors (it shouldn't):", D.intersect(10,10))
    print("Testing if a particle at 10,500 intersects any of the sensors:", D.intersect(10,500))
