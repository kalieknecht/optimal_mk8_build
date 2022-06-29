import numpy as np
import pandas as pd
from itertools import product

class optimal_build:
    def __init__(self,
        drivers,
        bodies,
        tires,
        gliders):
        '''
        Find optimal build for desired stats
        
        Parameters
        ----------
        drivers: pd.DataFrame
            df of driver data
        bodies: pd.DataFrame
            df of body data
        tires: pd.DataFrame
            df of tire data
        gliders: pd.DataFrame
            df of glider data
        Returns
        -------
        optimal_build: cls
            class to hold build optimization
        '''
        self.drivers = drivers
        self.bodies = bodies
        self.tires = tires
        self.gliders = gliders
        self.drivers_clean = remove_duplicates(drivers)
        self.bodies_clean = remove_duplicates(bodies)
        self.tires_clean = remove_duplicates(tires)
        self.gliders_clean = remove_duplicates(gliders)
    
    def find(self,
        stats,
        duplicates=False,
        method='nash'):
        '''
        Find optimal build for desired stats
        
        Parameters
        ----------
        stats: list
            list of str of values desired
        duplicates: bool
            whether to include duplicates (default False)
        method: str
            which method 'nash', 'ks', or 'ideal'
        
        Returns
        -------
        build: list
            optimal build params
        '''
        self.method = method
        # check for valid stats
        possible_stats = ['WG','AC','ON','OF','MT','SL','SW','SA','SG','TL','TW','TA','TG']
        assert set(stats).issubset(set(possible_stats)), "must be in stat list"+str(possible_stats)
        self.stats = stats
        
        # find all possible combinations
        self.combos = self.find_combos(duplicates=duplicates)

        # grab desired stats
        self.stat1 = self.drivers_clean.loc[self.combos[:,0],stats[0]].values + self.bodies_clean.loc[self.combos[:,1],stats[0]].values + self.tires_clean.loc[self.combos[:,2],stats[0]].values + self.gliders_clean.loc[self.combos[:,3],stats[0]].values
        self.stat2 = self.drivers_clean.loc[self.combos[:,0],stats[1]].values + self.bodies_clean.loc[self.combos[:,1],stats[1]].values + self.tires_clean.loc[self.combos[:,2],stats[1]].values + self.gliders_clean.loc[self.combos[:,3],stats[1]].values

        # find pareto fronteier
        self.p_front = pareto_frontier(self.stat1,self.stat2)
        self.p_front_idx = [np.argwhere(p == self.stat1)[0][0] for p in self.p_front[:,0]]

        # find optimal config on pareto front using desired method
        self.p1 = np.array([self.stat1.min(),self.stat2.min()])
        self.p2 = np.array([self.stat1.max(),self.stat2.max()])

        if method=='nash':
            self.opt_idx = nash_method(self.p_front,self.p1)
        elif method=='ideal':
            self.opt_idx = ideal_method(self.p_front,self.p2)
        elif method=='ks':
            self.opt_idx = ks_method(self.p_front,self.p1,self.p2)
        else:
            NotImplementedError('Method must be nash ideal or ks')

        self.build = self.combos[self.p_front_idx[self.opt_idx]]
        
        print('Optimal Configuration for '+stats[0]+' and '+stats[1])
        print(self.build)
        print('Stats: '+stats[0]+' and '+stats[1])
        print(self.p_front[self.opt_idx])
        return self.build


    def find_combos(self, duplicates=False):
        '''
        Find all possible combinations of configs
        
        '''
        if duplicates:
            drivers_list = self.drivers.index.to_list()
            bodies_list = self.bodies.index.to_list()
            tires_list = self.tires.index.to_list()
            gliders_list = self.gliders.index.to_list()
        else:
            drivers_list = self.drivers_clean.index.to_list()
            bodies_list = self.bodies_clean.index.to_list()
            tires_list = self.tires_clean.index.to_list()
            gliders_list = self.gliders_clean.index.to_list()
        combos = np.array([list(product(drivers_list,bodies_list,tires_list,gliders_list))])[0]
        return combos
    
    def find_matching_builds(self):
        # drivers
        drivers_mask = (self.drivers[self.drivers.duplicated()] == self.drivers.loc[self.build[0]]).sum(axis=1)==13
        matching_drivers = self.drivers[self.drivers.duplicated()][drivers_mask].index.to_list()
        # bodies
        bodies_mask = (self.bodies[self.bodies.duplicated()] == self.bodies.loc[self.build[1]]).sum(axis=1)==13
        matching_bodies = self.bodies[self.bodies.duplicated()][bodies_mask].index.to_list()
        # tires
        tires_mask = (self.tires[self.tires.duplicated()] == self.tires.loc[self.build[2]]).sum(axis=1)==13
        matching_tires = self.tires[self.tires.duplicated()][tires_mask].index.to_list()
        # gliders
        matching_mask = (self.gliders[self.gliders.duplicated()] == self.gliders.loc[self.build[3]]).sum(axis=1)==13
        matching_gliders = self.gliders[self.gliders.duplicated()][matching_mask].index.to_list()
        print('Drivers: '+str(matching_drivers))
        print('Bodies: '+str(matching_bodies))
        print('Tires: '+str(matching_tires))
        print('Gliders: '+str(matching_gliders))
        return (matching_drivers, matching_bodies, matching_tires, matching_gliders)

def pareto_frontier(Xs, Ys, maxX = True, maxY = True):
    '''
    Find pareto frontier of X and Y list

    From https://oco-carbon.com/metrics/find-pareto-frontiers-in-python/

    Parameters
    ----------
    Xs: list
        X values
    Ys: list
        Y values
    maxX: bool
        whether to maximize Xs
    maxY: bool
        whether to maximize Ys
    
    Returns
    -------
    p_front: np.array
        pareto frontier of data
    '''
    # Sort the list in either ascending or descending order of X
    myList = sorted([[Xs[i], Ys[i]] for i in range(len(Xs))], reverse=maxX)
    # Start the Pareto frontier with the first value in the sorted list
    p_front = [myList[0]]    
    # Loop through the sorted list
    for pair in myList[1:]:
        if maxY: 
            if pair[1] >= p_front[-1][1]: # Look for higher values of Y…
                p_front.append(pair) # … and add them to the Pareto frontier
        else:
            if pair[1] <= p_front[-1][1]: # Look for lower values of Y…
                p_front.append(pair) # … and add them to the Pareto frontier
    # Turn resulting pairs back into a list of Xs and Ys
    p_frontX = [pair[0] for pair in p_front]
    p_frontY = [pair[1] for pair in p_front]
    p_front = np.c_[p_frontX,p_frontY]
    return p_front

def nash_method(p_front,p1):
    '''
    Nash Bargaining Solution to finding best candidate on Pareto front
    
    Parameters
    ----------
    p_front: np.array
        pareto frontier
    p1: np.array
        disagreement point

    Returns
    -------
    max_idx_nash: int
        index of optimal point on p_front
    '''
    d_ = abs(p1 - p_front)
    ns = d_[:,0] * d_[:,1]
    max_idx_nash = np.argmax(ns)
    return max_idx_nash

def ideal_method(p_front,p2):
    '''
    Min Distance to Ideal method to finding best candidate on Pareto front
    
    Parameters
    ----------
    p_front: np.array
        pareto frontier
    p2: np.array
        ideal point

    Returns
    -------
    min_idx_ideal: int
        index of optimal point on p_front
    '''
    ideal_p_dist = np.linalg.norm(p2 - p_front, axis=1)
    min_idx_ideal = np.argmin(ideal_p_dist)
    return min_idx_ideal

def ks_method(p_front,p1,p2):
    '''
    KS Solution to finding best candidate on Pareto front
    
    Parameters
    ----------
    p_front: np.array
        pareto frontier
    p1: np.array
        disagreement point
    p2: np.array
        ideal point

    Returns
    -------
    min_idx_ks: int
        index of optimal point on p_front
    '''
    dist_to_ks = abs((p2[0]-p1[0])*(p1[1]-p_front[:,1]) - (p1[0] - p_front[:,0])*(p2[1]-p1[1]))/np.linalg.norm(p2-p1)
    min_idx_ks = np.argmin(dist_to_ks)
    return min_idx_ks

def remove_duplicates(df):
    clean_df = df[~df.duplicated()]
    return clean_df
