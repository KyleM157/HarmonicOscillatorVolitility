# -*- coding: utf-8 -*-
"""
Created on Tue Dec 22 16:06:18 2020
"""
#Startup Code Begin
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import t as T
from scipy.special import hermite
from scipy.constants import pi,hbar
from scipy.optimize import curve_fit
from scipy.signal import find_peaks
from math import factorial
from arch import arch_model
#Startup Code End

class HarmonicOscillator(object):
    def __init__(self,params,nlevels):
        self.params = params
        self.minima,self.cdf_params = self.get_cdfs(nlevels)
        

    @classmethod
    def create_ho(cls,lr,nlevels,kf=None):
        '''
        :type lr: ndarray of type float
        :param lr: log returns for the asset

        :type nlevels: int
        :param nlevels: number of levels to initialize Harmonic Oscillator.
        '''
        garch_model = arch_model(
            lr*1000,
            vol='GARCH',
            p=1,
            q=1,
            dist='StudentsT'
            )

        v = garch_model.fit(update_freq=5).conditional_volatility
        
        #Scale Volatility to Standard Deviation of Log Returns
        v = (v/v.mean())*lr.std()

        #Get Energy Levels
        v2 = v[abs(v-v.mean())<3*v.std()]
        hw = (v2.max()-v2.min())/nlevels
        
        #Set v2.min() to .5*hw and get Energy Level Labels
        vv = v-(v.min()-.5*hw)
        levels = np.zeros(shape=len(vv),dtype=np.int64)
        for i in range(1,nlevels):
            if i==nlevels-1:
                idx = np.where(vv>=vv.min()+hw*(nlevels-1))[0]
                levels[idx] = nlevels-1
            else:
                lo = vv.min()+hw*i
                hi = lo+hw
                idx = np.where((vv>=lo)&(vv<hi))[0]
                levels[idx] = i
                
                
        lvl_values = np.empty(shape=len(levels),dtype=float)
        for i in range(5):
            idx = np.where(levels==i)[0]
            lvl_values[idx] = vv.min()+hw*i
                    
        
        dv = np.diff(vv)
        labels = lvl_values[1:]
        
        #Get Positive Values from dv
        p_idx = np.where(dv>0)[0]
        pdv,plabels = dv[p_idx],labels[p_idx]
        pdv = np.concatenate((-1*pdv,pdv))
        plabels = np.concatenate((plabels.copy(),plabels))
        
        #Get kf
        if kf==None:
            def fx(x,k): return(.5*k*x**2)
            kf = curve_fit(fx,pdv,plabels)[0][0]
        else:
            pass
        
        print("Force Constant = {}".format(kf))
        print("***If IndexError Occurs, Increase Value of kf or Decrease nlevels***")
        
        #Calculate dropout at each level; x = sqrt(2y/kf)
        dropout = []
        def get_x(y): return np.sqrt((2*y)/kf)
        for i in range(nlevels):
            y = hw*i + vv.min()
            x = get_x(y)
            temp = (dv[(dv<0)&(labels==y)]).min()
            drop = abs((1-abs(temp/x))-0.5)
            dropout.append(drop)
            
        dropout = sum(dropout)/len(dropout)
        dropout = np.ceil(dropout*100)/100.
            
        #Get constants
        w = hw/hbar
        m = kf/w**2
        alpha = np.sqrt(hbar/(m*w))
        
        xmin,xmax = dv.min()-dv.std(),dv.max()+dv.std()
        
        params = dict(lr=lr,vol=vv,hw=hw,kf=kf,m=m,w=w,a=alpha,nlevels=nlevels,xmin=xmin,xmax=xmax,dropout=dropout)

        ho = cls(params,nlevels)
        return(ho)

    def wavefunction(self,nu,y_values):
        '''
        :type nu: int
        :param nu: Energy Level

        :type y_values: array of type float
        :param y_values: defined as x/alpha. Values used by the
        gaussian function and hermite polynomials.
        '''
        #Unpack Constants
        w = self.params['w']
        m = self.params['m']

        #Define Normalization Constant
        def Nv(nu): return ((m*w)/(pi*hbar))**(1/4) * 1/(np.sqrt(2**nu * factorial(nu)))

        #Define Wavefunctions
        def Psi(nu,y): return Nv(nu)*hermite(nu)(y)*np.exp(-y**2/2)

        return(Psi(nu,y_values))

    def get_cdfs(self,nlevels=5):
        '''
        :type nlevels: int
        :param nlevels: number of energy levels to create for the Harmonic Oscillator.

        Function Creates the Class Paramaters "minima" and "cdf_params". Allows
        for the sampling of the distribution at a given energy level.
        '''
        #Define Logit Functions
        def lfit1(x,A,k,x0): return A/(1+np.exp(-k*(x-x0))) #if nu==0
        def lfit2(x,A,k,x0,b): return A/(1+np.exp(-k*(x-x0))) + b #if nu!=0

        x = np.linspace(self.params['xmin'],self.params['xmax'],10000)
        x = x/self.params['a']
        
        minima = dict()
        cdf_params = dict()
        for i in range(nlevels):

            #Get Wavefunction of x
            psi = self.wavefunction(i,x)
            #psi = Psi(i,x) #For debugging
            
            #Get CDF Fits
            if i==0:
                #Continuous distribution; non parametric equation
                pdf = psi**2
                cdf = pdf.cumsum()/pdf.sum()

                #estimate fitting paramaters
                _A = cdf.max()-cdf.min()
                _k = 1.0
                _x0 = cdf.mean()

                cdf_fit = curve_fit(lfit1,x,cdf,p0=[_A,_k,_x0])[0]
                minima['0'] = None
                cdf_params['0'] = cdf_fit

            else:
                #Non-continuous distributions; Requires parametric equations.
                #Need to find peaks and get minimums between each.
                #Number of peaks = nu+1
                n_peaks = i+1
                pdf = psi**2
                peaks = find_peaks(pdf)[0]
                n_minima = n_peaks-1
                
                local_minima = []
                for j in range(n_minima):
                    p1 = peaks[j]
                    p2 = peaks[j+1]

                    temp = pdf[p1:p2]
                    local_minima.append(temp.argmin()+p1)
                    
                full_cdf = pdf.cumsum()/pdf.sum()
                cdf_fit_list = []
                for j in range(n_peaks):
                    if j==0:
                        idx = local_minima[0]
                        cdf = full_cdf[:idx]
                        xr = x[:idx+100]
                        cdf = np.hstack((cdf,np.full(100,cdf.max())))
                        _x0 = cdf.mean()
                        _A = cdf.max()-cdf.min()

                        cdf_fit = curve_fit(lfit1,xr,cdf,p0=[_A,_k,_x0])[0]
                        cdf_fit_list.append(cdf_fit)

                    elif j==n_peaks-1:
                        idx = local_minima[-1]
                        cdf = full_cdf[idx:]
                        xr = x[idx:]
                        _A = cdf.max()-cdf.min()
                        _k = 1.0
                        _x0 = cdf.mean()
                        _b = cdf.min()

                       
                        cdf_fit = curve_fit(lfit2,xr,cdf,p0=[_A,_k,_x0,_b])[0]
                        cdf_fit_list.append(cdf_fit)


                    else:
                        idx1 = local_minima[j-1]
                        idx2 = local_minima[j]

                        cdf = full_cdf[idx1:idx2]
                        xr = x[idx1:idx2]

                        _A = cdf.max()-cdf.min()
                        _k = 1.0
                        _x0 = cdf.mean()
                        _b = cdf.min()

                        
                        cdf_fit = curve_fit(lfit2,xr,cdf,p0=[_A,_k,_x0,_b])[0]
                        cdf_fit_list.append(cdf_fit)


                minima[str(i)] = local_minima
                cdf_params[str(i)] = cdf_fit_list

        return(minima,cdf_params)
    
    
    def pdf(self,nu):
        '''
        :type nu: int
        :param nu: energy level

        Function returns the probability density function at the given
        energy level.
        '''
        lo,hi = self.params['xmin'],self.params['xmax']
        x = np.linspace(lo,hi,10000)
        xr = x/self.params['a']

        psi = self.wavefunction(nu,xr)
        density = psi**2
        return(density)
    

    def cdf(self,nu):
        '''
        :type nu: int
        :param nu: energy level

        Function Returns the Cumulative Density Function for the energy level.
        '''
        density = self.pdf(nu)
        c = density.cumsum()/density.sum()
        return(c)


    def rvs(self,nu,size):
        '''
        :type nu: int
        :param nu: energy level to sample

        :type size: int or tuple
        :param size: size of the random sample to output

        Function returns a sample from the PPT given an energy level.
        '''
        #Define Quantile (Percent Point) Functions
        def ppt_fit1(y,A,k,x0): return np.log(abs((y*np.exp(k*x0))/(A-y)))/k
        def ppt_fit2(y,A,k,x0,b): return np.log(abs(((y-b)*np.exp(k*x0))/(A+b-y)))/k
                
        if isinstance(size,tuple):
            size2 = size[0]*size[1]
        else:
            size2 = size
        x = np.random.uniform(0,1,size=size2)
        if nu==0:
            sample = ppt_fit1(x,*self.cdf_params['0'])
        else:
            local_minima = self.minima[str(nu)]
            cdf_fits = self.cdf_params[str(nu)]
            cdf = self.cdf(nu)
            sample = np.empty(shape=len(x))
            for i in range(len(local_minima)+1):
                if i==0:
                    hi = cdf[local_minima[0]]
                    idx = np.where(x<hi)[0]
                    sample[idx] = ppt_fit1(x[idx],*cdf_fits[0])
                elif i==len(local_minima):
                    lo = cdf[local_minima[-1]]
                    idx = np.where(x>=lo)[0]
                    sample[idx] = ppt_fit2(x[idx],*cdf_fits[-1])
                else:
                    lo = cdf[local_minima[i-1]]
                    hi = cdf[local_minima[i]]
                    idx = np.where((x>=lo)&(x<hi))[0]
                    sample[idx] = ppt_fit2(x[idx],*cdf_fits[i])
                
        if isinstance(size,tuple):
            sample = sample.reshape(size[0],size[1])
            x = x.reshape(size[0],size[1])
        else:
            pass

        return(sample)


    def sample_posterior(self,vol,npaths=1000):
        '''
        '''
        tsteps = len(vol)
        n_paths = npaths+int(npaths*self.params['dropout'])
        nlevels = self.params['nlevels']
        hw = self.params['hw']
        vmin = .5*hw
        levels = np.zeros(shape=len(vol),dtype=int)
        for i in range(1,nlevels):
            if i==nlevels-1:
                idx = np.where(vol>=vmin.min()+hw*(nlevels-1))[0]
                levels[idx] = nlevels-1
            else:
                lo = vmin+hw*i
                hi = lo+hw
                idx = np.where((vol>=lo)&(vol<hi))[0]
                levels[idx] = i
                
        
        prd = np.empty(shape=(npaths,len(vol)),dtype=float)
        for i in range(tsteps):
            lvl = levels[i]
            rdraw = np.sort(self.rvs(lvl,n_paths)*self.params['a'])
            rdraw = rdraw[-npaths:]
            prd[:,i] = rdraw
    
        qho_vol = vol+prd
        qho_vol[qho_vol<0.]=0.
        
        return(qho_vol)
        
                



def load_data():
    fname = r'E:/Investing_Data/mcmc_example/{}.npy'
    log_returns = np.load(fname.format('lreturns'))
    garch_vol = np.load(fname.format('garch'))
    mc_vol = np.load(fname.format("trace"))
    data = pd.read_csv('E:/Investing/historical_data/SPY.csv',index_col='Date')
    
    return(data,log_returns,garch_vol,mc_vol)


def compare_vols(ho,g_vol):
    nlevels = ho.params['nlevels']
    hw = ho.params['hw']
    vmin = .5*hw
    levels = np.zeros(shape=len(g_vol),dtype=int)
    for i in range(1,nlevels):
        if i==nlevels-1:
            idx = np.where(g_vol>=vmin.min()+hw*(nlevels-1))[0]
            levels[idx] = nlevels-1
        else:
            lo = vmin+hw*i
            hi = lo+hw
            idx = np.where((g_vol>=lo)&(g_vol<hi))[0]
            levels[idx] = i
            
    prd = np.empty(shape=(1000,500),dtype=float)
    for i in range(500):
        lvl = levels[i]
        rdraw = ho.rvs(lvl,1000)
        prd[:,i] = rdraw*ho.params['a']
    
    qho_vol = g_vol+prd
            
    return(qho_vol)
        
            
     

def optimize_ho(ho,gvol,mc_vol,max_iter=5000):
    '''
    '''
    kf = ho.params['kf']
    
    p = ho.create_posterior(gvol)   
    s0 = gvol.std()
    counter = 0
    while True:
        if counter==max_iter:
            break
        elif abs(s0-p.std())<.0002:
            break
        else:
            if p.std()>s0:
                kf+=100
            else:
                kf-=100
                
        ho.params['kf'] = kf
        ho.params['a'] = np.sqrt(hbar/(ho.params['m']*ho.params['w']))
        ho.params['m'] = kf/ho.params['w']**2
        ho = HarmonicOscillator(ho.params,ho.params['nlevels'])
        p = ho.create_posterior(gvol)
        
        counter+=1
        
    return(ho)       
            
            
    

