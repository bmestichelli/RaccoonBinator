import itertools
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from mpl_toolkits.mplot3d import Axes3D
import os
import scipy.spatial as spat
from scipy.spatial import KDTree
from scipy import stats
import numpy as np
from scipy.stats import loguniform

import parameters as par
import star_params as sp
import general_functions as gf


class cluster:

    def __init__(self, *args):

        if(isinstance(args[0],str)):
            self.m, self.x, self.y, self.z, self.vx, self.vy, self.vz = gf.load_dat(*args)  

        elif(isinstance(args[0], (list, np.ndarray))):
            if(isinstance(args[0], list)):
                self.m, self.x, self.y, self.z, self.vx, self.vy, self.vz = np.array(args[0]), np.array(args[1]), np.array(args[2]), np.array(args[3]), np.array(args[4]), np.array(args[5]), np.array(args[6]) 
            else:
                self.m, self.x, self.y, self.z, self.vx, self.vy, self.vz = args[0], args[1], args[2], args[3], args[4], args[5], args[6]

        elif(isinstance(args[0], (int, float))):
            self.generate_imf(args[0])
            self.x, self.y, self.z = np.zeros(len(self.m)), np.zeros(len(self.m)), np.zeros(len(self.m))
            self.vx, self.vy, self.vz = np.zeros(len(self.m)), np.zeros(len(self.m)), np.zeros(len(self.m))


        else:
            print("No instructions given")

        self.index = np.arange(len(self.m))
        self.id = np.arange(len(self.m))

        self.Mtot = np.sum(self.m)
        self.r = np.sqrt(self.x**2.+self.y**2.+self.z**2.)
        self.d_var = np.sqrt((np.var(self.x)+np.var(self.y)+np.var(self.z))/3.)
        self.dv_var = np.sqrt((np.var(self.vx)+np.var(self.vy)+np.var(self.vz))/3.)



    def generate_imf(self, M, parameter="mass", tol=4.*sp.mmin, lognormal=sp.lognorm): 
        starsappo=[]
        if parameter == "mass":
            if lognormal:
                while M > 0:
                    appo = float(loguniform.rvs(sp.mmin, sp.mMax, size=1))
                    while M - appo < sp.mmin:
                        appo = float(loguniform.rvs(sp.mmin, sp.mMax, size=1))
                    starsappo.append(appo)
                    M -= appo
                    if M < tol and M >= sp.mmin:
                        starsappo.append(M)
                        M = 0.
            else:
                while(M>0): # generate random numbers in range[0,1]
                    x = float(np.random.uniform(0.,1.))
                    if((sp.norm*x-sp.norm1)>0) :
                        appo =((1-sp.alpha2)*(sp.norm*x-sp.norm1)/sp.continuity+sp.mchange**(1-sp.alpha2))**(1/(1-sp.alpha2))
                        coun=0
                        while(M-appo<sp.mmin):
                            x = float(np.random.uniform(0.,1.))
                            if (sp.norm * x - sp.norm1) > 0:
                                appo = ((1-sp.alpha2)*(sp.norm*x-sp.norm1)/sp.continuity + sp.mchange**(1-sp.alpha2))**(1/(1-sp.alpha2))
                            else:
                                appo = ((1-sp.alpha1)*sp.norm*x + sp.mmin**(1-sp.alpha1))**(1/(1-sp.alpha1))
                            coun+=1
                        starsappo.append(appo)
                        M+=-appo
                    else:
                        appo =((1-sp.alpha1)*sp.norm*x+sp.mmin**(1-sp.alpha1))**(1/(1-sp.alpha1))
                        coun=0
                        while(M-appo<sp.mmin):
                            x = float(np.random.uniform(0.,1.))
                            if (sp.norm * x - sp.norm1) > 0:
                                appo = ((1-sp.alpha2)*(sp.norm*x-sp.norm1)/sp.continuity + sp.mchange**(1-sp.alpha2))**(1/(1-sp.alpha2))
                            else:
                                appo = ((1-sp.alpha1)*sp.norm*x + sp.mmin**(1-sp.alpha1))**(1/(1-sp.alpha1))
                            coun+=1
                        starsappo.append(appo)
                        M+=-appo
                    if(M<tol):
                        starsappo.append(M)
                        M=0.
        elif(parameter=="numb"): #In this case M is the number of particles
            if(lognormal==True):
                starsappo = loguniform.rvs(sp.mmin, sp.mMax, size=M)
            else:
                X = np.random.uniform(0.,1., size=M)
                for x in X:
                    if((sp.norm*x-sp.norm1)>0) :
                        appo =((1-sp.alpha2)*(sp.norm*x-sp.norm1)/sp.continuity+sp.mchange**(1-sp.alpha2))**(1/(1-sp.alpha2))
                        starsappo.append(appo)
                    else:
                        appo =((1-sp.alpha1)*sp.norm*x+sp.mmin**(1-sp.alpha1))**(1/(1-sp.alpha1))
                        starsappo.append(appo)
        self.m =np.array(starsappo)

    def generate_secondaries(self, m2=0., P=0., ecc=0., import_binaries=False):
        stars = np.stack((self.id, self.m, self.x, self.y, self.z, self.vx, self.vy, self.vz), axis=-1)
        stars = stars[np.argsort(stars[:, 0])]
        self.id, self.m, self.x, self.y, self.z, self.vx, self.vy, self.vz = [stars[:, i] for i in range(8)]
        # Recreate binaries object
        try:
            m1 = self.binaries.m1
        except AttributeError:
            m1 = self.m
        self.binaries = binaries(m1=m1, m2=m2, P=P, ecc=ecc, import_binaries=import_binaries)
        # Update counts and recompute masses
        self.Nbin = np.sum(self.binaries.m2 > 0)
        self.find_bin()
        self.m = self.binaries.m1 + self.binaries.m2
        self.Mtot = np.sum(self.m)

    def find_bin(self):
        self.isbin = np.zeros(len(self.m))
        self.istrip = np.zeros(len(self.m))


        for i in range(len(self.m)):
            if(self.binaries.m2[i]>0):
                self.isbin[i] = 1   
                
        for i in range(len(self.m)):
            if(self.binaries.m3[i]>0):
                self.istrip[i] = 1     


    def copy(self):

        return cluster(self.m[:],self.x[:],self.y[:],self.z[:],self.vx[:],self.vy[:],self.vz[:])


    def find_com(self, center=False, verbose=par.verbose):

        self.x_cm = np.sum(self.m*self.x) / self.Mtot
        self.y_cm = np.sum(self.m*self.y) / self.Mtot
        self.z_cm = np.sum(self.m*self.z) / self.Mtot
        self.vx_cm = np.sum(self.m*self.vx) / self.Mtot
        self.vy_cm = np.sum(self.m*self.vy) / self.Mtot
        self.vz_cm = np.sum(self.m*self.vz) / self.Mtot

        if(verbose==True):
            print("Center of mass position: ", self.x_cm, " [pc] ", self.y_cm, " [pc] ", self.z_cm, " [pc] ")
            print("Center of mass velocity: ", self.vx_cm, " [km/s] ", self.vy_cm, " [km/s] ", self.vz_cm, " [km/s] ")

        if(center==True):
            self.x = self.x - self.x_cm
            self.y = self.y - self.y_cm
            self.z = self.z - self.z_cm
            self.vx = self.vx - self.vx_cm
            self.vy = self.vy - self.vy_cm
            self.vz = self.vz - self.vz_cm

    def energies(self):
        # Pairwise distances (with softening if needed to avoid division by zero)
        dx = self.x[:, None] - self.x
        dy = self.y[:, None] - self.y
        dz = self.z[:, None] - self.z
        r = np.sqrt(dx**2 + dy**2 + dz**2)

        # Prevent division by zero (set diagonals to np.inf)
        np.fill_diagonal(r, np.inf)
           
        # Gravitational potential energy
        mprod = self.m[:, None] * self.m
        Eg_matrix = -0.5 * par.G * mprod / r
        self.Eg = np.sum(Eg_matrix)  # Total potential energy
        
        # Kinetic energy
        self.Ek = 0.5 * np.sum(self.m * (self.vx**2 + self.vy**2 + self.vz**2))

        # Virial ratio
        self.vratio = 2 * self.Ek / abs(self.Eg)

        # Virial radius, time, and velocity scales
        self.v_radius = par.G * np.sum(self.m)**2 / (4 * self.Ek)
        self.t_scale = np.sqrt(self.v_radius**3 / (par.G * self.Mtot))
        self.v_scale = self.v_radius / np.sqrt(self.v_radius**3 / (par.G * self.Mtot))

    def rescale_to_virial(self, scaling="vel"):
        self.energies()

        if(scaling=="vel"):
            self.vx = self.vx / np.sqrt(self.vratio)
            self.vy = self.vy / np.sqrt(self.vratio)
            self.vz = self.vz / np.sqrt(self.vratio)

        elif(scaling=="pos"):      
            self.x = self.x / self.vratio
            self.y = self.y / self.vratio
            self.z = self.z / self.vratio 


    def density(self,k=500, center=False):

        pos = np.stack((self.x, self.y, self.z),axis=-1)
        pos_tree=spat.cKDTree(pos)
        denr=0.

        self.rho = np.zeros(len(self.m))

        for j in np.arange(len(self.m)):
            indneigh=pos_tree.query(pos[j],k)[1]
            dist_tree = np.sqrt((self.x[indneigh]-self.x[j])**2.+(self.y[indneigh]-self.y[j])**2.+(self.z[indneigh]-self.z[j])**2.)

            den=np.sum(self.m[indneigh])/(4./3.*np.pi*(max(dist_tree))**3.)
            self.rho[j] = den

            if(den>denr):
                m_big=self.m[indneigh]
                x_big=self.x[indneigh]
                y_big=self.y[indneigh]
                z_big=self.z[indneigh]
                vx_big=self.vx[indneigh]
                vy_big=self.vy[indneigh]
                vz_big=self.vz[indneigh]
                denr=den

        self.x_cod=np.sum(x_big*m_big)/(np.sum(m_big))
        self.y_cod=np.sum(y_big*m_big)/(np.sum(m_big))
        self.z_cod=np.sum(z_big*m_big)/(np.sum(m_big))
        self.vx_cod=np.sum(vx_big*m_big)/(np.sum(m_big))
        self.vy_cod=np.sum(vy_big*m_big)/(np.sum(m_big))
        self.vz_cod=np.sum(vz_big*m_big)/(np.sum(m_big))

        if(center==True):
            self.x = self.x - self.x_cod
            self.y = self.y - self.y_cod
            self.z = self.z - self.z_cod
            self.vx = self.vx - self.vx_cod
            self.vy = self.vy - self.vy_cod
            self.vz = self.vz - self.vz_cod


    def lagr_radius(self, lagr=50):
        r_vect = np.sqrt(self.x**2+self.y**2+self.z**2)

        rbin = np.linspace(min(r_vect),max(r_vect),4000)
        mbin_r = np.zeros(len(rbin))

        for j in range(len(rbin)):     
            mappovect= np.where(r_vect<rbin[j],self.m,0)
            mbin_r[j] = np.sum(mappovect)

        r_lagr = np.interp(self.Mtot/(100./lagr), mbin_r, rbin)

        return r_lagr

    def distances(self):

        self.dist = np.zeros((len(self.m),len(self.m)-1))

        for j in range(len(self.m)):
            xvectappo = np.delete(self.x,j)
            yvectappo = np.delete(self.y,j)
            zvectappo = np.delete(self.z,j)

            self.dist[j]=np.sqrt((self.x[j]-xvectappo)**2+(self.y[j]-yvectappo)**2+(self.z[j]-zvectappo)**2) #evaluate the distribution of distances


    def neighbors(self, binning):
        pos= np.stack((self.x, self.y, self.z),axis=-1)
        pos_tree=spat.cKDTree(pos)

        neig = np.zeros(len(binning))
        self.neigh = np.zeros(len(binning))

        for j in np.arange(len(self.m)):
            for n in range(len(binning)):
                indneigh=len(pos_tree.query_ball_point(pos[j],r=binning[n]))-1
                neig[n]+= indneigh

        self.neigh = neig/(len(self.m)-1)


    def stellar_radii(self, Z, verbose=par.verbose):
        self.R = sp.star_radius(self.m, Z)


    def binary_fraction(self, verbose=par.verbose):
        binstot, binsbinaries, binstriplets = np.zeros(sp.nbin), np.zeros(sp.nbin), np.zeros(sp.nbin)

        fbinaries, ftriplets, fmult = np.zeros(sp.nbin), np.zeros(sp.nbin), np.zeros(sp.nbin)

        for j in range(sp.nbin):
            ntotbin, ntotbinb, ntottrip = 0, 0, 0

            for i in range(len(self.m)):
                if((self.binaries.m1[i]>=sp.mbin[j]-sp.merr[j]) and (self.binaries.m1[i]< sp.mbin[j]+sp.merr[j])):
                    if(self.binaries.m3[i]>0.):
                        ntotbin+=1
                        ntottrip+=1
                    elif(self.binaries.m2[i]>0.):
                        ntotbinb+=1
                        ntotbin+=1  
                    elif(self.binaries.tertiary[i]==0):
                        ntotbin+=1
                
            binsbinaries[j]=ntotbinb
            binstriplets[j]=ntottrip
            binstot[j]= ntotbin
            
        #binsbinaries=binsbinaries-binstriplets
        Ntriplets = len(self.istrip[self.istrip[:]==1])
        Nbinaries = len(self.isbin[(self.isbin[:]==1) & (self.binaries.tertiary==0)])
        Nsingles = len(self.isbin[(self.isbin[:]==0) & (self.binaries.tertiary==0)])
        Nparticles= Nsingles + Nbinaries + Ntriplets

        fbinaries=np.divide(binsbinaries,binstot).round(4)
        ftriplets=np.divide(binstriplets,binstot).round(4)
        fmult=np.divide(binsbinaries+binstriplets,binstot).round(4)

        self.binaries.Nsing, self.binaries.Nbin, self.binaries.Ntrip  = Nsingles, Nbinaries, Ntriplets 
        self.binaries.fbin, self.binaries.ftrip = fbinaries, ftriplets  

        if(verbose==True):
            print("Total mass:", round(self.Mtot,4))        
            print("Total number of particles (single stars+binary systems):",  Nparticles)  
            print("Total number of stars:",  Nsingles+2*Nbinaries+3*Ntriplets)

            print("Total number of single stars:",  Nsingles)
            print("Total number of binary systems:", Nbinaries) 
            print("Total number of triple systems:", Ntriplets) 
            
            print("Total mass of stars in binaries:", np.sum(self.m[self.isbin[:]==1])) 
            print("\n")


            print("Total percentage of multiple systems:", round(100.*float(Nbinaries+Ntriplets)/float(Nparticles),4),"%")
            print("Total percentage of binaries:", round(100.*float(Nbinaries)/float(Nparticles),4),"%")
            print("Total percentage of triplets:", round(100.*float(Ntriplets)/float(Nparticles),4),"%")
            print ("Total percentage of stars in binaries:", round(100.*2.*float(Nbinaries+2*Ntriplets)/float(Nparticles+Nbinaries+2*Ntriplets),4), "%")


            print("Percentage of binaries (only stars with m>0.8 Msun )",round(100.*float(Nbinaries-binsbinaries[0])/float(Nparticles-binstot[0]),4),"%")
            print("Percentage of triplets (only stars with m>0.8 Msun )",round(100.*float(Ntriplets-binstriplets[0])/float(Nparticles-binstot[0]),4),"%")


            print("Number of primaries per bin: ",  binstot)    
            print("Number of multiplets per bin:", binsbinaries+binstriplets)
            print("Number of binaries per bin:", binsbinaries)
            print("Number of triplets per bin:", binstriplets)
            print("Fraction of multiplets per bin:", np.divide(binsbinaries+binstriplets,binstot).round(4))
            print("Fraction of binaries per bin:", np.divide(binsbinaries,binstot).round(4))
            print("Fraction of triplets per bin:", np.divide(binstriplets,binstot).round(4))


        
class binaries:

    def __init__(self, m1, m2, P, ecc, import_binaries=False):
        
        #Possibility to import an existing population of binaries
        if(import_binaries==True):
            
            self.m1= np.copy(m1)
            self.id = np.arange(len(m1))
            self.m2= np.copy(m2)
            self.m3 = np.zeros(len(self.m1))
            
            self.q = self.m2 / self.m1
            self.q3 = self.m3 / self. m1
            
            self.P = np.copy(P)
            self.ecc = np.copy(ecc)
            self.smax = pow(par.G*(m1+m2)*(self.P*24.*60.*60.)**2. / (2.*np.pi*par.pcinkm)**2., 1./3.)
            
            self.Nbin = len(self.m1[self.m2>0])
            self.tertiary= np.zeros(len(self.m1))
        
        #Otherwise import a population of primaries and then generate secondaries and orbital properties
        else:
            self.m1 = m1
            self.id = np.arange(len(self.m1))
            self.secondaries()
            self.q = self.m2 / self.m1
            self.q3 = self.m3 / self.m1
            
            self.P = np.zeros(len(self.m1))
            self.ecc = np.zeros(len(self.m1))
            self.smax = np.zeros(len(self.m1))

            self.Nbin = len(self.m1[self.m2>0])
            
            
        #self.index = np.arange(self.Nbin)
        #self.P = P #Periods in days
        #self.ecc = ecc
        #self.smax = pow(G*(self.m1+self.m2)*(self.P*24.*60.*60.)**2. / (2.*np.pi*pcinkm)**2., 1./3.) #semi-major axis in pc

    def radii(self, Z=sp.metal, verbose=par.verbose):
        self.R1 = sp.star_radius(self.m1, Z)
        self.R2 = sp.star_radius(self.m2, Z)



    def generate_dist(self, d_min, d_Max, d_k, num_bin): 
        norm = 1./(1.+d_k) * (d_Max**(1.+d_k)-d_min**(1.+d_k))

        x = np.random.uniform(0.,1., size=num_bin)
        appo =((1. + d_k)*norm*x + d_min**(1.+d_k))**(1./(1.+d_k))              
        return appo


    def secondaries(self, triple_syst=sp.multiples, m_thres=sp.m_thres_low):
 
        stars = np.stack((self.id,self.m1), axis=-1)
        stars = stars[np.argsort(stars[:,1])]
        
        self.id = stars[:,0]
        self.m1 = stars[:,1]
 
        self.m2 = np.zeros(len(self.m1))
        self.m3 = np.zeros(len(self.m1))
        
        self.tertiary = np.zeros(len(self.m1))
        
        mmin_prim = min(self.m1)
        
        ind_mass = 0
 
        #Chack the range in which there is the switch between high-mass and low-mass
        #For now we consider everything as high-mass
        for i in range(sp.nbin):
            if(sp.mbin[i]+sp.merr[i]<=m_thres):
                ind_mass = i        
                
        #generate low-mass binaries
        #print("Low-mass binaries")
        i = sp.nbin-1
        while(i>=0):
 
            print("Mass range: ", round(sp.mbin[i]-sp.merr[i],1), "-", round(sp.mbin[i]+sp.merr[i],1))
            
            #Select the stars in the mass bin
            index= len(self.m1[self.m1<sp.mbin[i]-sp.merr[i]]) 
            stars_bin = np.copy(self.m1[(self.m1>=sp.mbin[i]-sp.merr[i]) & (self.m1< sp.mbin[i]+sp.merr[i])])
            tert_bin = np.copy(self.tertiary[(self.m1>=sp.mbin[i]-sp.merr[i]) & (self.m1< sp.mbin[i]+sp.merr[i])])
            m3_bin = np.copy(self.m3[(self.m1>=sp.mbin[i]-sp.merr[i]) & (self.m1< sp.mbin[i]+sp.merr[i])])
            
            m1binmax = sp.binary_function(sp.mbin[i])[1]
            m1binmax_no_mult = sp.binary_function_no_multiples(sp.mbin[i])[1]
            m1binmax_mult = m1binmax-m1binmax_no_mult
            
            #Print the theoretical values
            if(triple_syst==True):
                print("Expected binary/tertiary fractions: fbin=", round(m1binmax_no_mult,2), ", ftrip=", round(m1binmax_mult,2))
            else:
                print("Expected binary/tertiary fractions: fbin=", round(m1binmax_no_mult,2), ", ftrip=", 0)
                
            
            prim = stars_bin
            seco, trip = np.zeros(len(prim)), np.zeros(len(prim))
            
            index_prim = np.arange(len(prim))
            
            num_prim = len(prim[tert_bin==0])
            
            #First generate triples, so I don't have to correct for binary fraction later   
            num_trip = 0
                
            if(triple_syst==True):
                #Generate triples until I get to the desired triple fraction
                while(num_trip/num_prim<m1binmax_mult):
                
                    #num_trip = int(len(prim)*(m1binmax-m1binmax_no_mult)) 
                    #Select the index of the primary star
                    index_trip = np.random.choice(index_prim[tert_bin==0], 1, replace=False)
                    
                    #NB it can't be the primary with the minimum mass
                    while(prim[index_trip]==mmin_prim):
                        index_trip = np.random.choice(index_prim[tert_bin==0], 1, replace=False)
                    
                    if sp.medium_interpolate == True:
                        # Three-regime case for tertiary companions
                        if sp.mbin[i] + sp.merr[i] <= sp.m_thres_low:
                            appo = self.generate_dist(sp.qmin3_LM, sp.qMax3_LM, sp.k3_LM, 1)
                        elif sp.mbin[i] - sp.merr[i] >= sp.m_thres_high:
                            appo = self.generate_dist(sp.qmin3_HM, sp.qMax3_HM, sp.k3_HM, 1)
                        else:
                            appo = self.generate_dist(sp.qmin3_MM, sp.qMax3_MM, sp.k3_MM, 1)
                    else:
                        # Two-regime case
                        if(i>ind_mass):
                            appo = self.generate_dist(sp.qmin3_HM, sp.qMax3_HM, sp.k3_HM, 1)
                        else:
                            appo = self.generate_dist(sp.qmin3_LM, sp.qMax3_LM, sp.k3_LM, 1)
                    
                    #Assign the theoretical mass of the triple star
                    m3_appo = prim[index_trip]*appo
                    
                    #Draw the mass of the triple from the primary masses in the cluster: choose the star with the closest value to the theoretical value
                    #We choose the masses from the primaries because these are "fake" triples, so primary stars that count as triples to adjust the binary fraction
                    idx = np.argsort(np.abs(self.m1-m3_appo))[0]
 
                    #Have to do it again if m3 > m1 or if I have selected a mass that is already in a triple system
                    while((self.m1[idx]>=prim[index_trip]) or (self.tertiary[idx]>0) or (self.m3[idx]>0)):
                    
                        if sp.medium_interpolate == True:
                            # Three-regime case for tertiary companions
                            if sp.mbin[i] + sp.merr[i] <= sp.m_thres_low:
                                appo = self.generate_dist(sp.qmin3_LM, sp.qMax3_LM, sp.k3_LM, 1)
                            elif sp.mbin[i] - sp.merr[i] >= sp.m_thres_high:
                                appo = self.generate_dist(sp.qmin3_HM, sp.qMax3_HM, sp.k3_HM, 1)
                            else:
                                appo = self.generate_dist(sp.qmin3_MM, sp.qMax3_MM, sp.k3_MM, 1)
                        else:
                            # Two-regime case
                            if(i>ind_mass):
                                appo = self.generate_dist(sp.qmin3_HM, sp.qMax3_HM, sp.k3_HM, 1)
                            else:
                                appo = self.generate_dist(sp.qmin3_LM, sp.qMax3_LM, sp.k3_LM, 1)
                            
                        m3_appo = prim[index_trip]*appo
                        idx = np.argsort(np.abs(self.m1-m3_appo))[0]
                    
                    
                    #Assign the value to m3
                    self.m3[index+index_trip] = np.copy(self.m1[idx])
                    self.tertiary[idx] = 1
                                        
                    tert_bin = np.copy(self.tertiary[(self.m1>=sp.mbin[i]-sp.merr[i]) & (self.m1< sp.mbin[i]+sp.merr[i])])
                    
                    
                    #Calculate the triple fraction 
                    m3_bin[index_trip] = np.copy(self.m1[idx])
                        
                    num_prim = len(prim[tert_bin==0])
                    num_trip = len(tert_bin[m3_bin>0])
                    
                    #print(num_prim, num_trip, num_trip/num_prim,m1binmax_mult)
                                        
                num_bin_no_mult = int(len(prim[tert_bin==0])*m1binmax_no_mult) 
 
                index_trip = index_prim[m3_bin>0]
                index_bin_appo = np.random.choice(index_prim[(m3_bin==0) & (tert_bin==0)], num_bin_no_mult, replace=False)
                index_bin = np.concatenate((index_trip,index_bin_appo), axis=-1)
                
                num_bin = len(index_bin)
            else: 
                num_bin = int(len(prim)*m1binmax) 
 
                index_bin = np.random.choice(len(prim), num_bin, replace=False)
                
                
            #Generate the second star following the appropriate prescription
            # For medium_interpolate=False: LM for m<2, HM for m>=2
            # For medium_interpolate=True: LM for m<2, MM for 2<=m<15, HM for m>=15
            
            if sp.medium_interpolate == True:
                # Three-regime case: low, medium, high mass
                if sp.mbin[i] + sp.merr[i] <= sp.m_thres_low:
                    # Low-mass regime
                    appo = self.generate_dist(sp.qmin_LM, sp.qMax_LM, sp.k_LM, num_bin)
                elif sp.mbin[i] - sp.merr[i] >= sp.m_thres_high:
                    # High-mass regime
                    appo = self.generate_dist(sp.qmin_HM, sp.qMax_HM, sp.k_HM, num_bin)
                else:
                    # Medium-mass regime (Moe & Di Stefano 2017)
                    appo = self.generate_dist(sp.qmin_MM, sp.qMax_MM, sp.k_MM, num_bin)
            else:
                # Two-regime case: low and high mass
                if i > ind_mass:     
                    appo = self.generate_dist(sp.qmin_HM, sp.qMax_HM, sp.k_HM, num_bin)
                else:
                    appo = self.generate_dist(sp.qmin_LM, sp.qMax_LM, sp.k_LM, num_bin)
            
            #Pick the maximum between the generated mass and the minimum mass in star_params
            seco[index_bin] = np.maximum(prim[index_bin]*appo,np.zeros(len(prim[index_bin]))+sp.mmin) #get the theoretical mass from m2
 
            self.m2[index+index_bin] = np.copy(seco[index_bin])
            
 
            #Now print the generated binary and triple fractions 
            
            if(triple_syst==True):
                fbin = round((num_bin - num_trip) / num_prim, 2) if num_prim != 0 else 0.0
                ftrip = round(num_trip / num_prim, 2) if num_prim != 0 else 0.0
                print(f"Obtained binary/tertiary fractions: fbin= {fbin}, ftrip= {ftrip}")
                #print("Obtained binary/tertiary fractions: fbin=", round((num_bin-num_trip)/num_prim,2), ", ftrip=", round(num_trip/num_prim,2))
#print(num_prim, num_bin, num_trip, len(prim))
            else:
                fbin = round((num_bin - num_trip) / num_prim, 2) if num_prim != 0 else 0.0
                ftrip = round(num_trip / num_bin, 2) if num_bin != 0 else 0.0
                print(f"Obtained binary/tertiary fractions: fbin= {fbin}, ftrip= {ftrip}")
                #print("Obtained binary/tertiary fractions: fbin=", round((num_bin-num_trip)/num_prim,2), ", ftrip=", round(num_trip/num_bin,2))        
 
            i+=-1
 
        stars = np.stack((self.id,self.m1,self.m2, self.m3, self.tertiary), axis=-1)
        stars = stars[np.argsort(stars[:,0])]
        
        self.id, self.m1, self.m2, self.m3, self.tertiary = stars[:,0], stars[:,1], stars[:,2], stars[:,3], stars[:,4]

    def orbital_properties(self, selection="pmin", m_thres_low=sp.m_thres_low, m_thres_high=sp.m_thres_high, medium_mass=sp.medium_interpolate):
        
        #Generate orbital properties for low-mass binaries
        lowmass = np.stack((self.id[self.m1<m_thres_low], self.m1[self.m1<m_thres_low], self.m2[self.m1<m_thres_low]), axis=-1)
        num_bin = len(lowmass[lowmass[:,2]>0][:,1])
        index_bin = np.where(lowmass[:,2] > 0)[0]       
 
        ecc = np.zeros(len(lowmass[:,0]))
        P = np.zeros(len(lowmass[:,0]))
        smax = np.zeros(len(lowmass[:,0]))
 
        ecc[index_bin] = self.generate_dist(sp.emin_LM, sp.eMax_LM, sp.eta_LM, num_bin)
        lowmass = np.column_stack((lowmass,ecc))
 
        appo_P = 10.** stats.truncnorm((sp.pmin_LM - sp.pmean_LM) / sp.psigma_LM, (sp.pMax_LM - sp.pmean_LM) / sp.psigma_LM, loc=sp.pmean_LM, scale=sp.psigma_LM).rvs(num_bin)
 
        lowmass = self.eP_relation(lowmass, appo_P)
 
        #Generate orbital properties for massive binaries
        if medium_mass == False: #we apply Sana et al. 2012 / Stacy & Bromm 2013 from m>2 Msun
            highmass = np.stack((self.id[self.m1>=m_thres_low], self.m1[self.m1>=m_thres_low], self.m2[self.m1>=m_thres_low]), axis=-1)
            num_bin= len(highmass[highmass[:,2]>0][:,1])
            index_bin = np.where(highmass[:,2]>0)
 
            ecc = np.zeros(len(highmass[:,0]))
            P = np.zeros(len(lowmass[:,0]))
            smax = np.zeros(len(lowmass[:,0]))
            if sp.bin_params == 's12':
                ecc[index_bin] = self.generate_dist(sp.emin_HM, sp.eMax_HM, sp.eta_HM, num_bin)
            elif sp.bin_params == 'sb13':
                ecc[index_bin] = self.generate_dist_linear_e(sp.emin_HM, sp.eMax_HM, num_bin)
 
            highmass = np.column_stack((highmass,ecc))
 
            if sp.bin_params == 's12':
                appo_P = 10.**self.generate_dist(sp.pmin_HM, sp.pMax_HM, sp.pigr_HM, num_bin)
            elif sp.bin_params == 'sb13':
                appo_P = 10.**np.random.normal(sp.pmu_HM, sp.psigma_HM, num_bin)
 
            highmass = self.eP_relation(highmass, appo_P)
 
        if medium_mass == True: #we apply interpolated distributions for 2<m<15 Msun and Sana/Stacy & Bromm for m>15 Msun
            # Medium-mass stars (2-15 Msun)
            mediummass = np.stack((self.id[(self.m1>=m_thres_low) & (self.m1<m_thres_high)], 
                                   self.m1[(self.m1>=m_thres_low) & (self.m1<m_thres_high)], 
                                   self.m2[(self.m1>=m_thres_low) & (self.m1<m_thres_high)]), axis=-1)
            num_bin_MM = len(mediummass[mediummass[:,2]>0][:,1])
            index_bin_MM = np.where(mediummass[:,2]>0)
 
            ecc_MM = np.zeros(len(mediummass[:,0]))
            P_MM = np.zeros(len(mediummass[:,0]))
            smax_MM = np.zeros(len(mediummass[:,0]))
            
            # Eccentricity: linear distribution f(e) ∝ e
            ecc_MM[index_bin_MM] = self.generate_dist_linear_e(sp.emin_MM, sp.eMax_MM, num_bin_MM)
            mediummass = np.column_stack((mediummass, ecc_MM))
 
            # Period: power law in log(P)
            appo_P_MM = 10.**self.generate_dist(sp.pmin_MM, sp.pMax_MM, sp.pigr_MM, num_bin_MM)
            mediummass = self.eP_relation(mediummass, appo_P_MM)
 
            # High-mass stars (m>15 Msun): Sana 2012 / Stacy & Bromm 2013
            highmass = np.stack((self.id[self.m1>=m_thres_high], 
                                self.m1[self.m1>=m_thres_high], 
                                self.m2[self.m1>=m_thres_high]), axis=-1)
            num_bin_HM = len(highmass[highmass[:,2]>0][:,1])
            index_bin_HM = np.where(highmass[:,2]>0)
 
            ecc_HM = np.zeros(len(highmass[:,0]))
            P_HM = np.zeros(len(highmass[:,0]))
            smax_HM = np.zeros(len(highmass[:,0]))
            
            if sp.bin_params == 's12':
                ecc_HM[index_bin_HM] = self.generate_dist(sp.emin_HM, sp.eMax_HM, sp.eta_HM, num_bin_HM)
            elif sp.bin_params == 'sb13':
                ecc_HM[index_bin_HM] = self.generate_dist_linear_e(sp.emin_HM, sp.eMax_HM, num_bin_HM)
 
            highmass = np.column_stack((highmass, ecc_HM))
 
            if sp.bin_params == 's12':
                appo_P_HM = 10.**self.generate_dist(sp.pmin_HM, sp.pMax_HM, sp.pigr_HM, num_bin_HM)
            elif sp.bin_params == 'sb13':
                appo_P_HM = 10.**np.random.normal(sp.pmu_HM, sp.psigma_HM, num_bin_HM)
 
            highmass = self.eP_relation(highmass, appo_P_HM)
            
        starmatrix = np.concatenate((lowmass, mediummass, highmass)) if medium_mass == True else np.concatenate((lowmass, highmass))
        starmatrix = starmatrix[np.argsort(starmatrix[:,0])]
 
        self.id, self.m1, self.m2, self.ecc, self.P, self.smax = starmatrix[:,0], starmatrix[:,1], starmatrix[:,2], starmatrix[:,3], starmatrix[:,4], starmatrix[:,5] 
        self.q = self.m2 / self.m1
 
        return         


    def generate_dist_linear_e(self, e_min, e_Max, num_bin):
        """Generate eccentricity from 2*e distribution (linear in e)"""
        x = np.random.uniform(0., 1., size=num_bin)
        appo = np.sqrt(e_min**2 + x * (e_Max**2 - e_min**2))
        return appo

    def eP_relation(self, binmatrix, appo_P, selection="pmin", Z=sp.metal, verbose=par.verbose):

        binmatrix = binmatrix[binmatrix[:,-1].argsort()[::-1]] # Sort binaries with decreasing eccentricity
        appo_P = appo_P[appo_P.argsort()[::-1]]

        # Let's assign a period to each binary according to a physical criteria
        """To choose the proper period for a binary, you can either use the criteria given 
        by eq.3 in Moe & Di Stefano 2017 or check if there's a collision. 
        Please uncomment just one of the following two rows according to the criteria you want"""    
        #P_min = lambda a_min, m1, m2 : pow((a_min**3*(2*np.pi*pc_to_km)**2) / ((m1+m2)*G*(24*60*60)**2), 1/2) # Min P to just avoid collisions
        P_min = lambda e : 2. * (1.+1e-8-e)**(-3./2.) # Eq.3 in Moe & Di Stefano 2017
        P_counter = 0 # Number of periods below P_min

        P, smax = np.zeros(len(binmatrix[:,0])), np.zeros(len(binmatrix[:,0]))


        # Cycle throug binaries
        for i in range(len(appo_P)): 
            # Get values for the current binary
            m1 = binmatrix[i,1]
            m2 = binmatrix[i,2]
            e = binmatrix[i,3]


            if(selection=="pmin"):
                good_Ps = appo_P[appo_P >= P_min(e)]

            elif(selection=="coll"):
                Rtot = sp.star_radius(m1, Z) + sp.star_radius(m2, Z) # Sum of radii
                dmin_to_Rtot = 1.2 # Ratio of min acceptable value of pericenter distance to Rtot
                a_min = Rtot * dmin_to_Rtot / (1-e)
                good_Ps = appo_P[appo_P >= P_min(a_min, m1, m2)]

            # Select one random period out of those greater than P_min
            if good_Ps.size != 0:
                index_choice = np.random.choice(len(good_Ps))
                rand_P = good_Ps[index_choice] #np.random.choice(good_Ps[index_choice])
            else:
                rand_P = np.random.choice(appo_P[:10])
                index_choice = np.where(appo_P==rand_P)
                P_counter += 1

            # Assign period and semi-major axis
            P[i] = rand_P
            smax[i] = pow(par.G*(m1+m2)*(rand_P*24.*60.*60.)**2. / (2.*np.pi*par.pcinkm)**2., 1./3.) #pow(G*(m1+m2)*(rand_P*24.*60.*60.)**2. / (2.*np.pi*pcinkm)**2., 1./3.)

            # Remove used period
            appo_P = np.delete(appo_P, index_choice)

        binmatrix = np.column_stack((binmatrix,P))
        binmatrix = np.column_stack((binmatrix,smax))
        binmatrix = binmatrix[np.argsort(binmatrix[:,0])]

        return binmatrix


    def orbits(self,verbose=par.verbose):

        print("Generate phases")
        self.phases()
        print("Generate binary positions")
        self.binary_positions()
        print("Randomly incline")
        self.binary_incline()

        return 


    def phases(self, psimin=0.0, psimax=2.*np.pi, tol=1e-7):
        self.phase = np.zeros(len(self.m1)) 

        phase = np.zeros(self.Nbin)

        for i in range(self.Nbin):

            p = float(np.random.uniform(psimin,psimax))
            Efinale = gf.bisec(p, self.ecc[i], tol) #call bisection and solves eccentric anomaly

            phase[i] = 2.0 * np.arctan((((1.+self.ecc[i])/(1.-self.ecc[i])))**0.5 * np.tan(Efinale/2.))

        self.phase[self.m2>0] = phase

        return


    def binary_positions(self):

        self.xb1, self.yb1, self.zb1, self.vxb1, self.vyb1, self.vzb1 = np.zeros(len(self.m1)), np.zeros(len(self.m1)), np.zeros(len(self.m1)), np.zeros(len(self.m1)), np.zeros(len(self.m1)), np.zeros(len(self.m1))

        self.xb2, self.yb2, self.zb2, self.vxb2, self.vyb2, self.vzb2 = np.zeros(len(self.m1)), np.zeros(len(self.m1)), np.zeros(len(self.m1)), np.zeros(len(self.m1)), np.zeros(len(self.m1)), np.zeros(len(self.m1))

        mask = self.m2 > 0
        m1, m2, ecc, smax, phase = self.m1[mask], self.m2[mask], self.ecc[mask], self.smax[mask], self.phase[mask]

        self.xb1[mask] = -m2/(m1+m2) * smax * (1.-ecc**2.) * np.cos(phase)/(1.+(ecc * np.cos(phase)))
        self.yb1[mask] = -m2/(m1+m2) * smax * (1.-ecc**2.) * np.sin(phase)/(1.+(ecc * np.cos(phase)))  

        self.vxb1[mask] = -m2/(m1+m2) * (ecc * np.cos(phase) /(1+ ecc * np.cos(phase)) - 1.) * np.sin(phase) * (1+ ecc * np.cos(phase)) * (par.G* (m1+m2) / (smax*(1- ecc*ecc)))**0.5 
        self.vyb1[mask] = -m2/(m1+m2) * (ecc * (np.sin(phase))**2 /(1+ ecc  * np.cos(phase)) + np.cos(phase)) * (1+ ecc * np.cos(phase)) * (par.G* (m1+m2) / (smax*(1- ecc*ecc)))**0.5     

        self.xb2[mask] = m1/(m1+m2) * smax * (1.-ecc**2.) * np.cos(phase)/(1.+(ecc * np.cos(phase)))
        self.yb2[mask] = m1/(m1+m2) * smax * (1.-ecc**2.) * np.sin(phase)/(1.+(ecc * np.cos(phase)))  

        self.vxb2[mask] = m1/(m1+m2) * (ecc * np.cos(phase) /(1.+ ecc * np.cos(phase)) - 1.) * np.sin(phase) * (1.+ ecc * np.cos(phase)) * (par.G* (m1+m2) / (smax*(1.- ecc*ecc)))**0.5   
        self.vyb2[mask] = m1/(m1+m2) * (ecc * (np.sin(phase))**2 /(1.+ ecc * np.cos(phase)) + np.cos(phase)) * (1.+ ecc * np.cos(phase)) * (par.G* (m1+m2) / (smax*(1.- ecc*ecc)))**0.5           

        return


    def binary_incline(self):

        mask = self.m2 > 0

        inclvect = np.zeros((len(self.m1),3))

        x1, y1, z1, x2, y2, z2 = self.xb1, self.yb1, self.zb1, self.xb2, self.yb2, self.zb2
        vx1, vy1, vz1, vx2, vy2, vz2 = self.vxb1, self.vyb1, self.vzb1, self.vxb2, self.vyb2, self.vzb2

        for i in range(len(self.m1)):
            
            if(self.m2[i]>0):

                print(i, end="\r")

                point1 = np.array([x1[i],y1[i],z1[i]])
                point2 = np.array([x2[i],y2[i],z2[i]])

                normv = (0.,0.,1.)

                xaxr, yaxr, zaxr = np.random.uniform(-1.,1.), np.random.uniform(-1.,1.), np.random.uniform(-1.,1.)  

                normvr=np.array([xaxr,yaxr,zaxr])
                normvr= normvr/ np.linalg.norm(normvr)
                inclvect[i] = normvr 

                costheta = np.dot(normv,normvr)/(np.linalg.norm(normv)*np.linalg.norm(normvr))
                rotaxis = np.cross(normv,normvr)/np.linalg.norm(np.cross(normv,normvr))

                c = costheta
                s = np.sqrt(1.-c*c)
                C = 1.-c
                rmat = np.array([[ rotaxis[0]*rotaxis[0]*C+c, rotaxis[0]*rotaxis[1]*C-rotaxis[2]*s,rotaxis[0]*rotaxis[2]*C+rotaxis[1]*s ],
                [ rotaxis[1]*rotaxis[0]*C+rotaxis[2]*s, rotaxis[1]*rotaxis[1]*C+c, rotaxis[1]*rotaxis[2]*C-rotaxis[0]*s ],
                [ rotaxis[2]*rotaxis[0]*C-rotaxis[1]*s, rotaxis[2]*rotaxis[1]*C+rotaxis[0]*s,rotaxis[2]*rotaxis[2]*C+c   ]])

                p1 = np.dot(rmat, point1)
                p2 = np.dot(rmat, point2)

                self.xb1[i]=p1[0]
                self.yb1[i]=p1[1]
                self.zb1[i]=p1[2]
                self.xb2[i]=p2[0]
                self.yb2[i]=p2[1]
                self.zb2[i]=p2[2]

                vpoint1 = np.array([vx1[i],vy1[i],vz1[i]])
                vpoint2 = np.array([vx2[i],vy2[i],vz2[i]])

                pv1 = np.dot(rmat, vpoint1)
                pv2 = np.dot(rmat, vpoint2)


                self.vxb1[i]=pv1[0]
                self.vyb1[i]=pv1[1]
                self.vzb1[i]=pv1[2]

                self.vxb2[i]=pv2[0]
                self.vyb2[i]=pv2[1]
                self.vzb2[i]=pv2[2]

