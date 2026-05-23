from scipy.stats import loguniform
import matplotlib.pyplot as plt
import os
import numpy as np
import parameters as par
import star_params as sp
import classes as cl
print( "--------------------------------------")
print( "   Initial conditions from Mcluster   ")
print( "--------------------------------------")

#input the filename as a string

#import the main parameters from Mcluster input file (Number of stars, number of binaries, metallicity, mean mass)
simnamepar = par.input_file_name
Nstars_old = int(np.genfromtxt(simnamepar,dtype="float", skip_header=1, usecols=(0), comments="#", unpack=True)[0])
Nbinaries_old = int(np.genfromtxt(simnamepar,dtype="float", skip_header=9, usecols=(3), comments="#", unpack=True)[0])
metal_star_old = float(np.genfromtxt(simnamepar,dtype="float",  skip_header=9,  skip_footer=1, usecols=(5), comments="#", unpack=True))
mean_old = float(np.genfromtxt(simnamepar,dtype="float", skip_header=2,  skip_footer=8, usecols=(8), comments="#", unpack=True))
rvir0 = float(np.genfromtxt(simnamepar,dtype="float", skip_header=2,  skip_footer=8, usecols=(7), comments="#", unpack=True))

print( "Total mass:", round(float(Nstars_old)*mean_old,5))
print( "Number of stars:", Nstars_old)
print( "Number of binaries:", Nbinaries_old)
print( "Metallicity: Z =", metal_star_old)


simname = par.input_dat
starmatrix = np.genfromtxt(simname,dtype="float", comments="#", unpack=True)

stars_old = cl.cluster(simname)
stars_old.find_com(center=True)

#vector of particles (c.o.m. of binaries and single stars)
m1_part = np.zeros(len(stars_old.m)-Nbinaries_old)
m2_part = np.zeros(len(stars_old.m)-Nbinaries_old)

x_part = np.zeros(len(stars_old.m)-Nbinaries_old)
y_part = np.zeros(len(stars_old.m)-Nbinaries_old)
z_part = np.zeros(len(stars_old.m)-Nbinaries_old)
vx_part = np.zeros(len(stars_old.m)-Nbinaries_old)
vy_part = np.zeros(len(stars_old.m)-Nbinaries_old)
vz_part = np.zeros(len(stars_old.m)-Nbinaries_old)

#A binary particle has position, velocity and mass of the c.o.m. of the binary
for i in range(0,2*Nbinaries_old,2):
    m1_part[int(i/2)] = max(stars_old.m[i],stars_old.m[i+1])
    m2_part[int(i/2)] = min(stars_old.m[i],stars_old.m[i+1])

    x_part[int(i/2)] = (stars_old.x[i]*stars_old.m[i]+stars_old.x[int(i+1)]*stars_old.m[int(i+1)]) / (stars_old.m[i]+stars_old.m[int(i+1)])
    y_part[int(i/2)] = (stars_old.y[i]*stars_old.m[i]+stars_old.y[int(i+1)]*stars_old.m[int(i+1)]) / (stars_old.m[i]+stars_old.m[int(i+1)])
    z_part[int(i/2)] = (stars_old.z[i]*stars_old.m[i]+stars_old.z[int(i+1)]*stars_old.m[int(i+1)]) / (stars_old.m[i]+stars_old.m[int(i+1)])
    vx_part[int(i/2)] = (stars_old.vx[i]*stars_old.m[i]+stars_old.vx[int(i+1)]*stars_old.m[int(i+1)]) / (stars_old.m[i]+stars_old.m[int(i+1)])
    vy_part[int(i/2)] = (stars_old.vy[i]*stars_old.m[i]+stars_old.vy[int(i+1)]*stars_old.m[int(i+1)]) / (stars_old.m[i]+stars_old.m[int(i+1)])
    vz_part[int(i/2)] = (stars_old.vz[i]*stars_old.m[i]+stars_old.vz[int(i+1)]*stars_old.m[int(i+1)]) / (stars_old.m[i]+stars_old.m[int(i+1)])
    #print(i,i+1)
    #a single particle has position, velocity and mass of the single mass
for i in range(2*Nbinaries_old,len(stars_old.m),1):
    m1_part[i-Nbinaries_old] = stars_old.m[i]
    m2_part[i-Nbinaries_old] = 0.
    x_part[i-Nbinaries_old] = stars_old.x[i]
    y_part[i-Nbinaries_old] = stars_old.y[i]
    z_part[i-Nbinaries_old] = stars_old.z[i]
    vx_part[i-Nbinaries_old] = stars_old.vx[i]
    vy_part[i-Nbinaries_old] = stars_old.vy[i]
    vz_part[i-Nbinaries_old] = stars_old.vz[i]
    #print(i)
	

	
stars = cl.cluster(m1_part, x_part, y_part, z_part, vx_part, vy_part, vz_part)


if(par.change_imf==True):
    
    #We order the stars for decreasing primary mass

    star_matrix = np.stack((m1_part, x_part, y_part, z_part, vx_part, vy_part, vz_part), axis=-1)
    star_matrix = star_matrix[star_matrix[:, 0].argsort()]

    m1_part, x_part, y_part, z_part, vx_part, vy_part, vz_part = star_matrix[:,0], star_matrix[:,1], star_matrix[:,2], star_matrix[:,3], star_matrix[:,4], star_matrix[:,5], star_matrix[:,6]
    stars = cl.cluster(m1_part, x_part, y_part, z_part, vx_part, vy_part, vz_part)
    
    stars.generate_imf(M=len(m1_part), parameter="numb")
    m1_appo = np.sort(stars.m)
    stars.m = np.copy(m1_appo)

	
	
stars.generate_secondaries(m2=m2_part, P=np.zeros(len(m2_part)), ecc=np.zeros(len(m2_part)), import_binaries=True)


stars.binary_fraction()


Ekstars_part_old = stars.energies()

print( "Length scale (virial radius):", stars.v_radius)
print( "Mass scale (mean mass):", np.mean(stars.m))
print( "Time scale:", stars.t_scale)
print( "Velocity scale:", stars.v_scale)



print( "------------------------------")
print( "  Generate the new binaries   ") 
print( "------------------------------")

#stars.generate_imf(len(m1_part), parameter="numb")
stars.generate_secondaries()

stars.find_com(center=True)


stars.rescale_to_virial()



stars.binary_fraction()


stars.binaries.orbital_properties()

stars.binaries.orbits()


print("")

print( "Length scale (virial radius):", stars.v_radius)
print( "Mass scale (mean mass):", np.mean(stars.m))
print( "Time scale:", stars.t_scale)
print( "Velocity scale:", stars.v_scale)

print("\n")


#plots the distributions of binary parameters
# Lines 138-148: fix local norm to match the theoretical curve
ma= np.logspace(np.log10(sp.mmin),np.log10(sp.mMax),1000)
if sp.lognorm == False:
	if(sp.alpha1!=1):
    		norm1 = 1/(1-sp.alpha1)*(sp.mchange**(1-sp.alpha1)-sp.mmin**(1-sp.alpha1))	
	else:
    		norm1 = np.log(sp.mchange) - np.log(sp.mmin)
    
	if(sp.alpha2!=1):
    		norm2 = sp.continuity/(1-sp.alpha2)*(sp.mMax**(1-sp.alpha2)-sp.mchange**(1-sp.alpha2))
	else:
    		norm2 = np.log(sp.mMax) - np.log(sp.mchange)

	norm = norm1 + norm2  # this is fine for the theoretical curve expom

	expom= []
	for i in ma:
		if(i<sp.mchange):
			expom.append(np.power(i,-sp.alpha1)/norm)
		else:
			expom.append(np.power(i,-sp.alpha2)/norm*sp.continuity)
else:
    
    norm = np.log(sp.mMax/sp.mmin)

    expom = []

    for i in ma:
        expom.append(1./(i*norm))

expoe= []
expoq= []
expoP= []
expoax = []

if sp.bin_params == 's12':
    axmin = pow(((par.G*((10**sp.pmin_HM)*24.*60.*60.)**2.)*(2*0.8))/(4.*np.pi**2.*par.pcinkm**2.),1./3.)
    axMax = pow(((par.G*((10**sp.pMax_HM)*24.*60.*60.)**2.)*(2*sp.mMax))/(4.*np.pi**2.*par.pcinkm**2.),1./3.)
    normax = 3./4. * (axMax**(4./3.)-axmin**(4./3.))/(par.G /(4.*np.pi**2.)**1./3.)

    aq= np.linspace(sp.qmin_HM,sp.qMax_HM,1000)
    ae= np.linspace(sp.emin_HM,sp.eMax_HM,1000)
    aP= np.linspace(sp.pmin_HM,sp.pMax_HM,1000)
    aax = np.linspace(axmin,axMax,1000)

    expoq = pow(aq,sp.k_HM)/sp.normq_HM
    expoe = pow(ae,sp.eta_HM)/sp.norme_HM
    expoP = pow(aP,sp.pigr_HM)/sp.normp_HM

    aq_LM = np.linspace(sp.qmin_LM,sp.qMax_LM,1000)
    expoq_LM = pow(aq_LM,sp.k_LM)/sp.normq_LM
    ae_LM = np.linspace(sp.emin_LM,sp.eMax_LM,1000)
    expoe_LM = 2 * ae_LM / sp.norme_LM
    aP_LM = np.linspace(sp.pmean_LM - 4*sp.psigma_LM, sp.pmean_LM + 4*sp.psigma_LM, 1000)
    expoP_LM = sp.pnorm_LM * np.exp(-((aP_LM - sp.pmean_LM)**2) / (2 * sp.psigma_LM**2))
    
    # Medium-mass distributions (interpolation scheme)
    if sp.medium_interpolate == True:
        aq_MM = np.linspace(sp.qmin_MM, sp.qMax_MM, 1000)
        ae_MM = np.linspace(sp.emin_MM, sp.eMax_MM, 1000)
        aP_MM = np.linspace(sp.pmin_MM, sp.pMax_MM, 1000)
        
        expoq_MM = pow(aq_MM, sp.k_MM) / sp.normq_MM
        expoe_MM = 2 * ae_MM / sp.norme_MM  # Linear in e
        expoP_MM = pow(aP_MM, sp.pigr_MM) / sp.normp_MM  # Power law in log(P)

if sp.bin_params == 'sb13':
    aq= np.linspace(sp.qmin_HM,sp.qMax_HM,1000)
    ae= np.linspace(sp.emin_HM,sp.eMax_HM,1000)
    aP = np.linspace(sp.pmu_HM - 4*sp.psigma_HM, sp.pmu_HM + 4*sp.psigma_HM, 1000)
    
    expoq = pow(aq,sp.k_HM)/sp.normq_HM
    expoe = 2 * ae / sp.norme_HM
    expoP = sp.normp_HM * np.exp(-((aP - sp.pmu_HM)**2) / (2 * sp.psigma_HM**2))

    aq_LM = np.linspace(sp.qmin_LM,sp.qMax_LM,1000)
    expoq_LM = pow(aq_LM,sp.k_LM)/sp.normq_LM
    ae_LM = np.linspace(sp.emin_LM,sp.eMax_LM,1000)
    expoe_LM = 2 * ae_LM / sp.norme_LM
    aP_LM = np.linspace(sp.pmean_LM - 4*sp.psigma_LM, sp.pmean_LM + 4*sp.psigma_LM, 1000)
    expoP_LM = sp.pnorm_LM * np.exp(-((aP_LM - sp.pmean_LM)**2) / (2 * sp.psigma_LM**2))
    
    # Medium-mass distributions (interpolation scheme)
    if sp.medium_interpolate == True:
        aq_MM = np.linspace(sp.qmin_MM, sp.qMax_MM, 1000)
        ae_MM = np.linspace(sp.emin_MM, sp.eMax_MM, 1000)
        aP_MM = np.linspace(sp.pmin_MM, sp.pMax_MM, 1000)
        
        expoq_MM = pow(aq_MM, sp.k_MM) / sp.normq_MM
        expoe_MM = 2 * ae_MM / sp.norme_MM  # Linear in e
        expoP_MM = pow(aP_MM, sp.pigr_MM) / sp.normp_MM  # Power law in log(P)

        
#Binary mass ratio distribution
q_new = []
q_new_mass = []
q3_new = []

eq3 = []
if sp.medium_interpolate == True:
    aP_eq3 = np.concatenate((aP_LM, aP_MM, aP))
else:
    aP_eq3 = np.concatenate((aP_LM, aP))

aP_eq3 = sorted(aP_eq3)
for i in range(len(aP_eq3)):
    if(10.**aP_eq3[i]>2):
        eq3.append(1.-((10.**aP_eq3[i])/2.)**(-2./3.))
    else:
        eq3.append(0)
		

#Print the output
		

path = "./"

if(par.par_bin_file==True):
	binary_file = path + "binaries_new.dat"
	f= open(binary_file, 'w')

	P, e, a  = np.zeros(len(stars.m)), np.zeros(len(stars.m)),  np.zeros(len(stars.m))

	f.write("#m[MSun], x[pc], y[pc], z[pc], vx[km/s], vy[km/s], vz[km/s], P[days], ecc, smax[pc], index  \n")

	for i in range(len(stars.m)):
		if(stars.isbin[i]==1):
			P[i] = stars.binaries.P[i]
			e[i] = stars.binaries.ecc[i]
			a[i] = stars.binaries.smax[i]

			f.write("%.12f %.12f %.12f %.12f %.12f %.12f %.12f %.12f %.12f %.12f %.12f \n" % (stars.binaries.m1[i], stars.binaries.xb1[i], stars.binaries.yb1[i], stars.binaries.zb1[i], stars.binaries.vxb1[i], stars.binaries.vyb1[i], stars.binaries.vzb1[i], P[i], e[i], a[i], stars.index[i]))
			f.write("%.12f %.12f %.12f %.12f %.12f %.12f %.12f %.12f %.12f %.12f %.12f \n" % (stars.binaries.m2[i], stars.binaries.xb2[i], stars.binaries.yb2[i], stars.binaries.zb2[i], stars.binaries.vxb2[i], stars.binaries.vyb2[i], stars.binaries.vzb2[i], P[i], e[i], a[i], stars.index[i]))

		else:
			f.write("%.12f %.12f %.12f %.12f %.12f %.12f %.12f %.12f %.12f %.12f %.12f \n" % (stars.m[i], stars.x[i], stars.y[i], stars.z[i], stars.vx[i], stars.vy[i], stars.vz[i], P[i], e[i], a[i], stars.index[i]))
	f.close()




output_file = os.path.join(path, "dat.10")
parameter_file = os.path.join(path, "input")

if(par.verbose==True):

    print("--------- Nbody6++ ICs ----------")
    print("Parameter file name:", parameter_file)
    print("Output file name:", output_file) 


try:
    os.remove(output_file)
    os.remove(parameter_file)
except OSError:
    pass
    
fin = open(simnamepar, "rt")
fout = open(parameter_file, "wt")
neighbour_number = int(2*pow(len(stars.m),0.5))

if(par.binaries_yn==True):
    mean_new = round(np.mean(np.concatenate((stars.binaries.m1,stars.binaries.m2[stars.binaries.m2>0]))),16)
else:
    mean_new = round(np.mean(np.concatenate(stars.m)),16)
    
if(par.binaries_yn==True):
    for line in fin:
        line = line.replace(str(Nstars_old), str(len(stars.m))).replace("50 1 10",str(neighbour_number)+ " 1 1").replace(str(metal_star_old),str(sp.metal)).replace("1.00000000 8.00000000 1000000.0000 1.0E-02","0.10000000 0.80000000 1000000.0000 1.0E-02").replace(str(mean_old), str(mean_new)).replace("0 1 0 1 2 0 0 0 3 6","0 1 0 1 2 0 0 0 0 6").replace(" 6 ", " 10 ").replace("1.0E-3 1.0E-2 ", "1.0E-7 1.0E-4 ").replace("-6","0")
        tokens = line.split()
        if len(tokens) == 8 and tokens[0].startswith("2.35"):  # uniquely identify the target line
            tokens[3] = str(int(stars.binaries.Nbin))     # update Nbinaries safely
            line = ' '.join(tokens)
        fout.write(line + '\n')
else:
    for line in fin:
        line = line.replace(str(Nstars_old), str(len(stars.m)))\
                   .replace("50 1 10",str(neighbour_number)+ " 1 1")\
                   .replace(str(metal_star_old),str(sp.metal))\
                   .replace("1.00000000 8.00000000 1000000.0000 1.0E-02","0.10000000 0.80000000 1000000.0000 1.0E-02")\
                   .replace(str(mean_old), str(mean_new))\
                   .replace("0 1 0 1 2 0 0 0 3 6","0 1 0 1 2 0 0 0 0 6")\
                   .replace(" 6 ", " 10 ")\
                   .replace("1.0E-3 1.0E-2 ", "1.0E-7 1.0E-4 ")\
                   .replace("-6","0")
        tokens = line.split()
        if len(tokens) == 8 and tokens[0].startswith("2.35"):  # uniquely identify the target line
            tokens[3] = str(int(0))     # update Nbinaries safely
            line = ' '.join(tokens)
        fout.write(line + '\n')

fin.close()
fout.close()

# It prints on a formatted file, in n-body units (with M=Msun,L=pc), in the Nbody6 format
f = open(output_file, 'w')

for i in range(len(stars.m)):
    if(stars.isbin[i]==1):
        f.write("%.12f %.12f %.12f %.12f %.12f %.12f %.12f \n" % (stars.binaries.m1[i], stars.x[i]+stars.binaries.xb1[i], stars.y[i]+stars.binaries.yb1[i], stars.z[i]+stars.binaries.zb1[i], stars.vx[i]+stars.binaries.vxb1[i], stars.vy[i]+stars.binaries.vyb1[i], stars.vz[i]+stars.binaries.vzb1[i]))
        f.write("%.12f %.12f %.12f %.12f %.12f %.12f %.12f \n" % (stars.binaries.m2[i], stars.x[i]+stars.binaries.xb2[i], stars.y[i]+stars.binaries.yb2[i], stars.z[i]+stars.binaries.zb2[i], stars.vx[i]+stars.binaries.vxb2[i], stars.vy[i]+stars.binaries.vyb2[i], stars.vz[i]+stars.binaries.vzb2[i]))

for i in range(len(stars.m)):
    if(stars.isbin[i]==0):
        f.write("%.12f %.12f %.12f %.12f %.12f %.12f %.12f \n" % (stars.m[i], stars.x[i], stars.y[i], stars.z[i], stars.vx[i], stars.vy[i], stars.vz[i]))

f.close()


#Show the new binary distributions

if par.plotta == True:
    fig, axv = plt.subplots(3,3,figsize=(16,10))

    nbins = 12
    mask_bin = stars.binaries.m2 > 0
    
    # primary mass distribution
    axv[0][0].plot(ma,expom,label=f'{sp.imf_type}')
    binar=np.logspace( np.log10(0.99*sp.mmin), np.log10(1.01*sp.mMax), nbins)
    axv[0][0].set_xlim(0.9*sp.mmin,1.1*sp.mMax)
    axv[0][0].set_xscale('log')
    axv[0][0].set_yscale('log')
    axv[0][0].set_xlabel(r'm [$M_{\odot}$]', fontsize=18)
    axv[0][0].set_ylabel('N(m)', fontsize=18)
    axv[0][0].tick_params(length=6, labelsize=16)
    axv[0][0].hist(stars.binaries.m1,bins=binar, density=True,label='New', histtype="step", linestyle="dashed", linewidth=2.)
    axv[0][0].legend(fontsize=16)

    # mass-ratio distribution
    qmin, qmax = 0.99*min(stars.binaries.q[(mask_bin)]), 1.01*max(stars.binaries.q[(mask_bin)])

    axv[0][1].plot(aq,expoq,label=r'HM: $\kappa$='+str(sp.k_HM))
    axv[0][1].plot(aq_LM,expoq_LM,label=r'LM: $\kappa$='+str(sp.k_LM))
    if sp.medium_interpolate == True:
        axv[0][1].plot(aq_MM,expoq_MM,label=r'MM: $\kappa$='+str(sp.k_MM), linestyle='--')
        
    axv[0][1].hist(stars.binaries.q[(mask_bin)], bins=np.logspace(np.log10(qmin),np.log10(qmax), nbins), histtype='step', linewidth=2., density='True', linestyle="dashed", label=r'New')
    axv[0][1].legend(loc='upper left', fontsize=16)
    axv[0][1].tick_params(length=6, labelsize=16)    
    axv[0][1].set_xscale('log')
    axv[0][1].set_xlabel('q', fontsize=18)
    axv[0][1].set_ylabel(r'$q^\kappa$', fontsize=18)

    #eccentricity distribution
    axv[1][0].plot(ae,expoe,label=f'HM: {sp.bin_params}')
    axv[1][0].plot(ae_LM,expoe_LM,label=f'LM: R2010')
    if sp.medium_interpolate == True:
        axv[1][0].plot(ae_MM,expoe_MM,label=f'MM: K2007', linestyle='--')
    emin, emax = 0.99*min(stars.binaries.ecc[(mask_bin)]), 1.01*max(stars.binaries.ecc[(mask_bin)])
    axv[1][0].hist(stars.binaries.ecc[mask_bin],bins=np.logspace(np.log10(emin),np.log10(emax),nbins), histtype='step', linewidth=2., linestyle="dashed", density='True',label='New')
    axv[1][0].legend(fontsize=16)
    axv[1][0].tick_params(length=6, labelsize=16)    
    axv[1][0].set_xscale('log')
    axv[1][0].set_yscale('log')
    axv[1][0].set_xlabel('e', fontsize=18)
    axv[1][0].set_ylabel(r'$e^\eta$', fontsize=18)

    #periods distribution

    pmin, pmax = 0.99*min(stars.binaries.P[(mask_bin)]), 1.01*max(stars.binaries.P[(mask_bin)])

    axv[1][1].plot(aP,expoP,label=f'HM: {sp.bin_params}')
    if sp.medium_interpolate == True:
        axv[1][1].plot(aP_MM,expoP_MM,label=f'MM: K2007', linestyle='--')
    axv[1][1].hist(np.log10(stars.binaries.P[mask_bin]), bins=np.linspace(np.log10(pmin),np.log10(pmax), nbins), histtype='step', linewidth=2., linestyle="dashed", density='True',label='New')
    axv[1][1].legend(fontsize=16)
    axv[1][1].tick_params(length=6, labelsize=16)
    axv[1][1].set_xlabel(r'$Log(P) [days] $', fontsize=18)
    axv[1][1].set_ylabel(r'$[Log(P)]^\pi$', fontsize=18)

    #semi-major axes distribution
    axv[0][2].hist(stars.binaries.smax[mask_bin], bins=np.logspace(np.log10(min(stars.binaries.smax[mask_bin][stars.binaries.smax[mask_bin][:]>0])),np.log10(max(stars.binaries.smax[mask_bin][stars.binaries.smax[mask_bin][:]>0])), nbins), density = 'True', histtype='step', linewidth=2., linestyle="dashed", label='New')
    axv[0][2].tick_params(length=6, labelsize=16)
    axv[0][2].legend(fontsize=16)
    axv[0][2].set_xscale('log')
    axv[0][2].set_yscale('log')    
    axv[0][2].set_xlabel(r'a [pc]', fontsize=18)
    axv[0][2].set_ylabel(r'P(a)', fontsize=18)  

    # eq.3 Moe & Di Stefano 2017
    axv[1][2].plot(aP_eq3,eq3, color="Red",label="Eq. 3 (M&DS)")
    axv[1][2].scatter(np.log10(stars.binaries.P[mask_bin]),stars.binaries.ecc[mask_bin],label='New', s=1)
    axv[1][2].legend(fontsize=16)
    axv[1][2].set_xlabel(r'$Log(P) [days] $', fontsize=18)
    axv[1][2].set_ylabel(r'$e$', fontsize=18)

    # mass-dependent binary fraction (Moe & Di Stefano 2017)
    axv[2][0].errorbar(sp.mbin,stars.binaries.fbin, xerr=sp.merr, yerr=0, label='New',fmt='^')
    axv[2][0].legend(loc=4, fontsize=16)
    axv[2][0].set_xscale('log')
    axv[2][0].set_xlabel(r'm $[M_{\odot}]$', fontsize=18)
    axv[2][0].set_ylabel(r'$f_{bin}$', fontsize=18)
    axv[2][0].tick_params(length=6, labelsize=16)

    # mass-dependent triple fraction (Moe & Di Stefano 2017)
    axv[2][1].errorbar(sp.mbin,stars.binaries.ftrip, xerr=sp.merr, yerr=0, label='New',fmt='^')
    axv[2][1].legend(loc=4, fontsize=16)
    axv[2][1].set_xscale('log')
    axv[2][1].set_xlabel(r'm $[M_{\odot}]$', fontsize=18)
    axv[2][1].set_ylabel(r'$f_{trip}$', fontsize=18)
    axv[2][1].tick_params(length=6, labelsize=16)

    # mass-dependent fraction of all multiples 
    axv[2][2].errorbar(sp.mbin,stars.binaries.fbin+stars.binaries.ftrip, xerr=sp.merr, yerr=0, label='New',fmt='^')
    axv[2][2].legend(loc=4, fontsize=16)
    axv[2][2].set_xscale('log')
    axv[2][2].set_xlabel(r'm $[M_{\odot}]$', fontsize=18)
    axv[2][2].set_ylabel(r'$f_{mult}$', fontsize=18)
    axv[2][2].tick_params(length=6, labelsize=16)

    plt.tight_layout()
    plt.savefig(f'new_binary_distributions_IMF{sp.imf_type}_{sp.bin_params}_medium{sp.medium_interpolate}_triples{par.multiples}.pdf')
    plt.close()
