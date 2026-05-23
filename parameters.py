import numpy as np

binaries_yn = True

plotta = True #plot stellar and binary distributions
par_bin_file=True #print file with new orbital parameters ("binaries_new.dat")

change_imf = False
verbose = True


#######################################################################################################
#######################################################################################################
#If cluster created from mcluster output

input_dat = "test.dat.10" #Mcluster output to recombine
input_file_name = "test.input" #Mcluster input file to import



#######################################################################################################
#######################################################################################################

#constants
#G=0.00449850214 #gravitational constant in astronomical units [pc]*[MSun]**(-1)*[km/s]**2 (the same as Mcluster)
G = 0.004499 # gravitational constant in pc^3 /Msun /Myr^2
Zsun = 0.02 #solar metallicity
pcinkm=3.08567758130573*10.**13. #parsec in kilometers


au2km = 1.495978707e+8
yr2sec = 3.15576e+7
G_s_kg_m = 6.6743e-11
GMSun = 1.3271244e+20; #///< Nominal solar mass parameter, m^3 s^-2,
MSun_kg = GMSun / G_s_kg_m

G_yr_msun_au = G_s_kg_m * ((yr2sec*yr2sec) * MSun_kg / (au2km*au2km*au2km) * 1e-9); #///< G in yr^-2, msun^-1, au^3



