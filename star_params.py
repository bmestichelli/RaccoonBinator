from parameters import *
import numpy as np
import scipy.stats as stats
###############################
#     Stellar parameters      #
###############################

metal = 0.02 #necessary for stellar radius only (not valid for Pop. III stars) 
bin_params = 's12' #orbital parameters of massive stars 
imf_type = 'kroupa'
medium_interpolate = True # interpolate the binary orbital parameters between 2 and 15 Msun with distributions from Kouwenhoven et al. 2007; recommended False for Pop. III

####    IMF parameters    ####
mmin = 0.08 # minimum mass in solar masses 
mMax= 300. # maximum mass in solar masses 
 
lognorm = False  # default; overridden below for log-flat
 
if imf_type == 'log-flat':
        # IMF: xi(M) ∝ M^{-1}
        # recommended for Pop. III (e.g. Stacy & Bromm 2013)
        alpha = 1
        lognorm = True
        
if imf_type == 'kroupa':
        # Kroupa (2001)
        # IMF : xi(M) ∝ M^{-1.3} for M<0.5 Msun and M^{-2.3} for M>=0.5 Msun 
        alpha1 = 1.3
        alpha2 = 2.3
        mchange = 0.5
        continuity=mchange**(alpha2-alpha1)
        norm1= 1./(1.-alpha1)*(mchange**(1.-alpha1)-mmin**(1.-alpha1)) #normalization of the first part
        norm2= continuity/(1.-alpha2)*(mMax**(1.-alpha2)-mchange**(1.-alpha2)) #normalization of the second part
        norm = norm1+norm2
 
if imf_type == 'top-heavy':
        # IMF: xi(M) ∝ M^{-alpha} * exp(-Mcut^2 / M^2)
        # Approximated as a broken power law around Mcut
        # recommended for Pop. III (e.g. Jaacks et al. 2019)
        alpha1 = 0.74   # below Mcut (effective slope near turnover
        alpha2 = 0.17  # above Mcut (true asymptotic slope)
        mchange = 20.0
        continuity = mchange**(alpha2 - alpha1)
        norm1 = 1./(1. - alpha1) * (mchange**(1. - alpha1) - mmin**(1. - alpha1))
        norm2 = continuity / (1. - alpha2) * (mMax**(1. - alpha2) - mchange**(1. - alpha2))
        norm = norm1 + norm2

###############################
#      Binary parameters      #
###############################

####   Binary fraction   #####

# for low-mass stars 
bin_fract0 = 0.2 #binary fraction of the first bin (for m<0.8 MSun) (not in Moe & Di Stefano 2017)

#binary fraction parameters (for m>0.8 MSun) (Moe & Di Stefano 2017)
nbin = 6
mbin=[0.44,1.4, 3.5, 7, 12.5, (mMax + 16.)/2.] #Mass bins centers
merr=[0.36,0.6,1.5,2.0,3.5, (mMax-16.)/2] #Mass bin uncertainties

fbinv = [0.4,0.59, 0.76, 0.84, 0.94] #binary fractions following Moe & Di Stefano (2017) (for m>0.8 MSun)
fbinvb = [0.30,0.37, 0.36, 0.32, 0.21] #binary fractions (only binaries) following Moe & Di Stefano (2017) (for m>0.8 MSun)
fbinvt = [0.10,0.22, 0.40, 0.52, 0.73] #binary fractions (only multiples) following Moe & Di Stefano (2017) (for m>0.8 MSun)


####   Orbital properties   #####

######   Massive stars   #######
if bin_params == 's12':
        #mass ratio distribution parameters (Sana 2012) 
        qmin_HM = 0.01
        qMax_HM = 1.
        k_HM = -0.1

        #triple
        qmin3_HM = 0.01
        qMax3_HM = 1.
        k3_HM = 0.0

        normq_HM = 1./(1.+k_HM) * (qMax_HM**(1.+k_HM)-qmin_HM**(1.+k_HM))
        normq3_HM = 1./(1.+k3_HM) * (qMax_HM**(1.+k3_HM)-qmin_HM**(1.+k3_HM))
	
        #eccentricity distribution parameters (Sana 2012)
        emin_HM = 10.**(-5.)
        eMax_HM = 0.9999
        eta_HM = -0.45
        norme_HM = 1./(1.+eta_HM) * (eMax_HM**(1.+eta_HM)-emin_HM**(1.+eta_HM))
        
        #period distribution parameters (Sana 2012)
        pmin_HM = 0.15
        pMax_HM = 5.5
        pigr_HM = -0.55
        normp_HM = 1./(1.+pigr_HM) * (pMax_HM**(1.+pigr_HM)-pmin_HM**(1.+pigr_HM))

if bin_params == 'sb13': #recommended for Pop. III 
        #mass ratio distribution parameters (Stacy & Bromm 2013)                              
        qmin_HM = 0.01
        qMax_HM = 1.
        k_HM = -0.55

        #triple
        qmin3_HM = 0.01
        qMax3_HM = 1.
        k3_HM = 0.0

        normq_HM = 1./(1.+k_HM) * (qMax_HM**(1.+k_HM)-qmin_HM**(1.+k_HM))
        normq3_HM = 1./(1.+k3_HM) * (qMax_HM**(1.+k3_HM)-qmin_HM**(1.+k3_HM))

        #eccentricity distribution parameters (Stacy & Bromm 2013)
        emin_HM = 10.**(-5.)
        eMax_HM = 0.9999
        norme_HM = 1. / (eMax_HM**2 - emin_HM**2)

        #period distribution parameters (Stacy & Bromm 2012)
        pmu_HM = 5.5 
        psigma_HM = 0.85 
        normp_HM = 1. / (np.sqrt(2*np.pi) * psigma_HM)

######   Low-mass stars   #######

# if you want to use Sana et al. 2012 / Stacy & Bromm 2013 across all the mass spectrum, put m_thres_low = m_thres_high = mmin*0.999 (recommended for Pop. III)

if medium_interpolate == False: 
        m_thres_low = 2. # threshold mass above which we use prescriptions by Raghavan et al. 2010
        m_thres_high = 2. # threshold mass above which we use prescriptions by Sana et al. 2012 / Stacy & Bromm 2013
else:
        m_thres_low = 2.
        m_thres_high = 15.
        
#mass ratio distribution parameters (Raghavan et al. 2010) 
qmin_LM = 0.1
qMax_LM = 1.
k_LM = 0.

#triple
qmin3_LM = 0.1
qMax3_LM = 1.
k3_LM = 0.0

normq_LM = 1./(1.+k_LM) * (qMax_LM**(1.+k_LM)-qmin_LM**(1.+k_LM))
normq3_LM = 1./(1.+k3_LM) * (qMax_LM**(1.+k3_LM)-qmin_LM**(1.+k3_LM))
	
#eccentricity distribution parameters (Raghavan et al. 2010)  
emin_LM = 10.**(-5.)
eMax_LM = 0.9999
eta_LM = 0.0
norme_LM = 1./(1.+eta_LM) * (eMax_LM**(1.+eta_LM)-emin_LM**(1.+eta_LM))

#period distribution parameters (Raghavan et al. 2010)  
pmean_LM = 5.03
psigma_LM = 2.28
pmin_LM = 0.1
pMax_LM = 10.

alpha = (pmin_LM - pmean_LM) / psigma_LM
beta = (pMax_LM - pmean_LM) / psigma_LM
pnorm_LM = stats.norm.cdf(beta) - stats.norm.cdf(alpha)


######   Medium-mass stars   #######

#mass ratio distribution parameters (Kouwenhoven et al. 2007)
qmin_MM = 0.1
qMax_MM = 1.
k_MM = -0.4

#triple
qmin3_MM = 0.01
qMax3_MM = 1.
k3_MM = 0.0

normq3_MM = 1./(1.+k3_MM) * (qMax_MM**(1.+k3_MM)-qmin_MM**(1.+k3_MM))
normq_MM = 1./(1.+k_MM) * (qMax_MM**(1.+k_MM)-qmin_MM**(1.+k_MM))

#eccentricity distribution parameters (Kouwenhoven et al. 2007)
emin_MM = 10.**(-5.)
eMax_MM = 0.9999
norme_MM = 1. / (eMax_MM**2 - emin_MM**2)

#period distribution parameters (Kouwenhoven et al. 2007)
pmin_MM = 0.15
pMax_MM = 9.
pigr_MM = 0.
normp_MM = 1./(1.+pigr_MM) * (pMax_MM**(1.+pigr_MM)-pmin_MM**(1.+pigr_MM))


#Moe & Di Stefano binary fraction (including multiplets)
def binary_function(m):
	if(m <= mbin[0]+merr[0]):
		return 0, bin_fract0
	elif(mbin[0]+merr[0] <= m < mbin[1]+merr[1]):
		return 1, fbinv[0]
	elif(mbin[1]+merr[1] <= m < mbin[2]+merr[2]):
		return 2, fbinv[1]
	elif(mbin[2]+merr[2] <= m < mbin[3]+merr[3]):
		return 3, fbinv[2]
	elif(mbin[3]+merr[3] <= m < mbin[4]+merr[4]):
		return 4, fbinv[3]
	elif(m >= mbin[4]+merr[4]):
		return 5, fbinv[4]

#Moe & Di Stefano binary fraction (no multiplets)
def binary_function_no_multiples(m):
    if(m <= mbin[0]+merr[0]):
        if(multiples==True):
            return 0, bin_fract0
        else:
            return 0, bin_fract0
    elif(mbin[0]+merr[0] <= m < mbin[1]+merr[1]):
        if(multiples==True):
            return 1, fbinvb[0]
        else:
            return 1, fbinv[0]
    elif(mbin[1]+merr[1] <= m < mbin[2]+merr[2]):
        if(multiples==True):
            return 2, fbinvb[1]
        else:
            return 2, fbinv[1]
    elif(mbin[2]+merr[2] <= m < mbin[3]+merr[3]):
        if(multiples==True):
            return 3, fbinvb[2]
        else:
             return 3, fbinv[2]
    elif(mbin[3]+merr[3] <= m < mbin[4]+merr[4]):
        if(multiples==True):
            return 4, fbinvb[3]
        else:
            return 4, fbinv[3]
    elif(m >= mbin[4]+merr[4]):
        if(multiples==True):
            return 5, fbinvb[4]
        else:
            return 5, fbinv[4]


#Functions to calculate stellar radius
def rad_par_calculator(Z, a, b, c, d, e):
	return a + b* np.log10(Z/Zsun) + c* np.power(np.log10(Z/Zsun),2.) + d* np.power(np.log10(Z/Zsun),3.) + e*np.power(np.log10(Z/Zsun),4.)

def star_radius(m_star, Z):
	atheta = 1.71535900
	btheta = 0.62246212
	ctheta = -0.92557761
	dtheta = -1.16996966
	etheta = -0.30631491
	theta = rad_par_calculator(Z, atheta, btheta, ctheta, dtheta, etheta)

	aelle = 6.59778800
	belle = -0.42450044
	celle = -12.13339427
	delle = -10.73509484
	eelle =	-2.51487077
	elle = rad_par_calculator(Z, aelle, belle, celle, delle, eelle)

	akappa = 10.08855000
	bkappa = -7.11727086
	ckappa = -31.67119479
	dkappa = -24.24848322
	ekappa = -5.33608972	
	kappa = rad_par_calculator(Z, akappa, bkappa, ckappa, dkappa, ekappa)

	allambda = 1.01249500
	bllambda = 0.32699690
	cllambda = -0.00923418
	dllambda = -0.03876858
	ellambda = -0.00412750
	llambda = rad_par_calculator(Z, allambda, bllambda, cllambda, dllambda, ellambda)

	amu = 0.07490166
	bmu = 0.02410413
	cmu = 0.07233664
	dmu = 0.03040467
	emu = 0.00197741
	mu = rad_par_calculator(Z, amu, bmu, cmu, dmu, emu)

	nu = 0.01077422
	
	acsi = 3.08223400
	bcsi = 0.94472050
	ccsi = -2.15200882
	dcsi = -2.49219496
	ecsi = -0.63848738
	csi = rad_par_calculator(Z, acsi, bcsi, ccsi, dcsi, ecsi)

	ao = 17.84778000
	bo = -7.45345690
	co = -48.96066856
	do = -40.05386135
	eo = -9.09331816
	o = rad_par_calculator(Z, ao, bo, co, do, eo)

	api = 0.00022582
	bpi = -0.00186899
	cpi = 0.00388783
	dpi = 0.00142402
	epi = -0.00007671
	pi = rad_par_calculator(Z, api, bpi, cpi, dpi, epi)

	Radius = (theta* np.power(m_star, 2.5) + elle* np.power(m_star, 6.5) + kappa * np.power(m_star, 11.) + llambda* np.power(m_star, 19.) + mu* np.power(m_star, 19.5)) / (nu + csi* np.power(m_star, 2.) + o* np.power(m_star, 8.5) + np.power(m_star, 18.5) + pi* np.power(m_star, 19.5))
	
	return Radius #*2.255*10.**(-8.) 

        


