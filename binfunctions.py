import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as pl
import matplotlib.colors as colors

from paramters_old import all
import star_params as sp

def anomaly(x,ecc,E):
    g = E - ecc * np.sin(E) - x  # x is random 0,2P; g is function I must find the zeroes
    return g

def bisec(x,ecc,tol):

    E1=1.0
    E2=100.

    g2=100.
    
    while(abs(g2)>tol):
        delta=0.5*(E2+E1) 
        g1 = anomaly(x,ecc,E1)
        g2 = anomaly(x,ecc,E2)


        if(np.sign(g1)== np.sign(g2)): #if g1 and g2 have same sign search other points
            if(np.sign(g1)==1):
                if((g1)>(g2)):
                    E1 = E2
                    E2 = E2 + delta
                else:
                    E2 = E1
                    E1 = E1 - delta
	  
	
            else:
                if((g1)<(g2)):
                    E1 = E2
                    E2 = E2 + delta
                    
                else:
                    E2 = E1
                    E1 = E1 - delta
	             
	        
                
        else:
            E3 = (E2 + E1)*0.5
            g3 = anomaly(x,ecc,E3)
            if(np.sign(g3)==np.sign(g1)):
                E1=E3
                #E2=E2
            else:
                #E1=E1
                E2=E3
	    
            
    return ((E1+E2)*0.5)

def rad_par_calculator(Z, a, b, c, d, e,):
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
	return Radius *2.255*10.**(-8.) 




def generate_binaries(stars, verbose=True):
    #divide the stars in mass intervals to reproduce the binary function
    Nstars = len(stars)

    binstot= np.empty(nbin)
    massbinvector = []

    #evaluate how many stars each bin contains
    for j in range(nbin):
        ntotbin = 0
        appomass = []
        for i in range(len(stars)):
            if(stars[i]>=mbin[j]-merr[j] and stars[i]< mbin[j]+merr[j]):
                ntotbin+=1
                appomass.append(stars[i])			
        binstot[j]=ntotbin
        massbinvector.append(appomass)

    #match the stars in order to reproduce Sana (2012) mass ratio q=m2/m1

    #parameters for Sana (2012) mass ratio 
    q = []
    starsdef = [] 
    binary_mass = []
    bindex= []
    counter=0
    m1binvect = np.zeros(nbin)	
    m1binvectbin = np.zeros(nbin)	
    p=nbin-1 #bin index
    m1binmax = binary_function(mbin[-1])[1] #maximum number of binary fraction for each bin
    starcounter=0
    numberbinariesinbin = np.zeros(nbin)
    numbertripletsinbin = np.zeros(nbin)
    tripindex=0
    trip0=0

    if(binary_function(mbin[0])[1]==0):
        nbmin=1
    else:
        nbmin=0

    while(p>-1):
        m1binmax = binary_function(mbin[p])[1]
        m1binmax_no_mult = binary_function_no_multiples(mbin[p])[1]
        
        while(m1binvect[p]<m1binmax and len(massbinvector[p])>0):
            m1= np.random.choice(massbinvector[p]) #randomly pick the primary star from the selected bin
            nb,m1binmax = binary_function(m1)
            while(m1==min(stars)):
                m1= np.random.choice(massbinvector[p]) #avoid picking the minimum mass star (only in the last bin)

            massbinvector[p].remove(m1) #remove the picked star (in order not to pick it again)

            #generate the second star according to Sana (2012) prescription
            x = float(np.random.uniform(0.,1.)) 
            appo =((1. + sp.k_HM)*sp.normq_HM*x + sp.qmin_HM**(1.+sp.k_HM))**(1./(1.+sp.k_HM))	
            m2th = m1*appo #get the theoretical mass from m2
            m2=  stars[np.argsort(np.abs(stars-m2th))[0]]  #get the secondary star as the star closest to the theoretical mass

            appoindex=1
            while(m2 >= m1): #if the secondary mass coincides with or is bigger than the first mass, take the second, third, fouth... closest star
                m2=  stars[np.argsort(np.abs(stars-m2th))[appoindex]]
                appoindex+=1

            if(abs(m2-m2th)/m2th>0.1): #check if the secondary star violates Sana (2012) rule by more than 10%
                counter+=1		

            starsdef.append(m1)			#append the binary stars to the vectors
            q.append(m2)
            binary_mass.append(m1+m2)
            bindex.append(1)			#add an index to the binary systems
            starcounter+=1
            stars = np.delete(stars, np.where(stars == m1)[0][0])
            stars = np.delete(stars, np.where(stars == m2)[0][0])

            nb2= binary_function(m2)[0]
            #if(nb2==nb):
                #print "nb2"
            if m2 in massbinvector[nb2]:
                massbinvector[nb2].remove(m2)
            elif m2 in massbinvector[nb2-1]:
                massbinvector[nb2-1].remove(m2)
            elif m2 in massbinvector[nb2+1]:
                massbinvector[nb2+1].remove(m2)
            numberbinariesinbin[nb] = numberbinariesinbin[nb] + 1
            binstot[nb2] = binstot[nb2] - 1
            m1binvect[nb] = (numberbinariesinbin[nb]+numbertripletsinbin[nb])/float(len(massbinvector[nb])+starcounter)

            m1binvectbin[nb] = (numberbinariesinbin[nb])/float(len(massbinvector[nb])+starcounter)

            if(m1binvectbin[p]>m1binmax_no_mult and p!=0 and multiples==True):
                #print "Triplet"
                if(tripindex==0):
                    triplimit=len(starsdef)-1
                    
                x = float(np.random.uniform(0.,1.)) 
                appo =((1. + sp.k3_HM)*sp.normq3_HM*x + sp.qmin_HM**(1.+sp.k3_HM))**(1./(1.+sp.k3_HM))

                m3th = m1*appo #get the theoretical mass from m2
                m3=  stars[np.argsort(np.abs(stars-m3th))[0]]  #get the secondary star as the star closest to the theoretical mass

                appoindex=1
                while(m3 >= m1): #if the secondary mass coincides with or is bigger than the first mass, take the second, third, fouth... closest star
                    m3=  stars[np.argsort(np.abs(stars-m3th))[appoindex]]
                    appoindex+=1

                starsdef.append(m3)			#append the binary stars to the vectors
                q.append(0.)
                binary_mass.append(m3)
                bindex.append(-m1)			#add an index to the binary systems	

                nb3= binary_function(m3)[0]
                massbinvector[nb3].remove(m3)

                binstot[nb3] = binstot[nb3] - 1
                stars = np.delete(stars, np.where(stars == m3)[0][0])
                numberbinariesinbin[nb] = numberbinariesinbin[nb] - 1
                numbertripletsinbin[nb] = numbertripletsinbin[nb] + 1
                m1binvect[nb] = (numberbinariesinbin[nb]+numbertripletsinbin[nb])/float(len(massbinvector[nb])+starcounter)
                m1binvectbin[nb] = (numberbinariesinbin[nb])/float(len(massbinvector[nb])+starcounter)

                if(nb2==nb and nb!=nbmin): #if the secondary star is in the same mass bin as the primary star, generate another triple to avoid that the binary fraction increases to much beacuse a star in that mass bin has been removed
                    
                    x = float(np.random.uniform(0.,1.)) 
                    appo =((1. + sp.k3_HM)*sp.normq3_HM*x + sp.qmin_HM**(1.+sp.k3_HM))**(1./(1.+sp.k3_HM))
                    
                    m1=starsdef[tripindex+trip0]
                    while(bindex[tripindex+trip0]<0 and -starsdef[tripindex+trip0] in bindex and tripindex<triplimit):
                        tripindex+=1
                        m1=starsdef[tripindex+trip0]

                    if(-starsdef[tripindex+trip0] not in bindex):
                        m3th = m1*appo #get the theoretical mass from m3
                        m3=  stars[np.argsort(np.abs(stars-m3th))[0]]  #get the secondary star as the star closest to the theoretical mass
                        nb3b= binary_function(m3)[0]

                        while(nb3b==nb):
                            x = float(np.random.uniform(0.,1.)) 
                            appo =((1. + sp.k3_HM)*sp.normq3_HM*x + sp.qmin_HM**(1.+sp.k3_HM))**(1./(1.+sp.k3_HM))
                            m1=starsdef[tripindex+trip0]

                            while(bindex[tripindex+trip0]<0 and -starsdef[tripindex+trip0] in bindex and tripindex<triplimit):
                                tripindex+=1
                                m1=starsdef[tripindex+trip0]
                            m3th = m1*appo #get the theoretical mass from m2
                            m3=  stars[np.argsort(np.abs(stars-m3th))[0]]  #get the secondary star as the star closest to the theoretical mass
                            nb3b= binary_function(m3)[0]

                        appoindex=1
                        while(m3 >= m1): #if the secondary mass coincides with or is bigger than the first mass, take the second, third, fouth... closest star
                            m3=  stars[np.argsort(np.abs(stars-m3th))[appoindex]]
                            appoindex+=1

                        starsdef.append(m3)			#append the binary stars to the vectors
                        q.append(0.)
                        binary_mass.append(m3)
                        bindex.append(-m1)			#add an index to the binary systems

                        nb3= binary_function(m3)[0]
                        massbinvector[nb3b].remove(m3)


                        binstot[nb3b] = binstot[nb3b] - 1
                        stars = np.delete(stars, np.where(stars == m3)[0][0])
                        numberbinariesinbin[nb] = numberbinariesinbin[nb] - 1
                        numbertripletsinbin[nb] = numbertripletsinbin[nb] + 1
                        m1binvect[nb] = (numberbinariesinbin[nb]+numbertripletsinbin[nb])/float(len(massbinvector[nb])+starcounter)
                        m1binvectbin[nb] = (numberbinariesinbin[nb])/float(len(massbinvector[nb])+starcounter)
                        tripindex+=1
                    #print(m1binvectbin[p])


                while(nb3==nb and tripindex<=triplimit and m1binvect[p]<m1binmax and m1binvectbin[p]>m1binmax_no_mult):
                    #print "nb3"
                    x = float(np.random.uniform(0.,1.)) 
                    appo =((1. + sp.k3_HM)*sp.normq_HM*x + sp.qmin_HM**(1.+sp.k3_HM))**(1./(1.+sp.k3_HM))
                    m1=starsdef[tripindex+trip0]
                    while(bindex[tripindex+trip0]<0 and -starsdef[tripindex+trip0] in bindex and tripindex<triplimit):
                        tripindex+=1
                        m1=starsdef[tripindex+trip0]
                        
                    if(-starsdef[tripindex+trip0] not in bindex):    
                        m3th = m1*appo #get the theoretical mass from m2
                        m3=  stars[np.argsort(np.abs(stars-m3th))[0]]  #get the secondary star as the star closest to the theoretical mass

                        appoindex=1
                        while(m3 >= m1): #if the secondary mass coincides with or is bigger than the first mass, take the second, third, fouth... closest star
                            m3=  stars[np.argsort(np.abs(stars-m3th))[appoindex]]
                            appoindex+=1

                        starsdef.append(m3)			#append the binary stars to the vectors
                        q.append(0.)
                        binary_mass.append(m3)
                        bindex.append(-m1)			#add an index to the binary systems

                        nb3= binary_function(m3)[0]
                        massbinvector[nb3].remove(m3)


                        binstot[nb3] = binstot[nb3] - 1
                        stars = np.delete(stars, np.where(stars == m3)[0][0])
                        numberbinariesinbin[nb] = numberbinariesinbin[nb] - 1
                        numbertripletsinbin[nb] = numbertripletsinbin[nb] + 1
                        m1binvect[nb] = (numberbinariesinbin[nb]+numbertripletsinbin[nb])/float(len(massbinvector[nb])+starcounter)
                        m1binvectbin[nb] = (numberbinariesinbin[nb])/float(len(massbinvector[nb])+starcounter)
                        tripindex+=1
                    
                    
        if(verbose==True):
            print("Completed: bin", p+1)
            print(m1binvect.round(4))
        starcounter =0
        p+=-1
        trip0 = len(starsdef)
        tripindex=0


    for i in range(len(stars)):
        starsdef.append(stars[i])			
        q.append(0.)
        binary_mass.append(stars[i])
        bindex.append(0)

    binstotdef = np.empty(nbin)

    for j in range(nbin):
        ntotbin = 0
        for i in range(len(starsdef)):
            if(starsdef[i]>=mbin[j]-merr[j] and starsdef[i]< mbin[j]+merr[j] and bindex[i]>=0):
                ntotbin+=1	
        binstotdef[j]=ntotbin

    Nsingles=0
    Nbinaries=0
    Ntriplets=0
    for i in range(len(bindex)):
        if(bindex[i]==0):
            Nsingles+=1
        elif(bindex[i]==1):
            Nbinaries+=1
        elif(bindex[i]<0):
            Ntriplets+=1
            Nbinaries+=-1

    Nparticles = Nsingles+Nbinaries+Ntriplets	

    if(verbose==True):
        print("\n")
        print("Total mass:", round(sum(starsdef)+sum(q),4))
        print("Total number of particles (single stars+binary systems):",  Nparticles)	
        print("Total number of stars:",  Nsingles+2*Nbinaries+3*Ntriplets)

        print("Total number of single stars:",  Nsingles)
        print("Total number of binary systems:", Nbinaries)	
        print("Total number of triple systems:", Ntriplets)	

        print("Total percentage of multiple systems:", round(100.*float(Nbinaries+Ntriplets)/float(Nparticles+0.000001),4),"%")
        print("Total percentage of binaries:", round(100.*float(Nbinaries)/float(Nparticles+0.000001),4),"%")
        print("Total percentage of triplets:", round(100.*float(Ntriplets)/float(Nparticles+0.000001),4),"%")

        print("Percentage of binaries (only stars with m>0.8 Msun )",round(100.*float(Nbinaries-numberbinariesinbin[0])/float(Nparticles-binstot[0]+0.000001),4),"%")
        print("Percentage of triplets (only stars with m>0.8 Msun )",round(100.*float(Ntriplets-numbertripletsinbin[0])/float(Nparticles-binstot[0]+0.000001),4),"%")


        print("Number of stars per bin:",  binstotdef)	
        print("Number of multiplets per bin:", numberbinariesinbin+numbertripletsinbin)
        print("Number of binaries per bin:", numberbinariesinbin)
        print("Number of triplets per bin:", numbertripletsinbin)
        print("Fraction of multiplets per bin:", np.divide(numberbinariesinbin+numbertripletsinbin,binstotdef+0.000001).round(4))
        print("Fraction of binaries per bin:", np.divide(numberbinariesinbin,binstotdef+0.000001).round(4))
        print("Fraction of triplets per bin:", np.divide(numbertripletsinbin,binstotdef+0.000001).round(4))
        print("Number of binaries theoretically violating Sana (2012) (>10%):", counter)
        print("\n")


    starmatrix = np.column_stack((binary_mass,starsdef, q, bindex))
    binary_mass = np.sort(binary_mass[:])
    starmatrix = starmatrix[np.argsort(starmatrix[:,0])]


    #print(binary_mass)
    #print(starmatrix)
    return binary_mass, starmatrix

def binary_distributions(ind_stars, m_stars, x_stars, y_stars, z_stars, vx_stars, vy_stars, vz_stars, R_stars):
	
	num_binaries=0

	for i in range(len(ind_stars)-1):
		if(ind_stars[i]== ind_stars[i+1]):
			num_binaries+=1

	#set the eccentricity of binaries
	
	m1 = np.zeros(num_binaries)
	m2 = np.zeros(num_binaries)
	
	j=0

	for i in range(len(ind_stars)-1):
		if(ind_stars[i]== ind_stars[i+1]):
			m1[j]= m_stars[i]
			m2[j]= m_stars[i+1]
			j+=1

	e = np.zeros(num_binaries) # Array of eccentricities

	for i in range(num_binaries): 
		if sp.bin_params == 's12':
			x = float(np.random.uniform(0.,1.)) 
			appo =((1. + sp.eta_HM)*sp.norme_HM*x + sp.emin_HM**(1.+sp.eta_HM))**(1./(1.+sp.eta_HM))
			e[i] = appo
		if sp.bin_params == 'sb13':
			x = float(np.random.uniform(0.,1.))
			appo = np.sqrt(sp.emin_HM**2 + x * (sp.eMax_HM**2 - sp.emin_HM**2))
			e[i] = appo
	# Stack sorted eccentricities and make space for periods in the array of masses	
	binaries = np.column_stack((m1,m2, e)) # 0: m_1; 1: m_2; 2: e
	binaries = binaries[binaries[:,-1].argsort()[::-1]] # Sort binaries with decreasing eccentricity
	binaries = np.column_stack((binaries, np.zeros(num_binaries))) # 0: m_1; 1: m_2; 2: e; 3: P

	# Generate periods
	P = np.zeros(num_binaries) # Array of periods

	for i in range(num_binaries): 		
		if sp.bin_params == 's12':
			x = float(np.random.uniform(0.,1.)) 
			appo =((1. + sp.pigr_HM)*sp.normp_HM*x + sp.pmin_HM**(1.+sp.pigr_HM))**(1./(1.+sp.pigr_HM))
			P[i] = 10.**appo 
		if sp.bin_params == 'sb13':
                        x = np.random.normal(sp.pmu_HM, sp.psigma_HM)
			P[i] = 10.**x

	# Sort periods in decreasing order
	P = P[P.argsort()[::-1]]
    
    # Let's assign a period to each binary according to a physical criteria
	"""To choose the proper period for a binary, you can either use the criteria given 
    by eq.3 in Moe & Di Stefano 2017 or check if there's a collision. 
    Please uncomment just one of the following two rows according to the criteria you want"""    
    #P_min = lambda a_min, m1, m2 : pow((a_min**3*(2*np.pi*pc_to_km)**2) / ((m1+m2)*G*(24*60*60)**2), 1/2) # Min P to just avoid collisions
	P_min = lambda e : 2. * (1.-e)**(-3./2.) # Eq.3 in Moe & Di Stefano 2017
    
	P_counter = 0 # Number of periods below P_min
	a_appo = np.zeros(num_binaries) # Array of semi-major axes

    # Cycle throug binaries
	for i in range(num_binaries): 

        # Get values for the current binary
		m1 = binaries[i,0]
		m2 = binaries[i,1]
		e = binaries[i,2]
				
		"""Uncomment if you want to prevent collisions"""
		#Rtot = star_radius(m1, metallicity) + star_radius(m2, metallicity) # Sum of radii
		#dmin_to_Rtot = 1.2 # Ratio of min acceptable value of pericenter distance to Rtot
		#a_min = Rtot * dmin_to_Rtot / (1-e) # 
		#
		#good_Ps = P[P >= P_min(a_min, m1, m2)]

		"""Uncomment if you want to enforce eq.3 from Moe & Di Stefano 2017"""
		good_Ps = P[P >= P_min(e)]
				
		# Select one random period out of those greater than P_min
		if good_Ps.size != 0:
			rand_P = np.random.choice(good_Ps)
		else:
			rand_P = np.random.choice(P[:10])
			P_counter += 1

        # Assign period and semi-major axis
		binaries[i,3] = rand_P
		a_appo[i] = pow(G*(m1+m2)*(rand_P*24.*60.*60.)**2. / (2.*np.pi*pcinkm)**2., 1./3.)
		
        # Remove used period
		P = np.delete(P, np.argwhere(P == rand_P))
	
	e = np.zeros(len(ind_stars)) # Array of eccentricities
	P = np.zeros(len(ind_stars)) # Array of eccentricities
	a = np.zeros(len(ind_stars)) # Array of eccentricities

	print("Number of periods below P_min:", P_counter)
	for i in range(len(ind_stars)-1):
		if(ind_stars[i]== ind_stars[i+1]):
			index = np.where(binaries[:,0]==m_stars[i])[0][0]
			e[i]=binaries[index,2]
			e[i+1]=binaries[index,2]
			P[i]=binaries[index,3]
			P[i+1]=binaries[index,3]
			a[i]=a_appo[index]
			a[i+1]=a_appo[index]
	
	psimin=0.0
	psimax=2.*np.pi

	psi=[] #angle psi (phase)
	tol=1e-7
	Efinale=np.zeros(len(ind_stars)) #eccentric anomaly of binary
	fase=np.zeros(len(ind_stars)) #phase of binary
	s=0
	j=0
	while(j<len(ind_stars)-1):
		if(ind_stars[j]== ind_stars[j+1]):
			p = float(np.random.uniform(psimin,psimax))
			psi.append(p)
			Efinale[j] = bisec(p, e[j],tol) #call bisection and solves eccentric anomaly
			Efinale[j+1] = bisec(p, e[j],tol)
			f=2.0 * np.arctan((((1.+e[j])/(1.-e[j])))**0.5 * np.tan(Efinale[j]/2.))
			fase[j] = f
			fase[j+1] = f
		j+=1  
	
	xb=np.zeros(len(ind_stars),float)
	yb=np.zeros(len(ind_stars),float)
	zb=np.zeros(len(ind_stars),float)
	xbappo1 = []
	xbappo2 = []
	ybappo1 = []
	ybappo2 = []
	zbappo1 = []
	zbappo2 = []
	vxb=np.zeros(len(ind_stars),float)
	vyb=np.zeros(len(ind_stars),float)
	vzb=np.zeros(len(ind_stars),float)

	i=0
	while(i<len(ind_stars)-1):
		if(ind_stars[i]== ind_stars[i+1]):
			xb[i] = -m_stars[i+1]/(m_stars[i]+m_stars[i+1]) * a[i] * (1.-e[i]**2) * np.cos(fase[i])/(1.+(e[i] * np.cos(fase[i])))
			yb[i] = -m_stars[i+1]/(m_stars[i]+m_stars[i+1]) * a[i] * (1.-e[i]**2) * np.sin(fase[i])/(1.+(e[i] * np.cos(fase[i])))
			zb[i] = 0.0   
			vxb[i] = -m_stars[i+1]/(m_stars[i]+m_stars[i+1]) * (e[i] * np.cos(fase[i]) /(1+ e[i] * np.cos(fase[i])) - 1.) * np.sin(fase[i]) * (1+ e[i] * np.cos(fase[i])) * (G * (m_stars[i]+m_stars[i+1]) / (a[i]*(1- e[i]*e[i])))**0.5 
			vyb[i] = -m_stars[i+1]/(m_stars[i]+m_stars[i+1]) * (e[i] * (np.sin(fase[i]))**2 /(1+ e[i] * np.cos(fase[i])) + np.cos(fase[i])) * (1+ e[i] * np.cos(fase[i])) * (G * (m_stars[i]+m_stars[i+1]) / (a[i]*(1- e[i]*e[i])))**0.5                                      
			vzb[i] = 0.0
			
			xb[i+1] = m_stars[i]/(m_stars[i]+m_stars[i+1]) * a[i] * (1.-e[i]**2) * np.cos(fase[i])/(1.+(e[i] * np.cos(fase[i])))
			yb[i+1] = m_stars[i]/(m_stars[i]+m_stars[i+1]) * a[i] * (1.-e[i]**2) * np.sin(fase[i])/(1.+(e[i] * np.cos(fase[i])))
			zb[i+1] = 0.0    
			vxb[i+1] = m_stars[i]/(m_stars[i]+m_stars[i+1]) * (e[i] * np.cos(fase[i]) /(1+ e[i] * np.cos(fase[i])) - 1.) * np.sin(fase[i]) * (1+ e[i] * np.cos(fase[i])) * (G * (m_stars[i]+m_stars[i+1]) / (a[i]*(1- e[i]*e[i])))**0.5   
			vyb[i+1] = m_stars[i]/(m_stars[i]+m_stars[i+1]) * (e[i] * (np.sin(fase[i]))**2 /(1+ e[i] * np.cos(fase[i])) + np.cos(fase[i])) * (1+ e[i] * np.cos(fase[i])) * (G * (m_stars[i]+m_stars[i+1]) / (a[i]*(1- e[i]*e[i])))**0.5                                      
			vzb[i+1] = 0.0

			xbappo1.append(xb[i])
			xbappo2.append(xb[i+1])
			ybappo1.append(yb[i])
			ybappo2.append(yb[i+1])
			zbappo1.append(zb[i])
			zbappo2.append(zb[i+1])

		i+=1
    
	
	xp = np.zeros(len(ind_stars))
	yp = np.zeros(len(ind_stars))
	zp = np.zeros(len(ind_stars))
	vxp = np.zeros(len(ind_stars))
	vyp = np.zeros(len(ind_stars))
	vzp = np.zeros(len(ind_stars))

	xpappo1 = []
	xpappo2 = []
	ypappo1 = []
	ypappo2 = []
	zpappo1 = []
	zpappo2 = []

	inclvect = np.zeros((len(ind_stars),3))

	i=0 
	while(i<len(ind_stars)-1):
		if(ind_stars[i]== ind_stars[i+1]):
			point1 = np.array([xb[i],yb[i],zb[i]])
			point2 = np.array([xb[i+1],yb[i+1],zb[i+1]])
			normv = (0.,0.,1.)
			xaxr = np.random.uniform(-1.,1.) 
			yaxr = np.random.uniform(-1.,1.)
			zaxr = np.random.uniform(-1.,1.) 
			normvr=np.array([xaxr,yaxr,zaxr])
			normvr= normvr/ np.linalg.norm(normvr)
			inclvect[i] = normvr
			inclvect[i+1] = normvr  
		
			costheta = np.dot(normv,normvr)/(np.linalg.norm(normv)*np.linalg.norm(normvr))
			rotaxis = np.cross(normv,normvr)/np.linalg.norm(np.cross(normv,normvr))
	
			c = costheta
			s = np.sqrt(1-c*c)
			C = 1-c
			rmat = np.array([[ rotaxis[0]*rotaxis[0]*C+c, rotaxis[0]*rotaxis[1]*C-rotaxis[2]*s,rotaxis[0]*rotaxis[2]*C+rotaxis[1]*s ],
			[ rotaxis[1]*rotaxis[0]*C+rotaxis[2]*s, rotaxis[1]*rotaxis[1]*C+c, rotaxis[1]*rotaxis[2]*C-rotaxis[0]*s ],
			[ rotaxis[2]*rotaxis[0]*C-rotaxis[1]*s, rotaxis[2]*rotaxis[1]*C+rotaxis[0]*s,rotaxis[2]*rotaxis[2]*C+c   ]])

			xp[i]=np.dot(rmat, point1)[0]
			yp[i]=np.dot(rmat, point1)[1]
			zp[i]=np.dot(rmat, point1)[2]
			xp[i+1]=np.dot(rmat, point2)[0]
			yp[i+1]=np.dot(rmat, point2)[1]
			zp[i+1]=np.dot(rmat, point2)[2]

			vpoint1 = np.array([vxb[i],vyb[i],vzb[i]])
			vpoint2 = np.array([vxb[i+1],vyb[i+1],vzb[i+1]])

			vxp[i]=np.dot(rmat, vpoint1)[0]
			vyp[i]=np.dot(rmat, vpoint1)[1]
			vzp[i]=np.dot(rmat, vpoint1)[2]

			vxp[i+1]=np.dot(rmat, vpoint2)[0]
			vyp[i+1]=np.dot(rmat, vpoint2)[1]
			vzp[i+1]=np.dot(rmat, vpoint2)[2]

			xpappo1.append(xp[i])
			xpappo2.append(xp[i+1])
			ypappo1.append(yp[i])
			ypappo2.append(yp[i+1])
			zpappo1.append(zp[i])
			zpappo2.append(zp[i+1])
		i+=1
	"""
	axs[0][1].scatter(xpappo1,ypappo1, c='r',  marker="*")
	axs[0][1].scatter(xpappo2,ypappo2, c='b',  marker=".")
	axs[0][1].set_xlim(-1.6*max(max(xp),max(xp)),1.6*max(max(xp),max(xp)))
	axs[0][1].set_ylim(-1.6*max(max(yp),max(yp)),1.6*max(max(yp),max(yp)))
	axs[0][1].set_xlabel('$x$ [pc]')
	axs[0][1].set_ylabel('$y$ [pc]')
	axs[0][1].ticklabel_format(style = 'sci', axis='x', scilimits=(0,0))
	axs[0][1].ticklabel_format(style = 'sci', axis='y', scilimits=(0,0))
	
			
	axs[1][1].scatter(ypappo1,zpappo1, c='r', marker="*")
	axs[1][1].scatter(ypappo2,zpappo2, c='b', marker=".")
	axs[1][1].set_xlim(-1.6*max(max(yp),max(yp)),1.6*max(max(yp),max(yp)))
	axs[1][1].set_ylim(-1.6*max(max(zp),max(zp)),1.6*max(max(zp),max(zp)))
	axs[1][1].set_xlabel('$y$ [pc]')
	axs[1][1].set_ylabel('$z$ [pc]')
	axs[1][1].ticklabel_format(style = 'sci', axis='x', scilimits=(0,0))
	axs[1][1].ticklabel_format(style = 'sci', axis='y', scilimits=(0,0))
	

	axs[2][1].scatter(zpappo1,xpappo1, c='r',  marker="*")
	axs[2][1].scatter(zpappo2,xpappo2, c='b', marker=".")
	axs[2][1].set_xlim(-1.6*max(max(zp),max(zp)),1.6*max(max(zp),max(zp)))
	axs[2][1].set_ylim(-1.6*max(max(xp),max(xp)),1.6*max(max(xp),max(xp)))
	axs[2][1].set_xlabel('$z$ (AU)')
	axs[2][1].set_ylabel('$x$ (AU)')
	axs[2][1].ticklabel_format(style = 'sci', axis='x', scilimits=(0,0))
	axs[2][1].ticklabel_format(style = 'sci', axis='y', scilimits=(0,0))
    
	
	pl.tight_layout()
	#pl.savefig("binary_spatial_distribution"+simnamered+"_bin.png")
	#pl.show()
	pl.close()
    """

	i=0
	while(i<len(ind_stars)-1):
		if(ind_stars[i]== ind_stars[i+1]):
			x_stars[i]= x_stars[i]+xp[i]
			x_stars[i+1]= x_stars[i+1]+xp[i+1]
			y_stars[i]= y_stars[i]+yp[i]
			y_stars[i+1]= y_stars[i+1]+yp[i+1]
			z_stars[i]= z_stars[i]+zp[i]
			z_stars[i+1]= z_stars[i+1]+zp[i+1] 

			vx_stars[i]= vx_stars[i]+vxp[i]
			vx_stars[i+1]= vx_stars[i+1]+vxp[i+1]
			vy_stars[i]= vy_stars[i]+vyp[i]
			vy_stars[i+1]= vy_stars[i+1]+vyp[i+1]
			vz_stars[i]= vz_stars[i]+vzp[i]
			vz_stars[i+1]= vz_stars[i+1]+vzp[i+1]
		i+=1	

	return e,P,a, x_stars, y_stars, z_stars, vx_stars, vy_stars, vz_stars, inclvect
