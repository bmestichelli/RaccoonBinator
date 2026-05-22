import matplotlib.pyplot as plt
import numpy as np

from parameters import *
#########################################################################################àà
#Input and output functions


def walklevel(some_dir, level=1):
    some_dir = some_dir.rstrip(os.path.sep)
    assert os.path.isdir(some_dir)
    num_sep = some_dir.count(os.path.sep)
    for root, dirs, files in os.walk(some_dir):
        yield root, dirs, files
        num_sep_this = root.count(os.path.sep)
        if num_sep + level <= num_sep_this:
            del dirs[:]


def load_dat(simname, verbose=True, remove_small_masses=True, mass_limit=0.1, type_d="float", skipheader=0, skipfooter=0):
    
    if(verbose==True):
        print("Simulation under consideration:", simname)
    
    m, x, y, z, vx, vy, vz= np.genfromtxt(simname, dtype=type_d, skip_header=skipheader, skip_footer=skipfooter, comments="#", unpack=True)
    
    if(remove_small_masses==True):
        if(verbose==True):
            print("Number of stars with mass below lower limit ("+str(mass_limit)+" MSun):", len(m[m[:]<0.1]))
        #I decided to simply remove the sinks that are less massive than 0.1 solar masses, if there is any (since it is difficult to treat them and they have a ridicolous total mass)
        x=x[m[:]>=0.1]
        y=y[m[:]>=0.1]
        z=z[m[:]>=0.1]
        vx=vx[m[:]>=0.1]
        vy=vy[m[:]>=0.1]
        vz=vz[m[:]>=0.1]
        m=m[m[:]>=0.1]
    if(verbose==True):
        print("Number of stars particles:", len(m))
    
    return m, x, y, z, vx, vy, vz


    
def print_out(file_name, m, x, y, z, vx, vy, vz, out_extension="dat"):
    fname = os.path.join(OUTPUT_PATH, file_name + "." + out_extension)
    f=open(fname,'w') 
    f.write("# m [Msun], x [pc], y [pc], z [pc], vx [km/s], vy [km/s], vz [km/s]  \n")
    for j in range(len(m)):
        f.write("%.10f %.10f %.10f %.10f %.10f %.10f %.10f\n" % (m[j], x[j], y[j], z[j], vx[j], vy[j], vz[j]))
    

######################################################################################
#Basic mathematical and physical functions
######################################################################################

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

    
def ode(N,m,x,y,z):
    ax=np.zeros(N,float)
    ay=np.zeros(N,float)
    az=np.zeros(N,float)
    #print(N,m,x,y,z)
    for i in range(0,N,1):
        for j in range(0,N,1):
            if(i!=j):
                rij=((x[i]-x[j])**2+(y[i]-y[j])**2+(z[i]-z[j])**2)**1.5
                #print(rij)
                ax[i]+=-G*m[j]*(x[i]-x[j])/rij
                ay[i]+=-G*m[j]*(y[i]-y[j])/rij
                az[i]+=-G*m[j]*(z[i]-z[j])/rij
    return ax,ay, az


def leapfrog(N,m,x,y,z,vx,vy,vz,h):
    ax,ay,az=ode(N,m,x,y,z)  #a(t)
    for i in range(N):
        x[i]=x[i]+h*vx[i]+0.5*h*h*ax[i] #x(t+h)
        y[i]=y[i]+h*vy[i]+0.5*h*h*ay[i] #y(t+h)
        z[i]=z[i]+h*vz[i]+0.5*h*h*az[i] #z(t+h)
    axh,ayh, azh=ode(N,m,x,y,z)  #a(t+h) uses x(t+h) v(t)
    for i in range(N):
        vx[i]=vx[i]+0.5*h*(ax[i]+axh[i]) #v(t+h) uses a(t), a(t+h) and v(t)
        vy[i]=vy[i]+0.5*h*(ay[i]+ayh[i])
        vz[i]=vz[i]+0.5*h*(az[i]+azh[i])
    return x,y,z,vx,vy,vz


def odej(N,m,x,y,z,vx,vy,vz):
    #print(N,m,x,y,vx,vy,h)
    ax=np.zeros(N,float)
    ay=np.zeros(N,float)
    az=np.zeros(N,float)
    jx=np.zeros(N,float)
    jy=np.zeros(N,float)
    jz=np.zeros(N,float)
    for i in range(0,N,1):
        for j in range(0,N,1):
            if(i!=j):
                xvx=(x[i]-x[j])*(vx[i]-vx[j])+(y[i]-y[j])*(vy[i]-vy[j])+(z[i]-z[j])*(vz[i]-vz[j])
                rij=((x[i]-x[j])**2+(y[i]-y[j])**2+(z[i]-z[j])**2)**0.5
                ax[i]+=-G*m[j]*(x[i]-x[j])/rij**3.
                ay[i]+=-G*m[j]*(y[i]-y[j])/rij**3.
                az[i]+=-G*m[j]*(z[i]-z[j])/rij**3.
                jx[i]+=-G*m[j]*((vx[i]-vx[j])/rij**3.-3.*xvx*(x[i]-x[j])/rij**5.)
                jy[i]+=-G*m[j]*((vy[i]-vy[j])/rij**3.-3.*xvx*(y[i]-y[j])/rij**5.)
                jz[i]+=-G*m[j]*((vz[i]-vz[j])/rij**3.-3.*xvx*(z[i]-z[j])/rij**5.)
    return ax,ay,az,jx,jy,jz


def predict(N,m,x,y,z,vx,vy,vz,h):
    xp=np.zeros(N,float)
    yp=np.zeros(N,float)
    zp=np.zeros(N,float)
    vxp=np.zeros(N,float)
    vyp=np.zeros(N,float)
    vzp=np.zeros(N,float)
    ax,ay,az,jx,jy,jz=odej(N,m,x,y,z,vx,vy,vz)
    #print(ax,ay,jx,jy)
    
    for i in range(N):
        xp[i]=x[i]+h*vx[i]+0.5*h*h*ax[i]+1./6.*h*h*h*jx[i]
        yp[i]=y[i]+h*vy[i]+0.5*h*h*ay[i]+1./6.*h*h*h*jy[i]
        zp[i]=z[i]+h*vz[i]+0.5*h*h*az[i]+1./6.*h*h*h*jy[i]
        vxp[i]=vx[i]+h*ax[i]+0.5*h*h*jx[i]
        vyp[i]=vy[i]+h*ay[i]+0.5*h*h*jy[i]
        vzp[i]=vz[i]+h*az[i]+0.5*h*h*jz[i]
    return xp,yp,zp,vxp,vyp,vzp


def correct(N,m,x,y,z,vx,vy,vz,h):
    #print(N,m,x,y,vx,vy,h)
    ax,ay,az,jx,jy,jz=odej(N,m,x,y,z,vx,vy,vz)
    #print(ax,ay,jx,jy,N)
    xp,yp,zp,vxp,vyp,vzp=predict(N,m,x,y,z,vx,vy,vz,h)   
    axp,ayp,azp,jxp,jyp,jzp=odej(N,m,xp,yp,zp,vxp,vyp,vzp)

    vxold=np.copy(vx)
    vyold=np.copy(vy)
    vzold=np.copy(vz)
    
    for i in range(N):
        vx[i]=vx[i]+0.5*h*(ax[i]+axp[i])+(1./12.)*h*h*(jx[i]-jxp[i])
        vy[i]=vy[i]+0.5*h*(ay[i]+ayp[i])+(1./12.)*h*h*(jy[i]-jyp[i])
        vz[i]=vz[i]+0.5*h*(az[i]+azp[i])+(1./12.)*h*h*(jz[i]-jzp[i])
        
        x[i]=x[i]+0.5*h*(vxold[i]+vx[i])+(1./12.)*h*h*(ax[i]-axp[i])
        y[i]=y[i]+0.5*h*(vyold[i]+vy[i])+(1./12.)*h*h*(ay[i]-ayp[i])
        z[i]=z[i]+0.5*h*(vzold[i]+vz[i])+(1./12.)*h*h*(az[i]-azp[i])
    return x,y,z,vx,vy,vz


def gauss(N, mean=0, sigma=1):
    v = 0.
    for i in range(3):
        s = np.random.uniform(0.,1.,N)
        t = np.random.uniform(0.,1.,N)
        va = np.sqrt(-2.*sigma**2*np.log(s))*np.cos(2*np.pi*t)
        v+= va**2.
      
    v = np.sqrt(v)
    th = np.arccos( np.random.uniform(-1.,1.,N))
    phi = np.random.uniform(0., 2.*np.pi, N)
    vx = v*np.sin(th)*np.cos(phi)
    vy = v*np.sin(th)*np.sin(phi)
    vz = v*np.cos(th)
    
    return vx, vy, vz

###################################################################################################
#Random generators
###################################################################################################

def plummer_sigma(x, a, M):
	return G*M/(6.*np.sqrt(x**2.+a**2.))


def mass_cutoff(radius_cutoff=22.8042468):
    scale_factor = 16.0 / (3.0 * np.pi)
    rfrac = radius_cutoff * scale_factor
    denominator = (1.0 + rfrac ** 2)**(1.5)
    numerator = rfrac ** 3
    return numerator/denominator


def plummer(N, rscale, M):
    #r = rscale/ np.sqrt(np.random.uniform(0.,min(mass_cutoff(),0.999), N)**(-(2.0/3.0)) - 1.0)
    r = rscale/ np.sqrt(np.random.uniform(0.,1, N)**(-(2.0/3.0)) - 1.0)
    th = np.arccos( np.random.uniform(-1.,1.,N))
    phi = np.random.uniform(0., 2.*np.pi,N)
    
    x = r*np.sin(th)*np.cos(phi)
    y = r*np.sin(th)*np.sin(phi)
    z = r*np.cos(th)
    
    radi =np.sqrt(x**2.+y**2.+z**2.)
    #radi=np.sort(radi)
     
    v = 0.
    for i in range(3):
        #s = np.random.uniform(0.,1.,N)
        #t = np.random.uniform(0.,1.,N)
        va = np.random.normal(0, np.sqrt(plummer_sigma(radi, rscale, M)),N) #gauss(np.sqrt(plummer_sigma(radi, rscale, M)),N)  #np.sqrt(-2.*np.sqrt(plummer_sigma(radi))**2*np.log(s))*np.cos(2*np.pi*t)
        v+= va**2.
      
    v = np.sqrt(v)
    th = np.arccos( np.random.uniform(-1.,1.,N))
    phi = np.random.uniform(0., 2.*np.pi, N)
    vx = v*np.sin(th)*np.cos(phi)
    vy = v*np.sin(th)*np.sin(phi)
    vz = v*np.cos(th)
    
    return x,y,z,vx,vy,vz  
"""

def plummer(N, rscale, sigma=1):
    r = rscale/ np.sqrt(np.random.uniform(0.,1., N)**(-(2.0/3.0)) - 1.0)
    th = np.arccos( np.random.uniform(-1.,1.,N))
    phi = np.random.uniform(0., 2.*np.pi,N)
    
    x = r*np.sin(th)*np.cos(phi)
    y = r*np.sin(th)*np.sin(phi)
    z = r*np.cos(th)
    
    v = np.random.normal(0, sigma,N) #gauss(np.sqrt(plummer_sigma(radi)),N)  #np.sqrt(-2.*np.sqrt(plummer_sigma(radi))**2*np.log(s))*np.cos(2*np.pi*t)
    v = abs(v)
    th = np.arccos( np.random.uniform(-1.,1.,N))
    phi = np.random.uniform(0., 2.*np.pi, N)
    vx = v*np.sin(th)*np.cos(phi)
    vy = v*np.sin(th)*np.sin(phi)
    vz = v*np.cos(th)
    
    return x,y,z,vx,vy,vz  
"""