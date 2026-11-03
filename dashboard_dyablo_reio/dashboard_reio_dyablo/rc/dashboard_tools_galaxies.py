import numpy as np
import pandas as pa
import matplotlib as mpl
mpl.use('agg')
# mpl.use('agg') is to suppress frontend displays => good for JZ
#mpl.use('nbagg')
# this is to have interactive plots
import matplotlib.pyplot as plt
from astropy.io import ascii 
# astropy is to load ascii files comfortably
# also cosmo
from astropy.cosmology import FlatLambdaCDM
import astropy.units as u # needed to compute tau_electron_scattering
from astropy.cosmology import z_at_value
import os
import glob
from scipy import spatial
from scipy import interpolate

# --- dashboard_reio_dyablo patch: data paths and cosmology are settable (were hard-coded "aux/" and CoDa II) ---
AUX = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "aux")
COSMO = FlatLambdaCDM(name="CoDaII", H0=67.77 * u.km / u.s / u.Mpc, Tcmb0=2.725 * u.K, Om0=0.307, Ob0=0.0482)

class _Interp2D:
    """Drop-in for scipy.interpolate.interp2d (removed in scipy 1.14): bilinear in (age, Z), clipped to the table."""
    def __init__(self, x, y, z):
        x, y, z = np.asarray(x, float), np.asarray(y, float), np.asarray(z, float)
        ox, oy = np.argsort(x), np.argsort(y)
        self.x, self.y = x[ox], y[oy]
        self.f = interpolate.RegularGridInterpolator((self.y, self.x), z[np.ix_(oy, ox)])

    def __call__(self, x, y):
        x = np.clip(np.asarray(x, float), self.x[0], self.x[-1])
        y = np.clip(np.asarray(y, float), self.y[0], self.y[-1])
        shape = np.broadcast(x, y).shape
        return self.f(np.stack(np.broadcast_arrays(y, x), axis=-1).reshape(-1, 2)).reshape(shape)


def set_cosmology(cosmo):
    """Use the cosmology of the run being analysed (astropy FlatLambdaCDM with Ob0 set)."""
    global COSMO
    COSMO = cosmo



def readstars(filename,boxsizehm1cMpc):
    """ read star files ramses-cudaton
    boxsize in h-1 cMpc
    use as 
    [cuage,cuz,starid,starmass,xstar,ystar,zstar,starage,starZ]=readstars(filename)
    cuage: age of the snap
    cuz: 
    """
    header=ascii.read(filename,data_end=1)
    cuage=float(header[0][-1])
    cuage=cuage*u.Myr
    cosmo = COSMO
    cuz=z_at_value(cosmo.age,cuage)
    print(header)
    print("Redshift of starfile z=",cuz)
    boxsize=boxsizehm1cMpc*u.Mpc/(cosmo.H0/(100.*u.km/u.s/u.Mpc))
    
    data = ascii.read(filename,data_start=2)
    d=data.to_pandas()
    d=d.values
    [starid,starmass,xstar,ystar,zstar,starage,starZ,dummy1,dummy2]=d.T
    return [cuage,cuz,starid,starmass,xstar*boxsize,ystar*boxsize,zstar*boxsize,starage,starZ]
    
def plotCSFRD(starage,starmass,cuage,boxsize,H0=67.77,dir='./PDFs/',simname='CoDa III',nodisp=1):
    """ Plots cosmic star formation rate density
    all vars as returned by readstars
        boxsize assumed in h^-1 cMpc  so no units needed
    """
    dtbins=10*u.Myr
    nbins=int(cuage/dtbins)
    agebins=np.linspace(0.,cuage,nbins)
    boxsizewithunits=boxsize*u.Mpc
    cosmo = COSMO
    SFH,bins=np.histogram(starage,bins=agebins,weights=starmass)
    SFH=SFH/dtbins.to(u.yr)/(boxsizewithunits/(cosmo.H0/(100.*u.km/u.s/u.Mpc)))**3
    zbins=[z_at_value(cosmo.age,cuage-age) for age in agebins[:-1]+dtbins/2.]

    # Points from Bouwens 2014
    zbou14=np.array([3.8,5.,5.9,6.8,8.0,9.2,10.])
    sfrdbou14=np.array([-1.1,-1.36,-1.67,-1.83,-2.17,-2.87,-3.45])
    sfrduncbou14=np.array([-1.48,-1.7,-1.92,-2.06,-2.32,-2.87,-3.45])
    sfrdmaxbou14=np.array([0.05,0.06,0.08,0.1,0.11,0.37,0.36])
    sfrdminbou14=np.array([0.05,0.06,0.08,0.1,0.11,0.49,0.36])
    sfrmin=sfrduncbou14-sfrdminbou14
    sfrmax=sfrdbou14+sfrdmaxbou14

    plt.fill_between(zbou14,sfrmin,sfrmax,facecolor='blue', edgecolor='none',alpha=0.25,label=r"Bouwens+14")
    #plt.errorbar(zbou14,sfrduncbou14,yerr=[sfrdmaxbou14,sfrdminbou14])

    plt.plot(zbins,np.log10((SFH*1*u.yr).to_value(u.Mpc**-3)),label=simname)
    plt.yscale('linear')
    plt.xscale('linear')
    plt.xlim((5.,15.))
    plt.ylim(-5.,-1.)

    plt.xlabel(r"$\mathrm{redshift} \, z$",fontsize=15)
    plt.ylabel(r"$\mathrm{SFR(M_{\odot}/yr/cMpc^3)}$ ",fontsize=15)
    plt.title('Cosmic SFR density')
    plt.legend()

    plt.savefig(dir+'CSFRD.pdf')
    if(nodisp==0): 
        plt.show()
#        return
    plt.close()
    
def readphew(namecore,boxsizehm1cMpc):
    """ boxsize in h-1 cMpc
        Example
        [haloid,ncell,hx,hy,hz,hpeak,hmass] = readphew("./output_00011/halo_00011.txt",8)
    
    """
    filelist=glob.glob(namecore+"*")
    nll=len(filelist)
    print(nll)
    alld=[]
    i=0

    for file in filelist :
        if(np.mod(i,1000)==0) :
            print(i)
        data = ascii.read(file,data_start=1)
        d=data.to_pandas()
        d=d.values
        alld=np.append(alld,d)
        i=i+1

    [haloid,ncell,hx,hy,hz,hpeak,hmass]=np.reshape(alld,(int(len(alld)/7),7)).T
    
    cosmo = COSMO
    boxsize=boxsizehm1cMpc*u.Mpc/(cosmo.H0/(100.*u.km/u.s/u.Mpc))
    Mbox=(cosmo.critical_density(0)*(boxsize.to(u.cm))**3).to(u.Msun)*(cosmo.Om0)
    
    hmass=Mbox*hmass
    
    return [haloid,ncell,hx*boxsize,hy*boxsize,hz*boxsize,hpeak,hmass]

def readfof(data_pth,boxsizehm1cMpc,ngrid):

       """
       First run add total halo number from each masst file
       Then build data array
       Then perform another run to get data
       """


       #Get dir contents
       files =  os.listdir(data_pth)
       print(files)

       masst = list(filter(lambda name : 'masst' in name,files))
    
       nhalos_tot = 0


       #Get halo number
       for target in masst[:]:
           print('Opening %s ... ' %target)
           with open(os.path.join(data_pth,target), 'rb' ) as ff:
                  
               nhalos_tot+=np.fromfile(ff,np.int32,count=3)[1] #Number between 2 buoys
               print('%s halos ... \n' %nhalos_tot)
           ff.close()


       #idx mass x y z file_nb
       halo_data = np.zeros((nhalos_tot,6))


       #Get data
       line = 0
       loc_line = 0
       for target in masst[:]:
           print('Opening %s ... ' %target)
           with open(os.path.join(data_pth,target), 'rb' ) as ff:
               nhalos=np.fromfile(ff,np.int32,count=3)[1] #Number between 2 buoys
               halo_data[line:line+nhalos,-1]=(float(target[-4:])*np.ones((1,nhalos)))
               for loc_line in range(nhalos):

                     buoy = np.fromfile(ff,np.int32,count=1)[0]
                     idx = np.fromfile(ff,np.int64,count=1)[0]
                     mass=np.fromfile(ff,np.int32,count=1)[0]
                     x,y,z = np.fromfile(ff,np.float32,count=3)
                     buoy = np.fromfile(ff,np.int32,count=1)[0]

                     halo_data[line,:-1]= np.asarray([idx,mass,x,y,z])
                     #print(halo_data[line,:])
                     line+=1
           ff.close()
           
       return(halo_data)



def R200(M,cosmo):
    """ returns the R200 radius in the given cosmology, in cMpc, with M in Msun"""
    rhocritdm=cosmo.critical_density0.to(u.Msun/u.Mpc**3)*cosmo.Odm0
    return pow((3.*M)/(4.*np.pi*200.*rhocritdm),1./3.)
    
def get_mag_tab_BPASSV221_betas(filename):                                                                       
                                                                                                         
    reading=[]                                                                                           
    with open(filename,'r') as src:   
                                                                                                         
        for line in src:                                                                                 
            if '#' not in str(line):                                                                     
                tmp=line.split(',')                                                                      
                tmp[-1]=tmp[-1].strip('\n')                                                              
                #print(tmp)                                                                              
                reading.append(np.float64(tmp))                                                          
                                                                                                                                                                                         
    age_bins=reading[2]                                                                                  
    metal_bins=reading[3]                                                                                
                                                                                                         
    xis=np.vstack(reading[4:17])                                                                         
    mags=np.vstack(reading[17:30])                                                                       
    contbetalow=np.vstack(reading[30:43])                                                                
    contbetahigh=np.vstack(reading[43:56])                                                               
    beta=np.vstack(reading[56:69])                                                                       
                                                                                                         
    return(mags,xis,contbetalow,contbetahigh,beta,metal_bins,age_bins) 

def gen_age_Z_interpolator(file=AUX+"/src_files/Emissivity_MAB1600_BPASSv2.2.1_kroupa_binary_MMax=100.txt"):
    """returns interpolators for mag,betahigh,betalow,beta from source file file
    use as [mag_age_Z,contbetahigh_age_Z,contbetalow_age_Z,beta_age_Z]=gen_age_Z_interpolator(file)
    age in Myr, Z in absolute scale (i.e. not Zsun=0.02)
    
    """
    #file=AUX+"/src_files/short_STEP_10Myr_BPASSv2.2.1_kroupa_binary_MMax=100.txt"
    print('Using Source file ',file)
    mags,xis,contbetalow,contbetahigh,beta,metal_bins,age_bins=get_mag_tab_BPASSV221_betas(file)
    mag_age_Z=_Interp2D(age_bins,metal_bins,mags)
    contbetahigh_age_Z=_Interp2D(age_bins,metal_bins,contbetahigh)
    contbetalow_age_Z=_Interp2D(age_bins,metal_bins,contbetalow)
    beta_age_Z=_Interp2D(age_bins,metal_bins,beta)
    
    return [mag_age_Z,contbetahigh_age_Z,contbetalow_age_Z,beta_age_Z]

def magpop(masses,ages,Zs,mag_age_Z):
    """ returns the total mag of the population made of stellar particles
    specifed by masses, ages, Zs, using the magnitude interpolator mag_age_Z
    """
    nstars=len(masses)
    if(nstars==0) :
        return -100.
    res=0.
    for istar in range(nstars):
        mag=mag_age_Z(ages[istar],Zs[istar])
        res=res+masses[istar]*np.power(10.,-mag/2.5)
    res=-2.5*np.log10(res)
    return res

def computeallmags(starmass,starage,starZ,allstarinds,initial_mass=None) :
    """returns for mags for all haloes
    allstarinds is a list of lists.
    Each element is the list of stars within R200
    initial_mass: per-star initial masses; default starmass/0.8 (eta_sn=0.2 already returned, as in ramses-cudaton)
    """
    if initial_mass is None:
        initial_mass = starmass/0.8

    Nhalo=len(allstarinds)
    mags=np.full(Nhalo,-100.)
    Zaverage=np.full(Nhalo,-100.)
    halo_stellar_mass=np.full(Nhalo,-100.)


    [mag_age_Z,contbetahigh_age_Z,contbetalow_age_Z,beta_age_Z]=gen_age_Z_interpolator()
    print('done recovering age-z-mags table')

    for i in range(Nhalo):
        if(np.mod(i,10)==0) :
            print(i)
        mags[i]=magpop(initial_mass[(allstarinds[i])],starage[(allstarinds[i])],starZ[(allstarinds[i])]*0.02,mag_age_Z)
        Zaverage[i]=np.average(starZ[(allstarinds[i])]*0.02)
        halo_stellar_mass[i]=np.sum(starmass[allstarinds[i]])
        # the /0.8 is because eta_sn=0.2
        # *0.02 is because getstarlist to generate the starfile outputs Z in Zsun units

    return [mags,Zaverage,halo_stellar_mass]
    
    
    
def associate_star_halo2(xstar,ystar,zstar,hx,hy,hz,hmass,boxsizehm1cMpc,Minf=1e8*u.solMass,verb=50000,workers=1) :
    """ from star positions
        and halo positions and masses
        returns the indices of stars within R200 of the halo centers
        assumes CoDa II cosmo parameters
        tries to use parallelism of querby_ball_point
        it works great
        downside is progress not shown
        """
    print("associate stars to haloes")
    print("version with Minf and verb")
    #compute tree
    print("Computing tree")
    starsxyz=np.array([xstar.value,ystar.value,zstar.value]).T
    #print(np.shape(starsxyz))
    cosmo = COSMO
    #print(type(np.array([xstar.value,ystar.value])))
    boxsizehm1cMpc=boxsizehm1cMpc+0.0000000001 # otherwise cKDTree complains that coords are larger than box size, which is wrong
    # I checked and its equal at worst
    boxsize=boxsizehm1cMpc*u.Mpc/(cosmo.H0/(100.*u.km/u.s/u.Mpc))
    print('boxsize',boxsize)
    #print(np.max(zstar))
    treeStar = spatial.cKDTree(starsxyz,boxsize=boxsize/u.Mpc)
    print('Done computing tree')
    #print(treeStar.boxsize) # to set up periodicity
    #print("ecriture de l'arbre")
    starhalo=np.full(len(xstar),int(-1))
    starhalomass=np.full(len(xstar),-1.) 
    rhocrit=cosmo.critical_density0.to(u.Msun/u.Mpc**3) # thats total density and I dont even need it here!!
    
    #print(R200(hmass[i],cosmo))
    allstarinds=[]
    allhaloinds=[]
    Nhalo=len(hmass)
    print('Number of haloes to process ',Nhalo)

    if(0): # old serial use of treeStar.query_ball_point

        for i in range(len(hmass)) : 
            if np.mod(i,verb)==0 :
                print('Halo number',i)
            if hmass[i]<Minf:
                continue

            # old call
#            ind=treeStar.query_ball_point([hx[i]/u.Mpc,hy[i]/u.Mpc,hz[i]/u.Mpc], R200(hmass[i],cosmo)/u.Mpc,n_jobs=1)
            # old call here workers=8 does nothing since we're doing len(hmass) queries
            ind=treeStar.query_ball_point([hx[i]/u.Mpc,hy[i]/u.Mpc,hz[i]/u.Mpc], R200(hmass[i],cosmo)/u.Mpc,workers=workers)


            starhalo[ind]=int(i)
            starhalomass[ind]=hmass[i]
            # NOTE: append makes a copy, maybe faster with extend?
            # but at least this works, not having success with extend
            allstarinds.append(ind)
            allhaloinds.append(i)

    ind=np.where(hmass>Minf)[0]
    # try to use the parallelism of query_ball_point
    # seems to work
    print('starting query_ball_point parallel')
    inds=treeStar.query_ball_point(np.transpose([hx[ind]/u.Mpc,hy[ind]/u.Mpc,hz[ind]/u.Mpc]),R200(hmass[ind],cosmo)/u.Mpc,workers=workers)
    print('Association finished')
    if(0):
        starhaloind=range(len(inds))
        for i in range(len(inds)):
            if(len(inds[i])>0):
                starhalo[inds[i]]=ind[i]
                starhalomass[inds[i]]=ind[i]
    
    allhaloinds=ind
    
                
    
# return the list of indexes
# allhaloinds: the index of haloes that were processed
# allstarinds => the list of star index list belonging to each halo
#    return [allstarinds,allhaloinds]
    return[inds,ind]


def associate_star_halo_ori(xstar,ystar,zstar,hx,hy,hz,hmass,boxsizehm1cMpc,Minf=1e8*u.solMass,verb=50000,workers=1) :
    """ from star positions
        and halo positions and masses
        returns the indices of stars within R200 of the halo centers
        assumes CoDa II cosmo parameters
        """
    print("associate stars to haloes")
    print("version with Minf and verb")
    #compute tree
    print("Computing tree")
    starsxyz=np.array([xstar.value,ystar.value,zstar.value]).T
    #print(np.shape(starsxyz))
    cosmo = COSMO
    #print(type(np.array([xstar.value,ystar.value])))
    boxsizehm1cMpc=boxsizehm1cMpc+0.0000000001 # otherwise cKDTree complains that coords are larger than box size, which is wrong
    # I checked and its equal at worst
    boxsize=boxsizehm1cMpc*u.Mpc/(cosmo.H0/(100.*u.km/u.s/u.Mpc))
    print('boxsize',boxsize)
    #print(np.max(zstar))
    treeStar = spatial.cKDTree(starsxyz,boxsize=boxsize/u.Mpc)
    #print(treeStar.boxsize) # to set up periodicity
    #print("ecriture de l'arbre")
    starhalo=np.full(len(xstar),int(-1))
    starhalomass=np.full(len(xstar),-1.)
    rhocrit=cosmo.critical_density0.to(u.Msun/u.Mpc**3)

    #print(R200(hmass[i],cosmo))
    allstarinds=[]
    allhaloinds=[]
    Nhalo=len(hmass)
    print('Number of haloes to process ',Nhalo)

    for i in range(len(hmass)) : 
        if np.mod(i,verb)==0 :
            print('Halo number',i)
        if hmass[i]<Minf:
            continue
            
        # old call
        #ind=treeStar.query_ball_point([hx[i]/u.Mpc,hy[i]/u.Mpc,hz[i]/u.Mpc], R200(hmass[i],cosmo)/u.Mpc,n_jobs=1)
        ind=treeStar.query_ball_point([hx[i]/u.Mpc,hy[i]/u.Mpc,hz[i]/u.Mpc], R200(hmass[i],cosmo)/u.Mpc,workers=8)

        
        starhalo[ind]=int(i)
        starhalomass[ind]=hmass[i]
        # NOTE: append makes a copy, maybe faster with extend?
        # but at least this works, not having success with extend
        allstarinds.append(ind)
        allhaloinds.append(i)

        # cant get it right with extend, so stick with append
        if 0:
            allstarinds.extend(ind)
            if(len(ind)>0):
                allhaloinds.extend([i])

    print('Association finished')
# return the list of indexes
# allhaloinds: the index of haloes that were processed
# allstarinds => the list of star index list belonging to each halo
    return [allstarinds,allhaloinds]



def plot_LF(mags,cuz,boxsizehm1cMpc,H0,redshift,dir='./PDFs/',simname='CoDa III',nodisp=1):
    """Plot LF against Bouwens"""
    magbins=np.linspace(-25.,0.,53)
    [h,bins]=np.histogram(mags,magbins)
    bins=[mag +0.5  for mag in magbins[:-1]]
    print(len(h))
    print(len(bins))
    plt.plot(bins,h/np.diff(magbins)/(boxsizehm1cMpc/(H0/100.))**3,label=r"$\mathrm{calib}$")

    poissonnoise=np.sqrt(h)
    LFmax=(h+poissonnoise)/np.diff(magbins)/(boxsizehm1cMpc/(H0/100.))**3
    LFmin=(h-poissonnoise)/np.diff(magbins)/(boxsizehm1cMpc/(H0/100.))**3
    plt.fill_between(bins,LFmin,LFmax,facecolor='blue', alpha=0.25,edgecolor='none')
    
    zsbou14=[4,5,7,8,10]
    ibou14=np.abs(zsbou14-cuz).argmin()
    #print(ibou14)
    #print(zsbou14[ibou14])
    data = ascii.read(AUX+"/obs/bouwens2014_reformatted/z="+str(zsbou14[ibou14]))
    d=data.to_pandas()
    d=d.values
    [magbou14,Phibou14,deltaPhibou14]=d.T

    data = ascii.read(AUX+"/obs/bouwens2016_reformatted/z=6")
    d=data.to_pandas()
    d=d.values
    [magbou16,Phibou16,erpPhibou16,ermPhibou16]=d.T
    ermPhibou16=-ermPhibou16

    plt.errorbar(magbou14,Phibou14,yerr=deltaPhibou14,label=r"$\mathrm{Bouwens(2014) \, \, z="+str(zsbou14[ibou14])+"}$",c='b')
    plt.errorbar(magbou16,Phibou16,yerr=[ermPhibou16,erpPhibou16],label=r"$\mathrm{Bouwens(2016) \, \, z=6}$",c='r')

    plt.axis([-10.,-25.,1.e-6,1.e1])
    plt.yscale('log')
    plt.xlabel(r"$\mathrm{M_{AB1600}}$",fontsize=15)
    plt.ylabel(r"$\mathrm{\Phi(M_{AB1600}) (mag^{-1}cMpc^{-3})}$",fontsize=15)
    plt.title('UV LF z='+str(np.around(cuz,1)))
    plt.legend()
    
    plt.savefig(dir+'UVLF_z='+str(np.around(cuz,2))+'.pdf')
    if(nodisp==0): plt.show()
    plt.close()
    
    return 0

def plotSFR_vs_z_M(hmass,allstarinds,starage,starmass,cuage,cuz,dir="./PDFs/",simname="CoDa III",nodisp=1) :

    """Make the SFR vs z plot for a range of masses"""
    
    print("Computing Halo SFR vs z vs M")
    print("Assuming CoDa II cosmology")
    dtbins=10*u.Myr
    nbins=int(cuage/dtbins)
    agebins=np.linspace(0.,cuage,nbins)
    #boxsizewithunits=boxsizehm1cMpc*u.Mpc
    cosmo = COSMO
    zbins=[z_at_value(cosmo.age,cuage-age) for age in agebins[:-1]+dtbins/2.]


    lmin=7
    lmax=12
    lmassbins=np.linspace(lmax,lmin,int(lmax-lmin+1))

    for i in range(len(lmassbins)-1) :

        sstarmass=[]
        starinhaloinds=[]

        massmax=10**lmassbins[i]
        massmin=10**lmassbins[i+1]
        slmassmin=str(int(lmassbins[i+1]))
        slmassmax=str(int(lmassbins[i]))

        ind=np.where((hmass > massmin*u.Msun)&(hmass  < massmax*u.Msun))[0]
        nhalo=len(ind)
        print('Number of haloes to process',nhalo)
        # get all ages of stars in these haloes
        c = [ starage[allstarinds[i]] for i in ind]
        # flatten list to get all ages in a list
        sstarage = [item for sublist in c for item in sublist]

        # get all masses of stars in these haloes
        c = [ starmass[allstarinds[i]] for i in ind]
        # flatten list to get all ages in a list
        sstarmass = [item for sublist in c for item in sublist]
        SFH,bins=np.histogram(u.Quantity(sstarage,u.Myr),bins=agebins,weights=np.asarray(sstarmass,dtype=float))
        SFH=SFH/dtbins.to(u.yr)/float(max(nhalo,1))
        #plt.plot(zbins,np.log10(SFH*1*u.yr+1.e-10),label=r"$\mathrm{10^{"+slmassmin+"-"+slmassmax+"}\,M_{\odot}}$")   
        plt.plot(zbins,np.log10(SFH*1*u.yr+1.e-10),label=r"$\mathrm{"+slmassmin+"-"+slmassmax+"}$")   


    plt.yscale('linear')
    plt.xscale('log')
    plt.xlim((2.,20.))
    plt.ylim((-8.,2.))
    iticks=[2,3,4,5,6,7,8,9,10,12,15,20]
    nticks=len(iticks)
    ticks=[str(i) for i in iticks]

    plt.xticks(iticks,ticks)

    plt.xlabel(r"$\mathrm{redshift} \, z$",fontsize=15)
    plt.ylabel(r"$\mathrm{log_{10}(SFR[M_{\odot}/yr])}$ ",fontsize=15)
    plt.title('Halo Star Formation History z='+str(np.around(cuz,1)))
    plt.legend(loc=2,title=r"$\mathrm{log_{10}(M/M_{\odot})}$ ")
    
    plt.savefig(dir+'Halo_SFR_M_z_z='+str(np.around(cuz,2))+'.pdf')
    if(nodisp==0): plt.show()
    plt.close()




def plot_stellarmass_Z(halo_stellarmass,Zaverage,cuz,dir='./PDFs/',simname='CoDa III',nodisp=1):
    """Z vs stellar mass"""
    plt.scatter(np.log10(halo_stellarmass),np.log10(Zaverage/0.02),label=r"$\mathrm{calib}$")
    #plt.axis([-10.,-25.,1.e-6,1.e1])
    #plt.yscale('log')
    plt.xlabel('log10(Halo stellar mass M*)')
    plt.ylabel('log10(<Z*>/Zsun) (absolute)')
    plt.ylim(-4.,0.)
    plt.title('z='+str(np.around(cuz,1)))
    plt.grid()
    plt.legend()
    plt.savefig(dir+'starmass_Z_z='+str(np.around(cuz,2))+'.pdf')
    if(nodisp==0): plt.show()
    plt.close()

    return 0


