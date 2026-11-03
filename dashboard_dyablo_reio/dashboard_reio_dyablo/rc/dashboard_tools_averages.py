import numpy as np
import pandas as pa
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
from astropy.io import ascii 
# astropy is to load ascii files comfortably
from astropy.cosmology import FlatLambdaCDM
import astropy.units as u # needed to compute tau_electron_scattering
import os

# --- dashboard_reio_dyablo patch: data paths and cosmology are settable (were hard-coded "aux/" and CoDa II) ---
AUX = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "aux")
COSMO = FlatLambdaCDM(name="CoDaII", H0=67.77 * u.km / u.s / u.Mpc, Tcmb0=2.725 * u.K, Om0=0.307, Ob0=0.0482)

def set_cosmology(cosmo):
    """Use the cosmology of the run being analysed (astropy FlatLambdaCDM with Ob0 set)."""
    global COSMO
    COSMO = cosmo


# READ AVERAGES FUNCTIONS

def read_averages(filename) :
    """read averages from ramses-cudaton"""
    tab=pa.read_csv(filename,header=0)
    
    nhash=0
    i=0
    indcomment=[]
    indgood=[]
    for line in tab.values :
        if(line[0].find("#")>-1) :
            #print(line[0])
            nhash=nhash+1
            indcomment=np.append(indcomment,int(i))
        else : indgood=np.append(indgood,int(i))
        i=int(i+1)
    
    if(nhash>0) : print('Skipping '+str(nhash)+' interloping comments lines')
    
    b=tab.values[indgood.astype(int)]

    nlines=int(len(b)/4)
    a=b.reshape((len(b)))
    stab=np.zeros([nlines,11])

    for i in range(len(a)) :
        li=int(i/4)
        if (np.mod(i,4)==0) :
            stab[li][0:3]=np.fromstring(a[i], dtype=float, sep=' ',count=3)
        if (np.mod(i,4)==1) :
            stab[li][3:6]=np.fromstring(a[i], dtype=float, sep=' ',count=3)
        if (np.mod(i,4)==2) :
            stab[li][6:9]=np.fromstring(a[i], dtype=float, sep=' ',count=3)
        if (np.mod(i,4)==3) :
            stab[li][9:11]=np.fromstring(a[i], dtype=float, sep=' ',count=2)

    #re-read to get header
    tab2=pa.read_csv(filename,header=0)
    h=tab2.head(0)

    
    str.split(h.to_string())
    colnames=str.split(h.to_string())[5:16]
    print(colnames)
    return stab


def read_averages_old(filename) :
    """read averages from ramses-cudaton"""
    tab=pa.read_csv(filename,header=0,)

    b=tab.values

    nlines=int(len(b)/4)
    a=b.reshape((len(b)))
    stab=np.zeros([nlines,11])

    for i in range(len(a)) :
        li=int(i/4)
        if (np.mod(i,4)==0) :
            stab[li][0:3]=np.fromstring(a[i], dtype=float, sep=' ',count=3)
        if (np.mod(i,4)==1) :
            stab[li][3:6]=np.fromstring(a[i], dtype=float, sep=' ',count=3)
        if (np.mod(i,4)==2) :
            stab[li][6:9]=np.fromstring(a[i], dtype=float, sep=' ',count=3)
        if (np.mod(i,4)==3) :
            stab[li][9:11]=np.fromstring(a[i], dtype=float, sep=' ',count=2)

    h=tab.head(0)
    str.split(h.to_string())
    colnames=str.split(h.to_string())[5:16]
    print(colnames)
    return stab

def read_avgs(path):
    """read averages by Joe"""
    try:
        cols = ['aexp','nH','x','x*nH','T','J','S','nb_stars']
        datas = np.genfromtxt(os.path.join(path,'averages.txt'),skip_header=1)
        format_datas=np.reshape(np.ravel(datas),(int(len(datas)/3),9))
        return({key:col for col,key in zip(np.transpose(format_datas[:,[0,2,3,4,5,6,7,8]]),cols)})
    
    except:
        cols = ['aexp','nH','x','x*nH','T','J','Z','d','S','nb_stars']

        with open(os.path.join(path,'averages.txt'), 'r') as src:
            data_lines=[]
            tmp_line=[]
            for line in src:
                if '#' in line:continue
                tmp=line.strip('\n').split(' ')
                tmp=[entry for entry in tmp if entry!='']
                if len(tmp_line)<11:
                    tmp_line.extend(tmp)
                else:
                    data_lines.append(np.float64(tmp_line))
                tmp_line=tmp

        format_datas=np.asarray(data_lines)
        return({key:col for col,key in zip(np.transpose(format_datas[:,[0,2,3,4,5,6,7,8,9,10]]),cols)})

#PLOT AVERAGES FUNCTIONS

def plotxhi(z,xHI,dir='./PS/',simname='CoDa III',nodisp=1):
    plt.plot(z,xHI,label=r"$\mathrm{"+simname+"}$",color='black')
    data = ascii.read(AUX+"/fan2006.csv")
    d=data.to_pandas()
    d=d.values
    [zfan,xHIfan,xHIfanm,xHIfanp]=d.T
    xHIfan=xHIfan*1.e-4
    xHIfanm=xHIfan-xHIfanm*1.e-4
    xHIfanp=xHIfanp*1.e-4-xHIfan
    plt.errorbar(zfan,xHIfan,yerr=[xHIfanm,xHIfanp],label=r"$\mathrm{Fan+2006}$",c='b')
    plt.errorbar(7.,10**(-0.7),yerr=abs(0.1-10**-0.7),label=r"$\mathrm{Ouchi 2010}$",c='r')
    
    plt.yscale('log')
    plt.xscale('linear')
    plt.axis([2.,10.,1.e-6,1.e0])
    plt.xlabel(r"$\mathrm{redshift} \, z$",fontsize=15)
    plt.ylabel(r"$\mathrm{x_{HI}}$",fontsize=15)
    plt.legend()
    plt.savefig(dir+'xHI.pdf')
    if(nodisp==0): plt.show()
    plt.close()

def plotxhii(z,xHII,dir='./PS/',simname="CoDa III",nodisp=1):
    plt.plot(z,xHII,label=r"$\mathrm{"+simname+"}$",color='black')

    zdavies18=[7.09,7.54]  
    xhiidavies18=[0.52,0.4]   
    ermxhiidavies18=[0.26,0.2]
    erpxhiidavies18=[0.26,0.23]  
    
    zmason18=7.                                                                                           
    xhiimason18=1.-0.59                                                                                   
    erpxhiimason18=[0.15]                                                                                 
    ermxhiimason18=[0.11]
    
    zgreig17=7.1                                                                                         
    xhiigreig17=1.-0.4                                                                                   
    erpxhiigreig17=[0.21]                                                                                 
    ermxhiigreig17=[0.19]

    zhoag19=7.6
    zhoag19m=[0.6]
    zhoag19p=[0.6]
    xhiihoag19=0.12
    ermxhiihoag19=[0.05]
    erpxhiihoag19=[0.1]

    plt.errorbar(zdavies18,xhiidavies18,yerr=[erpxhiidavies18,ermxhiidavies18],fmt='o',color='r',mec='r',label=r"Davies+18")
    plt.errorbar(zmason18,xhiimason18,yerr=[erpxhiimason18,ermxhiimason18],fmt='p',color='g',mec='g',label=r"Mason+18")
    plt.errorbar(zgreig17,xhiigreig17,yerr=[ermxhiigreig17,erpxhiigreig17],fmt='v',color='b',mec='b',label=r"Greig+17")
    plt.errorbar(zhoag19,xhiihoag19,xerr=[zhoag19m,zhoag19p],yerr=[ermxhiihoag19,erpxhiihoag19],color='cyan',label=r"Hoag+19")

    plt.yscale('linear')
    plt.xscale('linear')
    plt.axis([4.,15.,0.,1.e0])
    plt.xlabel(r"$\mathrm{redshift} \, z$",fontsize=15)
    plt.ylabel(r"$\mathrm{x_{HII}}$",fontsize=15)
    plt.legend()

    plt.savefig(dir+'xHII.pdf')
    if(nodisp==0): plt.show()
    plt.close()
    
def plotJ21(z,J,xHII,dir='./PS/',simname='CoDa III',nodisp=1):
    plt.plot(z,np.log10(J/xHII)+21.,label=r"$\mathrm{"+simname+"}$",color='black')
    zcal11=[5.,6.]   
    lJ21cal11=[-12.15,-12.84]
    for i in range(len(lJ21cal11)) :
        lJ21cal11[i]=lJ21cal11[i]+21.-np.log10(6.38e22*3.e8*2.493e-22) # J21 cal11
    erlJ21cal11=[0.16,0.18] # log error
    plt.errorbar(zcal11,lJ21cal11,yerr=erlJ21cal11,fmt='o',color='r',mec='r',label=r"Calverley+11")
    plt.yscale('linear')
    plt.xscale('linear')
    plt.axis([4.,15.,-3.,0.])
    plt.xlabel(r"$\mathrm{redshift} \, z$",fontsize=15)
    plt.ylabel(r"$\mathrm{log_{10}(J_{21})}$",fontsize=15)
    plt.legend()

    plt.savefig(dir+'J21.pdf')
    if(nodisp==0): plt.show()
    plt.close()
    
def plotgamma(z,J,xHII,dir='./PS/',simname='CoDa III',nodisp=1):
        # check how J21 is computed in aton => should provide an easy way to extract photon density from that
    # => should provide clean, easy way to compute gamma as photd*c*cross-section
    # in aton, photd = 6.326e22 * J21
    #photd = 6.326e22 * J
    #gamma=J* 6.38e22*3.e8*2.493e-22  
    # so in essence, gamma= J * 4.772e9
    # wait is that calculation ok? yeah thats the sigma we use in ATON
    # Im not sure how gamma is measured but is really an "average" value? is it not biased towards something?
    # in SPHINX gamma only considers cells taht are already reionized...
    # Well shit, in Keatings, it says:
    #The evolution of the H i photoionization rate measured in ionized regions compared with measurements from Becker & Bolton
    #(2013), Calverley et al. (2011) and D’Aloisio et al. (2018).
    # So I should average gamma IN IONIZED REGIONS ONLY! 
    # HOW IONIZED? WHERE CAN GAMMA ACTUALLY BE MEASURED?
    # check idfference between gamma average and gamma from ionized regions only
    # how high to you need to push the "ionized criterion" to get the right gamma?
    # can I compute gamma in ionized regions just from teh average stats I have?
    # assuming all ionized regions have some gamma and all neutral have 0
    # add faucher giguere it should be a bit lower so better agreement with us

    gamma=J* 6.38e22*3.e8*2.493e-22 
    plt.plot(z,np.log10(gamma/xHII),label=r"$\mathrm{"+simname+"}$",color='black')

    zcal11=[5.,6.]                                                                                                                 
    lgammacal11=[-12.15,-12.84] # loggamma calverley 2011
    erlgammacal11=[0.16,0.18] # log error

    data = ascii.read(AUX+"/obs/faucher-giguere2008/table1")
    d=data.to_pandas()
    d=d.values
    [zfc08,tauafc08,taubfc08,gamfc08,e_gamfc08]=d.T
    lgamfc08=np.log10(gamfc08)-12.
    erlgamfc08=np.log10(gamfc08+e_gamfc08)-np.log10(gamfc08)

    data = ascii.read(AUX+"/obs/becker_bolton2013/table1")
    d=data.to_pandas()
    d=d.values
    [zbb13,lgambb13,erplgbb13,ermlgbb13]=d

    data = ascii.read(AUX+"/obs/daloisio2018/table2")
    d=data.to_pandas()
    d=d.values
    [zdal18,gamdal18,erpdal18,ermdal18]=d
    lgamdal18=np.log10(gamdal18)
    erpldal18=np.log10(gamdal18+erpdal18)-np.log10(gamdal18)
    ermldal18=(np.log10(gamdal18-ermdal18)-np.log10(gamdal18))


    plt.errorbar(zcal11,lgammacal11,yerr=erlgammacal11,fmt='o',color='r',mec='r',label=r"Calverley11")
    plt.errorbar(zfc08,lgamfc08,yerr=erlgamfc08,fmt='s',color='g',mec='g',label=r"Faucher-Giguere08")
    plt.errorbar(zbb13,lgambb13-12.,yerr=[erplgbb13,-ermlgbb13],fmt='^',color='b',mec='b',label=r"Becker-Bolton13")
    plt.errorbar(zdal18,lgamdal18-12.,yerr=[erpldal18,-ermldal18],fmt='*',color='m',mec='m',label=r"D'Aloisio18")

    plt.yscale('linear')
    plt.xscale('linear')
    plt.axis([4.,12.,-15.,-11.])
    plt.xlabel(r"$\mathrm{redshift} \, z$",fontsize=15)
    plt.ylabel(r"$\mathrm{log_{10}(\Gamma[s^{-1}])}$",fontsize=15)
    plt.legend()

    plt.savefig(dir+'gamma.pdf')
    if(nodisp==0): plt.show()
    plt.close()
    
    
def plotTemp(z,T,dir='./PS/',simname='CoDa III',nodisp=1):
    plt.plot(z,T,label=r"$\mathrm{"+simname+"}$",color='black')

    # Boera+19
    # https://arxiv.org/pdf/1809.06980.pdf 
    zboe19=[4.2,4.6,5.0] 
    Tboe19=[8.13e3,7.31e3,7.37e3]
    erpboe19=[1.34e3,1.35e3,1.67e3]
    ermboe19=[0.97e3,0.88e3,1.39e3]
    plt.errorbar(zboe19,Tboe19,yerr=[erpboe19,ermboe19],fmt='o',color='b',mec='b',label=r"Boera+19")

    # Walther+19
    # https://arxiv.org/pdf/1808.04367.pdf
    zwal19=[4.0,4.2,4.6,5.0,5.4]    
    Twal19=[0.863e4,0.905e4,0.910e4,0.535e4,0.597e4]
    erpwal19=[0.27e4,0.12e4,0.12e4,0.12e4,0.15e4]
    ermwal19=[0.19e4,0.082e4,0.12e4,0.092e4,.13e4]

    plt.errorbar(zwal19,Twal19,yerr=[erpwal19,ermwal19],fmt='v',color='r',mec='r',label=r"Walther+19")

    # Botlon+12
    zbol12=[6.1]
    Tbol12=[7.3e3]
    errTbol12=[0.23e4]
    errzbol12=[0.3]

    plt.errorbar(zbol12,Tbol12,xerr=errzbol12,yerr=errTbol12,fmt='p',color='g',mec='g',label=r"Bolton+12")


    plt.yscale('linear')
    plt.xscale('linear')
    plt.axis([3.9,12.,0.,1.5e4])
    plt.xlabel(r"$\mathrm{redshift} \, z$",fontsize=15)
    plt.ylabel(r"$\mathrm{T(K)}$",fontsize=15)
    plt.legend()

    plt.savefig(dir+'Temp.pdf')
    if(nodisp==0): plt.show()
    plt.close()
    
def plottaues(z,xnH,dir='./PS/',simname='CoDa III',nodisp=1):
    """ Plots tau electron scattering vs tau Planck18"""
    
    c=299792.e3 # speed of light m/s;                                                                                    
    sigmaT=6.65e-29 # Thomson scattering cross-section in m^2 
    #print(c)
    #print(sigmaT)

    aexp=1./(z+1.)
    
    #plt.plot(z,xnH,label=r"$\mathrm{"+simname+"}$",color='black')

    # first thing to do is extend xnH down to z=0
    allzs=np.copy(z)
    zendsim=z[len(z)-1]
    #print(z[len(z)-2]-zendsim)
    allxnHs=np.copy(xnH)
    nzs=3000
    firstlowzs=zendsim-(z[len(z)-2]-zendsim)
    lowzs=np.linspace(firstlowzs,0.,nzs)
    allzs=np.append(allzs,lowzs)
    lowzxnHs=xnH[-1]*(aexp[-1]/(1./(1.+lowzs)))**3
    allxnHs=np.append(allxnHs,lowzxnHs)
    #plt.plot(allzs,allxnHs,label=r"$\mathrm{"+simname+"}$",color='blue', linestyle='dashed')

    #Compute ages => time
    cosmo = COSMO
    allages=cosmo.age(allzs).to(u.s)
    # next compute dtau
    # *1.e6 because xnH is in /cm^3
    # The 0.76 comes from:
    # ATON number density is 0.76 * gas_density_mass  / m_H
    # But we want to account for HeII as well, whichi is 0.24 of the mass but mass_He=4
    # => For 76 H we have only 6 He , hence the correction factor 0.76/0.82
    # Will have to change that if I change the density in ATON from 0.76 to 0.82
    dtau=c*sigmaT*allxnHs[:-1]*np.diff(allages)*1.e6/(0.76/0.82)
    #print(len(np.cumsum(dtau)))
    #print(len(allzs[:-1][::-1]))

    plt.plot(allzs[:-1][::-1],np.cumsum(dtau[::-1]),label=r"$\mathrm{"+simname+"}$",color='black')

    nx=int(50)
    ztau=np.linspace(0.,50.,nx)
    taumax=np.full(nx,0.061)
    taumin=np.full(nx,0.047)

    plt.fill_between(ztau,taumin,taumax,facecolor='green', alpha=0.25,label=r"Planck+18")

    plt.title(r"Electron scattering $\tau$")

    plt.yscale('linear')
    plt.xscale('linear')
    plt.axis([0.,20.,0.,0.1])
    plt.xlabel(r"$\mathrm{redshift} \, z$",fontsize=15)
    plt.ylabel(r"$\mathrm{\tau}$ ",fontsize=15)
    plt.legend()

    plt.savefig(dir+'tau_planck.pdf')
    if(nodisp==0): plt.show()
    plt.close()

    
    
    
def plotallaverages(file="averages00000.txt",plotdir="./PDFs/",nodisp=1,simname="calib"):
    """Plots all that can be plotted with averages"""
    
    try :
        a=os.mkdir(plotdir)
    except FileExistsError:
        print(plotdir+" dir exists")
    
    [aexp,one,nH,xHII,xnH,Temp,J,Z,rhodust,S,nstars]=read_averages(file).T
    z=1./aexp-1.
    xHI=1.-xHII
    photd = 6.326e22 * J
    gamma= J * 6.38e22*3.e8*2.493e-22 # 6.38e22 is from Timothy's thesis
    ne=xnH/0.76 # Because xnH only counts electrons from H, but He is ionized once if H is ionized
    # => need to add He electrons, therefore divide by 0.76
    # S is the source term, intrinsic as due to star formation, in ph/s

    plotxhi(z,xHI,dir=plotdir,simname=simname,nodisp=nodisp)
    plotxhii(z,xHII,dir=plotdir,simname=simname,nodisp=nodisp)
    plotJ21(z,J,xHII,dir=plotdir,simname=simname,nodisp=nodisp)
    plotgamma(z,J,xHII,dir=plotdir,simname=simname,nodisp=nodisp)
    plotTemp(z,Temp,dir=plotdir,simname=simname,nodisp=nodisp)
    plottaues(z,xnH,dir=plotdir,simname=simname,nodisp=nodisp)
