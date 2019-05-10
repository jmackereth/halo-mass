import numpy as np
import isodist
import mwdust
from isodist import FEH2Z, Z2FEH
from scipy.interpolate import interp1d
import multiprocessing
import tqdm
import sys
import apogee.select as apsel
from galpy.util import bovy_coords
from scipy.optimize import newton


def join_on_id(dat1,dat2,joinfield='APOGEE_ID'):
    '''
    Takes two recarrays and joins them based on a ID string (hopefully)
    '''
    #find common fields
    names1 = [dat1.dtype.descr[i][0] for i in range(len(dat1.dtype.descr))]
    names2 = [dat2.dtype.descr[i][0] for i in range(len(dat2.dtype.descr))]
    namesint = np.intersect1d(names1,names2)
    if joinfield not in namesint:
        return NameError('Field '+joinfield+' is not present in both arrays.')
    #work out which fields get appended from dat2
    descr2 = dat2.dtype.descr
    fields_to_append = []
    names_to_append = []
    for i in range(len(names2)):
        if names2[i] not in namesint:
            fields_to_append.append(descr2[i])
            names_to_append.append(names2[i])
        else:
            continue   
    # Faster way to join structured arrays (see https://stackoverflow.com/questions/5355744/numpy-joining-structured-arrays)
    newdtype= dat1.dtype.descr+fields_to_append
    newdata= np.empty(len(dat1),dtype=newdtype)
    for name in dat1.dtype.names:
        newdata[name]= dat1[name]
    for f in names_to_append:
        newdata[f]= np.zeros(len(dat1))-9999.
    dat1= newdata
    
    hash1= dict(zip(dat1[joinfield],
                    np.arange(len(dat1))))
    hash2= dict(zip(dat2[joinfield],
                    np.arange(len(dat2))))
    common, indx1, indx2 = np.intersect1d(dat1[joinfield],dat2[joinfield],return_indices=True)
    for f in names_to_append:
        dat1[f][indx1]= dat2[f][indx2]
    return dat1


#utilities for transformations etc

def transform_aby(xyz,alpha,beta,gamma):
    """
    Transform xyz coordinates by rotation around x-axis (alpha), transformed y-axis (beta) and twice transformed z-axis (gamma)
    """
    Rx = np.zeros([3,3])
    Ry = np.zeros([3,3])
    Rz = np.zeros([3,3])
    Rx[0,0] = 1
    Rx[1] = [0, np.cos(alpha), -np.sin(alpha)]
    Rx[2] = [0, np.sin(alpha), np.cos(alpha)]
    Ry[0] = [np.cos(beta), 0, np.sin(beta)]
    Ry[1,1] = 1
    Ry[2] = [-np.sin(beta), 0, np.cos(beta)]
    Rz[0] = [np.cos(gamma), -np.sin(gamma), 0]
    Rz[1] = [np.sin(gamma), np.cos(gamma), 0]
    Rz[2,2] = 1
    if np.ndim(xyz) == 1:
        tgalcenrect = np.dot(Rx, xyz)
        tgalcenrect = np.dot(Ry, tgalcenrect)
        tgalcenrect = np.dot(Rz, tgalcenrect)
        x, y, z = tgalcenrect[0], tgalcenrect[1], tgalcenrect[2]
    else:
        tgalcenrect = np.einsum('ij,aj->ai', Rz, xyz)
        tgalcenrect = np.einsum('ij,aj->ai', Ry, tgalcenrect)
        tgalcenrect = np.einsum('ij,aj->ai', Rx, tgalcenrect)
        x, y, z = tgalcenrect[:,0], tgalcenrect[:,1], tgalcenrect[:,2]
    return x, y, z

#effective selection function utilities
def generate_isogrid():
    """
    generate a recarray with all the entries from PARSEC isochrones in isodist
    """
    zs = np.arange(0.0005,0.0605, 0.0005)
    zlist = []
    for i in range(len(zs)):
        zlist.append(format(zs[i],'.4f'))
    iso = isodist.PadovaIsochrone(type='2mass-spitzer-wise', Z=zs, parsec=True)

    logages = []
    mets = []
    js = []
    hs = []
    ks = []
    loggs = []
    teffs = []
    imf = []
    deltam = []
    M_ini = []
    M_act = []

    iso_logages = iso._logages
    iso_Zs = iso._ZS
    for i in tqdm.tqdm(range(len(iso_logages))):
        for j in range(len(iso_Zs)):
            thisage = iso_logages[i]
            thisZ = iso_Zs[j]
            thisiso = iso(thisage, Z=thisZ)
            so = np.argsort(thisiso['M_ini'])
            loggs.extend(thisiso['logg'][so][1:])
            logages.extend(thisiso['logage'][so][1:])
            mets.extend(np.ones(len(thisiso['H'][so])-1)*thisZ)
            js.extend(thisiso['J'][so][1:])
            hs.extend(thisiso['H'][so][1:])
            ks.extend(thisiso['Ks'][so][1:])
            teffs.extend(thisiso['logTe'][so][1:])
            imf.extend(thisiso['int_IMF'][so][1:])
            deltam.extend(thisiso['int_IMF'][so][1:]-thisiso['int_IMF'][so][:-1])
            #thisimf = (thisiso['int_IMF'][so][1:]+thisiso['int_IMF'][so][:-1])/2.
            #deltam.extend(thisimf-np.cumsum(thisimf))
            M_ini.extend(thisiso['M_ini'][so][1:])
            M_act.extend(thisiso['M_act'][so][1:])
    logages = np.array(logages)
    mets = np.array(mets)
    js = np.array(js)
    hs = np.array(hs)
    ks = np.array(ks)
    loggs = np.array(loggs)
    teffs = 10**np.array(teffs)
    imf = np.array(imf)
    deltam = np.array(deltam)
    M_ini = np.array(M_ini)
    M_act = np.array(M_act)

    rec = np.recarray(len(deltam), dtype=[('logageyr', float),
                                          ('Z', float),
                                          ('J', float),
                                          ('H', float),
                                          ('K', float),
                                          ('logg', float),
                                          ('teff', float),
                                          ('int_IMF', float),
                                          ('deltaM', float),
                                          ('M_ini', float),
                                          ('M_act', float)])

    rec['logageyr'] = logages
    rec['Z'] = mets
    rec['J'] = js
    rec['H'] = hs
    rec['K'] = ks
    rec['logg'] = loggs
    rec['teff'] = teffs
    rec['int_IMF'] = imf
    rec['deltaM'] = deltam
    rec['M_ini'] = M_ini
    rec['M_act'] = M_act
    return rec

def sampleiso(N, iso, return_inds=False, return_iso=False):
    """
    Sample isochrone recarray iso weighted by lognormal chabrier (2001) IMF
    """
    weights = iso['deltaM']*(10**(iso['logageyr']-9)/iso['Z'])
    sort = np.argsort(weights)
    tinter = interp1d(np.cumsum(weights[sort])/np.sum(weights), range(len(weights[sort])), kind='linear')
    randinds = np.round(tinter(np.random.rand(N))).astype(np.int64)
    if return_inds:
        return randinds, iso['J'][sort][randinds], iso['H'][sort][randinds], iso['K'][sort][randinds]
    elif return_iso:
        return iso[sort][randinds]
    else:
        return iso['J'][sort][randinds], iso['H'][sort][randinds], iso['K'][sort][randinds]
    
def average_mass(iso):
    """
    find the average mass for a given slice of the isochrone recarray
    """
    weights = iso['deltaM']*(10**(iso['logageyr']-9)/iso['Z'])
    return np.sum(iso['M_ini']*weights)/np.sum(weights)

def mass_ratio(iso):
    """
    find the mass ratio between stars in the cuts adopted for giants in APOGEE, and the rest of the isochrones
    """
    weights = iso['deltaM']*(10**(iso['logageyr']-9)/iso['Z'])
    mask = (iso['J']-iso['K'] > 0.3) & (iso['logg'] < 3.0) & (iso['logg'] > 1.0)
    return np.sum(iso['M_ini'][mask]*weights[mask])/np.sum(iso['M_ini']*weights)

def APOGEE_iso_samples(nsamples, rec, fehrange=[-1,-1.5]):
    """
    get samples from the isochrones in the apogee selection (including Log(g) cut for giants, and minimum mass)
    """
    trec = np.copy(rec)
    mask = (trec['J']-trec['K'] > 0.5)  & (Z2FEH(trec['Z']) > fehrange[0]) & (Z2FEH(trec['Z']) < fehrange[1]) & (trec['M_ini'] > 0.75) & (trec['logg'] < 3.) & (trec['logg'] > 1.)
    niso = sampleiso(nsamples,trec[mask], return_iso=True)
    mask = (trec['J']-trec['K'] > 0.3001)  & (Z2FEH(trec['Z']) > fehrange[0]) & (Z2FEH(trec['Z']) < fehrange[1]) & (trec['M_ini'] > 0.75) & (trec['logg'] < 3.) & (trec['logg'] > 1.)
    p3niso = sampleiso(nsamples,trec[mask], return_iso=True)
    return niso, p3niso

#density models
def triaxial(R,phi,z,params=[2.5,1.,1.,0.,0.]):
    """
    general triaxial power-law density model with some rotation about the center of the ellipsoid
    INPUT
        R, phi, z - Galactocentric cylindrical coordinates
        params - [alpha, b, c, beta/(pi/2), gamma/(pi/2)]
    OUTPUT
        density at R, phi, z (no normalisation!)
    """
    alpha = 0. #rotation around x axis (fix to zero? always symmetric around x??)
    beta = params[3]*np.pi/2. #rotate around y
    gamma = params[4]*np.pi/2. #rotate around z
    grid = False
    if np.ndim(R) > 1:
        grid = True
        dim = np.shape(R)
        R = R.reshape(np.product(dim))
        phi = phi.reshape(np.product(dim))
        z = z.reshape(np.product(dim))
    x, y, z = R*np.cos(phi), R*np.sin(phi), z
    xyz = np.dstack([x,y,z])[0]
    x, y, z = transform_aby(xyz, alpha,beta,gamma)
    dens = (x**2+y**2/params[1]**2+z**2/params[2]**2)**params[0]
    dens = dens/np.sqrt(8**2+0.02**2)**params[0]
    if grid:
        dens = dens.reshape(dim)
    return dens

def triaxial_with_spherical_outlier(R,phi,z,params=[2.5,1.,1.,3.,0.1]):
    """
    general triaxial power-law density model with some rotation about the center of the ellipsoid
    INPUT
        R, phi, z - Galactocentric cylindrical coordinates
        params - [alpha, b, c, alpha_outlier, epsilon]
    OUTPUT
        density at R, phi, z (no normalisation!)
    """
    grid = False
    if np.ndim(R) > 1:
        grid = True
        dim = np.shape(R)
        R = R.reshape(np.product(dim))
        phi = phi.reshape(np.product(dim))
        z = z.reshape(np.product(dim))
    x, y, z = R*np.cos(phi), R*np.sin(phi), z
    xyz = np.dstack([x,y,z])[0]
    dens = (x**2+y**2/params[1]**2+z**2/params[2]**2)**params[0]
    outdens = (x**2+y**2+z**2)**params[3]
    dens = (1-params[4])*dens+params[4]*outdens
    dens = dens/np.sqrt(8**2+0.02**2)**params[0]
    if grid:
        dens = dens.reshape(dim)
    return dens

def triaxial_with_fixed_spherical_outlier(R,phi,z,params=[2.5,1.,1.,0.1], just_main=False, just_outlier=False):
    """
    general triaxial power-law density model with some rotation about the center of the ellipsoid
    INPUT
        R, phi, z - Galactocentric cylindrical coordinates
        params - [alpha, b, c, alpha_outlier, epsilon]
    OUTPUT
        density at R, phi, z (no normalisation!)
    """
    grid = False
    if np.ndim(R) > 1:
        grid = True
        dim = np.shape(R)
        R = R.reshape(np.product(dim))
        phi = phi.reshape(np.product(dim))
        z = z.reshape(np.product(dim))
    x, y, z = R*np.cos(phi), R*np.sin(phi), z
    xyz = np.dstack([x,y,z])[0]
    dens = (x**2+y**2/params[1]**2+z**2/params[2]**2)**params[0]
    outdens = (x**2+y**2+z**2)**-4.
    sundens = (1-params[3])*((8.**2+0.02**2/params[2]**2)**params[0])+params[3]*((8.**2+0.02**2)**-4.)
    if just_main:
        dens = (1-params[3])*dens
    elif just_outlier:
        dens = params[3]*outdens
    else:
        dens = ((1-params[3])*dens+params[3]*outdens)/sundens
    if grid:
        dens = dens.reshape(dim)
    return dens

def triaxial_with_expdisk_outlier(R,phi,z,params=[2.5,1.,1.,0.1], just_main=False, just_outlier=False, outhr=3., outhz=2.):
    """
    general triaxial power-law density model with some rotation about the center of the ellipsoid
    INPUT
        R, phi, z - Galactocentric cylindrical coordinates
        params - [alpha, b, c, alpha_outlier, epsilon]
    OUTPUT
        density at R, phi, z (no normalisation!)
    """
    grid = False
    if np.ndim(R) > 1:
        grid = True
        dim = np.shape(R)
        R = R.reshape(np.product(dim))
        phi = phi.reshape(np.product(dim))
        z = z.reshape(np.product(dim))
    x, y, z = R*np.cos(phi), R*np.sin(phi), z
    xyz = np.dstack([x,y,z])[0]
    dens = (1-params[3])*((x**2+y**2/params[1]**2+z**2/params[2]**2)**params[0])/((1-params[3])*((8.**2+0.02**2/params[2]**2)**params[0])+params[3])
    outdens = params[3]*np.exp(-1*(R-8.)/outhr-np.fabs(z)/outhz)/((1-params[3])*((8.**2+0.02**2/params[2]**2)**params[0])+params[3])
    #sundens = (1-params[3])*((8.**2+0.02**2/params[2]**2)**params[0])+params[3]*((8.**2+0.02**2)**-0.4)
    if just_main:
        dens = dens
    elif just_outlier:
        dens = outdens
    else:
        dens = dens+outdens
    if grid:
        dens = dens.reshape(dim)
    return dens

def spherical(R,phi,z,params=[2.5,]):
    """
    general spherical power-law density model
    INPUT
        R, phi, z - Galactocentric cylindrical coordinates
        params - [alpha,]
    OUTPUT
        density at R, phi, z (no normalisation!)
    """
    grid = False
    if np.ndim(R) > 1:
        grid = True
        dim = np.shape(R)
        R = R.reshape(np.product(dim))
        phi = phi.reshape(np.product(dim))
        z = z.reshape(np.product(dim))
    x, y, z = R*np.cos(phi), R*np.sin(phi), z
    dens = (x**2+y**2+z**2)**params[0]
    dens = dens/np.sqrt(8**2+0.02**2)**params[0]
    if grid:
        dens = dens.reshape(dim)
    return dens

def axisymmetric(R,phi,z,params=[2.5,1.]):
    """
    general axisymmetric power-law density model
    INPUT
        R, phi, z - Galactocentric cylindrical coordinates
        params - [alpha,c]
    OUTPUT
        density at R, phi, z (no normalisation!)
    """
    grid = False
    if np.ndim(R) > 1:
        grid = True
        dim = np.shape(R)
        R = R.reshape(np.product(dim))
        phi = phi.reshape(np.product(dim))
        z = z.reshape(np.product(dim))
    x, y, z = R*np.cos(phi), R*np.sin(phi), z
    dens = (x**2+y**2+z**2/params[1]**2)**params[0]
    dens = dens/np.sqrt(8**2+0.02**2)**params[0]
    if grid:
        dens = dens.reshape(dim)
    return dens

def triaxial_norot(R,phi,z,params=[2.5,1.,1.]):
    """
    general triaxial power-law density model (no rotation)
    INPUT
        R, phi, z - Galactocentric cylindrical coordinates
        params - [alpha,b,c]
    OUTPUT
        density at R, phi, z (no normalisation!)
    """
    grid = False
    if np.ndim(R) > 1:
        grid = True
        dim = np.shape(R)
        R = R.reshape(np.product(dim))
        phi = phi.reshape(np.product(dim))
        z = z.reshape(np.product(dim))
    x, y, z = R*np.cos(phi), R*np.sin(phi), z
    dens = (x**2+y**2/params[1]**2+z**2/params[2]**2)**params[0]
    dens = dens/np.sqrt(8**2+0.02/params[2]**2)**params[0]
    if grid:
        dens = dens.reshape(dim)
    return dens

def triaxial_iorio(R,phi,z,params=[1/10.,2.,3.,0.5,0.5]):
    grid = False
    r_eb = 1/params[0]
    if np.ndim(R) > 1:
        grid = True
        dim = np.shape(R)
        R = R.reshape(np.product(dim))
        phi = phi.reshape(np.product(dim))
        z = z.reshape(np.product(dim))
    x, y, z = R*np.cos(phi), R*np.sin(phi), z
    r_e = np.sqrt(x**2+y**2/params[3]**2+z**2/params[4]**2)
    r_e_sun = np.sqrt(8.**2+0.02**2/params[4]**2)
    dens = (r_e/r_eb)**(-params[1])*(1+r_e/r_eb)**(-(params[2]-params[1]))
    sundens = (r_e_sun/r_eb)**(-params[1])*(1+r_e_sun/r_eb)**(-(params[2]-params[1]))
    dens = dens/sundens
    if grid:
        dens = dens.reshape(dim)
    return dens


def q_re(re, q_0, q_inf, req):
    return q_inf - (q_inf-q_0)*np.exp(1-(np.sqrt(re**2+req**2)/req))

def r_e(re,x,y,z,p,q_0,q_inf,req):
    return re**2-x**2-y**2*p**-2-z**2*q_re(re,q_0,q_inf,req)**-2

def triaxialSP_iorio_qvar(R,phi,z,params=[2.,0.5,0.5,0.8,1/10.,0.,0.,0.]):
    grid = False
    alpha = (np.pi*params[5])-np.pi/2.
    beta = (np.pi*params[6])-np.pi/2.
    gamma = (np.pi*params[7])-np.pi/2.
    if np.ndim(R) > 1:
        grid = True
        dim = np.shape(R)
        R = R.reshape(np.product(dim))
        phi = phi.reshape(np.product(dim))
        z = z.reshape(np.product(dim))
    x, y, z = R*np.cos(phi), R*np.sin(phi), z
    x, y, z = transform_aby(np.dstack([x,y,z])[0], alpha,beta,gamma)
    xyz = np.dstack([x,y,z])[0]
    r = np.linalg.norm(xyz, axis=1)
    q_r = q_re(r,params[2],params[3],params[4])
    tr_e = np.sqrt(x**2+y**2/params[2]**2+z**2/q_r**2)
    txsun, tysun, tzsun = transform_aby([8.,0.,0.02], alpha,beta,gamma)
    r_e_sun = np.sqrt(txsun**2+tysun**2/params[2]**2+tzsun**2/q_r**2)
    dens = (tr_e)**(-params[0])
    sundens = (r_e_sun)**(-params[0])
    dens = dens/sundens
    if grid:
        dens = dens.reshape(dim)
    return dens


#Maximum likelihood utilities

# Likelihood calculation

def Rphizgrid(apo,distmods):
    """
    Generates a grid of R, phi, z for each location in apo at the distance moduli supplied
    """
    ds = 10**(distmods/5.-2.)
    Rgrid = np.zeros((len(apo._locations),len(ds)))
    phigrid = np.zeros((len(apo._locations),len(ds)))
    zgrid = np.zeros((len(apo._locations),len(ds)))
    for i in range(len(apo._locations)):
        glon,glat = apo.glonGlat(apo._locations[i])
        glon = np.ones(len(ds))*glon[0]
        glat = np.ones(len(ds))*glat[0]
        xyz = bovy_coords.lbd_to_XYZ(glon,glat,ds, degree=True)
        rphiz = bovy_coords.XYZ_to_galcencyl(xyz[:,0], xyz[:,1], xyz[:,2], Xsun=8., Zsun=0.02)
        Rgrid[i] = rphiz[:,0]
        phigrid[i] = rphiz[:,1]
        zgrid[i] = rphiz[:,2]
    return Rgrid, phigrid, zgrid

def xyzgrid(apo,distmods):
    """
    Generates a grid of x, y, z for each location in apo at the distance moduli supplied
    """
    ds = 10**(distmods/5.-2.)
    xgrid = np.zeros((len(apo._locations),len(ds)))
    ygrid = np.zeros((len(apo._locations),len(ds)))
    zgrid = np.zeros((len(apo._locations),len(ds)))
    for i in range(len(apo._locations)):
        glon,glat = apo.glonGlat(apo._locations[i])
        glon = np.ones(len(ds))*glon[0]
        glat = np.ones(len(ds))*glat[0]
        xyz = bovy_coords.lbd_to_XYZ(glon,glat,ds,degree=True)
        xgrid[i] = xyz[:,0]
        ygrid[i] = xyz[:,1]
        zgrid[i] = xyz[:,2]
    return zgrid, ygrid, zgrid



def tdens(densfunc, Rgrid, phigrid, zgrid, params=None):
    """
    returns the densities at the supplied grid given the supplied densfunc
    """
    if params is None:
        dens = densfunc(Rgrid,phigrid,zgrid)
    else:
        dens = densfunc(Rgrid,phigrid,zgrid,params=params)
    return dens

def effvol(densfunc, effsel, Rgrid, phigrid, zgrid, params=None):
    """
    returns the effective volume given a density function, an effective selection function 
    (including D**2 deltaD factor...) and the grid that it was evaluated on.
    """
    if params is None:
        effdens = tdens(densfunc,Rgrid,phigrid,zgrid)
    else:
        effdens = tdens(densfunc,Rgrid,phigrid,zgrid,params=params)
    return np.sum(effdens*effsel)
    
def loglike(params, densfunc, effsel, Rgrid, phigrid, zgrid, dataR, dataphi, dataz):
    """
    log-likelihood for the inhomogeneous Poisson point process
    """
    if not check_prior(densfunc, params):
        return -np.inf
    logdatadens = np.log(tdens(densfunc, dataR, dataphi, dataz, params=params))
    logeffvol = np.log(effvol(densfunc,effsel,Rgrid,phigrid,zgrid,params=params))
    loglike = np.sum(logdatadens)-len(dataR)*logeffvol
    if not np.isfinite(loglike):
        return -np.inf
    return loglike

def check_prior(densfunc, params):
    """
    check the (uninformative?) prior for the given density model and parameters.
    """
    if densfunc is triaxial:
        if params[0] > 0.:return False
        elif params[1] < 0.:return False
        elif params[1] > 1.:return False
        elif params[2] < 0.:return False
        elif params[2] > 1.:return False
        elif params[3] < 0.: return False
        elif params[3] > 1.: return False
        elif params[4] < 0.: return False
        elif params[4] > 1.:return False
        elif params[3] < params[4]: return False
        else: return True
    if densfunc is spherical:
        if params[0] > 0.:return False
        else: return True
    if densfunc is axisymmetric:
        if params[0] > 0.:return False
        elif params[1] < 0.:return False
        elif params[1] > 1.:return False
        else: return True
    if densfunc is triaxial_norot:
        if params[0] > 0.:return False
        elif params[1] < 0.:return False
        elif params[1] > 1.:return False
        elif params[2] < 0.:return False
        elif params[2] > 1.:return False
        else: return True
    if densfunc is triaxial_with_spherical_outlier:
        if params[0] > 0.:return False
        elif params[1] < 0.:return False
        elif params[1] > 1.:return False
        elif params[2] < 0.:return False
        elif params[2] > 1.:return False
        elif params[3] > 0.:return False
        elif params[4] < 0.: return False
        elif params[4] > 1.: return False
        else: return True
    if densfunc is triaxial_with_fixed_spherical_outlier:
        if params[0] > 0.:return False
        elif params[1] < 0.:return False
        elif params[1] > 1.:return False
        elif params[2] < 0.:return False
        elif params[2] > 1.:return False
        elif params[3] < 0.: return False
        elif params[3] > 1.: return False
        else: return True
    if densfunc is triaxial_with_expdisk_outlier:
        if params[0] > 0.:return False
        elif params[1] < 0.:return False
        elif params[1] > 1.:return False
        elif params[2] < 0.:return False
        elif params[2] > 1.:return False
        elif params[3] < 0.: return False
        elif params[3] > 1.: return False
        else: return True
    if densfunc is triaxial_iorio:
        if 1/params[0] < 0.:return False
        elif params[1] < 0.:return False
        elif params[2] < 0.:return False
        else: return True
    if densfunc is triaxialSP_iorio_qvar:
        if params[0] < 0.:return False
        elif params[1] < 0.1:return False
        elif params[1] > 10.:return False
        elif params[2] < 0.1:return False
        elif params[2] > 10.:return False
        elif params[3] < 0.1:return False
        elif params[3] > 10.:return False
        elif params[4] < 0.1:return False
        elif params[4] > 100.:return False
        elif params[5] < 0.1:return False
        elif params[5] > 0.9:return False
        elif params[6] < 0.1:return False
        elif params[6] > 0.9:return False
        elif params[7] < 0.1:return False
        elif params[7] > 0.9:return False
        else:return True
    return True
        
       
def mloglike(*args, **kwargs):
    """
    return the negative log-likehood
    """
    return -loglike(*args,**kwargs)

