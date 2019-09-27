import numpy as np
from galpy.util import bovy_coords, _rotate_to_arbitrary_vector
from scipy.optimize import newton
from scipy.special import erfinv

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
    R = np.matmul(Rx,np.matmul(Ry,Rz))
    if np.ndim(xyz) == 1:
        tgalcenrect = np.dot(R, xyz)
        x, y, z = tgalcenrect[0], tgalcenrect[1], tgalcenrect[2]
    else:
        tgalcenrect = np.einsum('ij,aj->ai', R, xyz)
        x, y, z = tgalcenrect[:,0], tgalcenrect[:,1], tgalcenrect[:,2]
    return x, y, z

def transform_zvecpa(xyz,zvec,pa):
    """
    transform coordinates using the axis-angle method
    """
    pa_rot= np.array([[np.cos(pa),np.sin(pa),0.],
                         [-np.sin(pa),np.cos(pa),0.],
                         [0.,0.,1.]])

    zvec/= np.sqrt(np.sum(zvec**2.))
    zvec_rot= _rotate_to_arbitrary_vector(np.array([[0.,0.,1.]]),zvec,inv=True)[0]
    trot= np.dot(pa_rot,zvec_rot)
    if np.ndim(xyz) == 1:
        tgalcenrect = np.dot(trot, xyz)
        x, y, z = tgalcenrect[0], tgalcenrect[1], tgalcenrect[2]
    else:
        tgalcenrect = np.einsum('ij,aj->ai', trot, xyz)
        x, y, z = tgalcenrect[:,0], tgalcenrect[:,1], tgalcenrect[:,2]
    return x, y, z

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
    dens = (x**2+y**2+z**2)**(-params[0])
    dens = dens/(np.sqrt(8**2+0.02**2)**(-params[0]))
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
    dens = np.sqrt(x**2+y**2+z**2/params[1]**2)**-params[0]
    dens = dens/np.sqrt(8**2+0.02**2/params[1]**2)**-params[0]
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
    dens = np.sqrt(x**2+y**2/params[1]**2+z**2/params[2]**2)**-params[0]
    dens = dens/np.sqrt(8**2+0.02**2/params[2]**2)**-params[0]
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

def triaxial_single_angle(R,phi,z,params=[2.,0.5,0.5,0.5,0.5,0.5]):
    grid = False
    alpha = 0.9*np.pi*params[3]+0.05*np.pi-np.pi/2.
    beta = 0.9*np.pi*params[4]+0.05*np.pi-np.pi/2.
    gamma = 0.9*np.pi*params[5]+0.05*np.pi-np.pi/2.
    if np.ndim(R) > 1:
        grid = True
        dim = np.shape(R)
        R = R.reshape(np.product(dim))
        phi = phi.reshape(np.product(dim))
        z = z.reshape(np.product(dim))
    x, y, z = R*np.cos(phi), R*np.sin(phi), z
    x, y, z = transform_aby(np.dstack([x,y,z])[0], alpha,beta,gamma)
    r_e = np.sqrt(x**2+y**2/params[1]**2+z**2/params[2]**2)
    r_e_sun = np.sqrt(8.**2+0.02**2/params[2]**2)
    dens = (r_e)**(-params[0])
    sundens = (r_e_sun)**(-params[0])
    dens = dens/sundens
    if grid:
        dens = dens.reshape(dim)
    return dens

def triaxial_single_angle_zvecpa(R,phi,z,params=[2.,0.5,0.5,0.,0.,0.]):
    grid = False
    theta = params[3]*2*np.pi
    tz = (params[4]*2)-1
    zvec = np.array([np.sqrt(1-tz**2)*np.cos(theta), np.sqrt(1-tz**2)*np.sin(theta), tz])
    pa = (params[5]*np.pi)#-np.pi/2.
    if np.ndim(R) > 1:
        grid = True
        dim = np.shape(R)
        R = R.reshape(np.product(dim))
        phi = phi.reshape(np.product(dim))
        z = z.reshape(np.product(dim))
    x, y, z = R*np.cos(phi), R*np.sin(phi), z
    x, y, z = transform_zvecpa(np.dstack([x,y,z])[0], zvec,pa)
    xsun,ysun,zsun = transform_zvecpa([8.,0.,0.02],zvec,pa)
    r_e = np.sqrt(x**2+y**2/params[1]**2+z**2/params[2]**2)
    r_e_sun = np.sqrt(xsun**2+ysun**2/params[1]**2+zsun**2/params[2]**2)
    dens = (r_e)**(-params[0])
    sundens = (r_e_sun)**(-params[0])
    dens = dens/sundens
    if grid:
        dens = dens.reshape(dim)
    return dens

def justexpdisk(R,phi,z,params=[1/1.8,1/0.8]):
    _R0 = 8.
    hr = 1/1.8
    hz = 1/0.8
    diskdens = np.exp(-hr*(R-_R0)-hz*np.fabs(z))
    diskdens_sun = np.exp(-hr*(_R0-_R0)-hz*np.fabs(0.02))
    return diskdens/diskdens_sun


def triaxial_single_angle_zvecpa_plusexpdisk(R,phi,z,params=[2.,0.5,0.5,0.,0.,0.,0.01],split=False):
    _R0 = 8.
    original_z = np.copy(z)
    grid = False
    theta = params[3]*2*np.pi
    tz = (params[4]*2)-1
    zvec = np.array([np.sqrt(1-tz**2)*np.cos(theta), np.sqrt(1-tz**2)*np.sin(theta), tz])
    pa = (params[5]*np.pi)#-np.pi/2.
    if np.ndim(R) > 1:
        grid = True
        dim = np.shape(R)
        R = R.reshape(np.product(dim))
        phi = phi.reshape(np.product(dim))
        z = z.reshape(np.product(dim))
        original_z = original_z.reshape(np.product(dim))
    x, y, z = R*np.cos(phi), R*np.sin(phi), z
    x, y, z = transform_zvecpa(np.dstack([x,y,z])[0], zvec,pa)
    xsun,ysun,zsun = transform_zvecpa([8.,0.,0.02],zvec,pa)
    r_e = np.sqrt(x**2+y**2/params[1]**2+z**2/params[2]**2)
    r_e_sun = np.sqrt(xsun**2+ysun**2/params[1]**2+zsun**2/params[2]**2)
    hr = 1/2.
    hz = 1/0.8
    diskdens = np.exp(-hr*(R-_R0)-hz*np.fabs(original_z))
    diskdens_sun = np.exp(-hr*(_R0-_R0)-hz*np.fabs(0.02))
    dens = (r_e)**(-params[0])
    sundens = (r_e_sun)**(-params[0])
    if split:
        dens, diskdens = (1-params[6])*dens/sundens, (params[6])*diskdens/diskdens_sun
        if grid:
            dens = dens.reshape(dim)
            diskdens = diskdens.reshape(dim)
        return dens, diskdens
    else:
        dens = (1-params[6])*dens/sundens+(params[6]*diskdens/diskdens_sun)
        #dens = ((1-params[6])*dens+params[6]*diskdens)/((1-params[6])*sundens+params[6]*diskdens_sun)
        if grid:
            dens = dens.reshape(dim)
        return dens
    
def triaxial_single_cutoff_zvecpa_plusexpdisk(R,phi,z,params=[2.,1.,0.5,0.5,0.,0.,0.,0.01],split=False):
    _R0 = 8.
    original_z = np.copy(z)
    grid = False
    theta = params[4]*2*np.pi
    tz = (params[5]*2)-1
    zvec = np.array([np.sqrt(1-tz**2)*np.cos(theta), np.sqrt(1-tz**2)*np.sin(theta), tz])
    pa = (params[6]*np.pi)#-np.pi/2.
    if np.ndim(R) > 1:
        grid = True
        dim = np.shape(R)
        R = R.reshape(np.product(dim))
        phi = phi.reshape(np.product(dim))
        z = z.reshape(np.product(dim))
        original_z = original_z.reshape(np.product(dim))
    x, y, z = R*np.cos(phi), R*np.sin(phi), z
    x, y, z = transform_zvecpa(np.dstack([x,y,z])[0], zvec,pa)
    xsun,ysun,zsun = transform_zvecpa([8.,0.,0.02],zvec,pa)
    r_e = np.sqrt(x**2+y**2/params[2]**2+z**2/params[3]**2)
    r_e_sun = np.sqrt(xsun**2+ysun**2/params[2]**2+zsun**2/params[3]**2)
    hr = 1/2.
    hz = 1/0.8
    diskdens = np.exp(-hr*(R-_R0)-hz*np.fabs(original_z))
    diskdens_sun = np.exp(-hr*(_R0-_R0)-hz*np.fabs(0.02))
    dens = (r_e)**(-params[0])*np.exp(-params[1]*r_e)
    sundens = (r_e_sun)**(-params[0])*np.exp(-params[1]*r_e_sun)
    if split:
        dens, diskdens = (1-params[7])*dens/sundens, (params[7])*diskdens/diskdens_sun
        if grid:
            dens = dens.reshape(dim)
            diskdens = diskdens.reshape(dim)
        return dens, diskdens
    else:
        dens = (1-params[7])*dens/sundens+(params[7]*diskdens/diskdens_sun)
        #dens = ((1-params[6])*dens+params[6]*diskdens)/((1-params[6])*sundens+params[6]*diskdens_sun)
        if grid:
            dens = dens.reshape(dim)
        return dens
    
def triaxial_broken_angle_zvecpa_plusexpdisk(R,phi,z,params=[2.,3.,20.,0.5,0.5,0.,0.,0.,0.01],split=False):
    _R0 = 8.
    original_z = np.copy(z)
    grid = False
    theta = params[5]*2*np.pi
    tz = (params[6]*2)-1
    zvec = np.array([np.sqrt(1-tz**2)*np.cos(theta), np.sqrt(1-tz**2)*np.sin(theta), tz])
    pa = (params[7]*np.pi)#-np.pi/2.
    if np.ndim(R) > 1:
        grid = True
        dim = np.shape(R)
        R = R.reshape(np.product(dim))
        phi = phi.reshape(np.product(dim))
        z = z.reshape(np.product(dim))
        original_z = original_z.reshape(np.product(dim))
    x, y, z = R*np.cos(phi), R*np.sin(phi), z
    x, y, z = transform_zvecpa(np.dstack([x,y,z])[0], zvec,pa)
    xsun,ysun,zsun = transform_zvecpa([8.,0.,0.02],zvec,pa)
    r_e = np.sqrt(x**2+y**2/params[3]**2+z**2/params[4]**2)
    r_e_sun = np.sqrt(xsun**2+ysun**2/params[3]**2+zsun**2/params[4]**2)
    hr = 1/2.
    hz = 1/0.8
    diskdens = np.exp(-hr*(R-_R0)-hz*np.fabs(original_z))
    diskdens_sun = np.exp(-hr*(_R0-_R0)-hz*np.fabs(0.02))
    dens = np.zeros(len(r_e))
    dens[r_e < params[2]] = (r_e[r_e < params[2]])**(-params[0])
    dens[r_e > params[2]] = (params[2])**(params[1]-params[0])*(r_e[r_e > params[2]])**(-params[1])
    if params[2] > r_e_sun:
        sundens = (params[2])**(params[1]-params[0])*(r_e_sun)**(-params[1])
    else:
        sundens = (r_e_sun)**(-params[0])
    if split:
        dens, diskdens = (1-params[8])*dens/sundens, (params[8])*diskdens/diskdens_sun
        if grid:
            dens = dens.reshape(dim)
            diskdens = diskdens.reshape(dim)
        return dens, diskdens
    else:
        dens = (1-params[8])*dens/sundens+(params[8]*diskdens/diskdens_sun)
        #dens = ((1-params[6])*dens+params[6]*diskdens)/((1-params[6])*sundens+params[6]*diskdens_sun)
        if grid:
            dens = dens.reshape(dim)
        return dens



def spherical(R,phi,z,params=[2.]):
    grid = False
    x, y, z = R*np.cos(phi), R*np.sin(phi), z
    r_e = np.sqrt(x**2+y**2+z**2)
    r_e_sun = np.sqrt(8.**2+0.02**2)
    dens = (r_e)**(-params[0])
    sundens = (r_e_sun)**(-params[0])
    dens = dens/sundens
    if grid:
        dens = dens.reshape(dim)
    return dens

def triaxial_double_angle(R,phi,z,params=[1/10.,2.,3.,0.5,0.5,0.5,0.5,0.5]):
    grid = False
    r_eb = 1/params[0]
    alpha = (0.8*np.pi*params[5]+0.1*np.pi)
    beta = (0.8*np.pi*params[6]+0.1*np.pi)
    gamma = (0.8*np.pi*params[7]+0.1*np.pi)
    if np.ndim(R) > 1:
        grid = True
        dim = np.shape(R)
        R = R.reshape(np.product(dim))
        phi = phi.reshape(np.product(dim))
        z = z.reshape(np.product(dim))
    x, y, z = R*np.cos(phi), R*np.sin(phi), z
    x, y, z = transform_aby(np.dstack([x,y,z])[0], alpha,beta,gamma)
    r_e = np.sqrt(x**2+y**2/params[3]**2+z**2/params[4]**2)
    r_e_sun = np.sqrt(8.**2+0.02**2/params[4]**2)
    dens = (r_e/r_eb)**(-params[1])*(r_e/r_eb)**(-(params[2]-params[1]))
    sundens = (r_e_sun/r_eb)**(-params[1])*(r_e_sun/r_eb)**(-(params[2]-params[1]))
    dens = dens/sundens
    if grid:
        dens = dens.reshape(dim)
    return dens

def triaxial_double_angle_zvecpa(R,phi,z,params=[1/10.,2.,3.,0.5,0.5,0.5,0.5,0.5,0.5]):
    grid = False
    r_eb = 1/params[0]
    zvec = np.sqrt(2)*erfinv(2*np.array([params[5],params[6],params[7]])-1.)
    #zvec = (rvec.T/numpy.linalg.norm(rvec, axis=1)).T
    pa = params[8]*2*np.pi
    if np.ndim(R) > 1:
        grid = True
        dim = np.shape(R)
        R = R.reshape(np.product(dim))
        phi = phi.reshape(np.product(dim))
        z = z.reshape(np.product(dim))
    x, y, z = R*np.cos(phi), R*np.sin(phi), z
    x, y, z = transform_zvecpa(np.dstack([x,y,z])[0], zvec,pa)
    r_e = np.sqrt(x**2+y**2/params[3]**2+z**2/params[4]**2)
    r_e_sun = np.sqrt(8.**2+0.02**2/params[4]**2)
    dens = (r_e/r_eb)**(-params[1])*(r_e/r_eb)**(-(params[2]-params[1]))
    sundens = (r_e_sun/r_eb)**(-params[1])*(r_e_sun/r_eb)**(-(params[2]-params[1]))
    dens = dens/sundens
    if grid:
        dens = dens.reshape(dim)
    return dens

def triaxial_einasto_zvecpa(R,phi,z,params=[10.,3.,0.8,0.8,0.,0.99,0.]):
    grid = False
    r_eb = params[0]
    n = params[1]
    p = params[2]
    q = params[3]
    theta = params[4]*2*np.pi
    tz = (params[5]*2)-1
    zvec = np.array([np.sqrt(1-tz**2)*np.cos(theta), np.sqrt(1-tz**2)*np.sin(theta), tz])
    pa = (params[6]*np.pi)#-np.pi/2.
    if np.ndim(R) > 1:
        grid = True
        dim = np.shape(R)
        R = R.reshape(np.product(dim))
        phi = phi.reshape(np.product(dim))
        z = z.reshape(np.product(dim))
    x, y, z = R*np.cos(phi), R*np.sin(phi), z
    x, y, z = transform_zvecpa(np.dstack([x,y,z])[0], zvec,pa)
    xsun,ysun,zsun = transform_zvecpa([8.,0.,0.02],zvec,pa)
    r_e = np.sqrt(x**2+y**2/p**2+z**2/q**2)
    r_e_sun = np.sqrt(xsun**2+ysun**2/p**2+zsun**2/q**2)
    dn = 3*n - 1./3. + 0.0079/n
    dens = np.exp(-dn*((r_e/r_eb)**(1/n)-1))
    sundens = np.exp(-dn*((r_e_sun/r_eb)**(1/n)-1))
    dens = dens/sundens
    if grid:
        dens = dens.reshape(dim)
    return dens

def triaxial_einasto_zvecpa_plusexpdisk(R,phi,z,params=[10.,3.,0.8,0.8,0.,0.99,0.,0.], split=False):
    grid = False
    _R0 = 8.
    original_z = np.copy(z)
    r_eb = params[0]
    n = params[1]
    p = params[2]
    q = params[3]
    theta = params[4]*2*np.pi
    tz = (params[5]*2)-1
    zvec = np.array([np.sqrt(1-tz**2)*np.cos(theta), np.sqrt(1-tz**2)*np.sin(theta), tz])
    pa = (params[6]*np.pi)#-np.pi/2.
    if np.ndim(R) > 1:
        grid = True
        dim = np.shape(R)
        R = R.reshape(np.product(dim))
        phi = phi.reshape(np.product(dim))
        z = z.reshape(np.product(dim))
        original_z = original_z.reshape(np.product(dim))
    x, y, z = R*np.cos(phi), R*np.sin(phi), z
    x, y, z = transform_zvecpa(np.dstack([x,y,z])[0], zvec,pa)
    xsun,ysun,zsun = transform_zvecpa([8.,0.,0.02],zvec,pa)
    r_e = np.sqrt(x**2+y**2/p**2+z**2/q**2)
    r_e_sun = np.sqrt(xsun**2+ysun**2/p**2+zsun**2/q**2)
    dn = 3.*n - 1./3. + 0.0079/n
    dens = np.exp(-dn*((r_e/r_eb)**(1/n)-1))
    sundens = np.exp(-dn*((r_e_sun/r_eb)**(1/n)-1))
    hr = 1/2.
    hz = 1/0.8
    diskdens = np.exp(-hr*(R-_R0)-hz*np.fabs(original_z))
    diskdens_sun = np.exp(-hr*(_R0-_R0)-hz*np.fabs(0.02))
    if split:
        dens, diskdens = (1-params[7])*dens/sundens, (params[7])*diskdens/diskdens_sun
        if grid:
            dens = dens.reshape(dim)
            diskdens = diskdens.reshape(dim)
        return dens, diskdens
    else:
        dens = (1-params[6])*dens/sundens+(params[6]*diskdens/diskdens_sun)
        if grid:
            dens = dens.reshape(dim)
        return dens
    return dens

def q_re(re, q_0, q_inf, req):
    return q_inf - (q_inf-q_0)*np.exp(1-(np.sqrt(re**2+req**2)/req))

def r_e(re,x,y,z,p,q_0,q_inf,req):
    return re**2-x**2-y**2*p**-2-z**2*q_re(re,q_0,q_inf,req)**-2

def triaxial_single_angle_qvar(R,phi,z,params=[2.,0.5,0.5,0.8,1/10.,0.5,0.5,0.5], sun_norm=True):
    grid = False
    alpha = (0.8*np.pi*params[5]+0.1*np.pi)-np.pi/2.
    beta = (0.8*np.pi*params[6]+0.1*np.pi)-np.pi/2.
    gamma = (0.8*np.pi*params[7]+0.1*np.pi)-np.pi/2.
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
    q_r = q_re(r,params[2],params[3],1/params[4])
    tr_e = np.sqrt(x**2+y**2/params[1]**2+z**2/q_r**2)
    txsun, tysun, tzsun = transform_aby([8.,0.,0.02], alpha,beta,gamma)
    r_e_sun = np.sqrt(txsun**2+tysun**2/params[1]**2+tzsun**2/q_r**2)
    dens = (tr_e)**(-params[0])
    sundens = (r_e_sun)**(-params[0])
    if sun_norm:
        dens = dens/sundens
    if grid:
        dens = dens.reshape(dim)
    return dens

