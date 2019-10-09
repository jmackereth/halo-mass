# halo-mass - code for reproduction of Mackereth et al. (2019c)

This repository contains code which facilitates the reproduction (or re-mixing) of the results presented in the paper 'Weighing the stellar constituents of the Galactic halo with APOGEE red giant stars' (Mackereth et al. 2019c). Specifically, by running code in the notebooks here you will: 
* compute the APOGEE DR14 selection function 
* fit the density of APOGEE red giants in the halo, both for the total population, and mono-abundance populations (MAPs) at low and high orbital eccentricity
* estimate the mass which is contained in the stellar halo MAPs, estimating the total stellar halo mass, and the mass of accreted populations, setting a limit on the mass of the *Gaia-Enceladus*/Sausage debris

While I offer no guarantee that the code will run perfectly (this code will not be maintained, but please feel free to make  any queries), following the instructions here should allow you to reproduce the results with minimal effort. For simplicity, all code runs in jupyter notebooks. Some of the code can take some considerable time to run.

### Dependencies

First, make sure you have the usual `scipy` stack installed (e.g. `numpy`, `scipy`, `matplotlib`) along with a working install of `astropy` (likely already installed, if you use anaconda, for example). You will also need to have a working installation of [@jobovy](https://github.com/jobovy)'s [`galpy`](https://github.com/jobovy/galpy), [`apogee`](https://github.com/jobovy/apogee), [`mwdust`](https://github.com/jobovy/mwdust) and [`gaia_tools`](https://github.com/jobovy/gaia_tools) packages. [`emcee`](https://emcee.readthedocs.io/en/stable/) is used for performing MCMC.

Next up, you will have to download the custom low metallicity PARSEC isochrone grid used for this paper. You can find this (likely temporarily) at [this link](https://www.dropbox.com/sh/3gq0npsgffcp1rm/AABEinPPPdCfRcNEDRPciYpla?dl=0). If these files are no longer available, you will have to generate a new set of isochrones between Z = 0.0001 and 0.0030 with Delta_Z = 0.0001 from the [CMD interface](http://stev.oapd.inaf.it/cgi-bin/cmd). This grid should then be placed in a logical place, and the environment variable `ISODIST_DATA` set pointing to this location. `ISODIST_DATA` is the environment variable used for locating isochrones in the [`isodist`](https://github.com/jobovy/isodist) package, in case you already use that. Note however, that this package is *not* required explicitly for running the code.

### Running the code

Now load up the two jupyter notebooks. Running [selection-function.ipynb](https://github.com/jmackereth/halo-mass/blob/master/py/selection-function.ipynb) will calculate and compile the APOGEE selection function, and [density-fitting.pynb](https://github.com/jmackereth/halo-mass/blob/master/py/density-fitting.ipynb) prepares the APOGEE sample and performs the density modelling and mass estimation for MAPs. These notebooks contain everything necessary to reproduce the results of the paper.

In case you want to dig a little deeper, the file [utils.py](https://github.com/jmackereth/halo-mass/blob/master/py/utils.py) contains all the helper functions which sample the isochrones, prepare the data, perform the fitting etc. You can find the density models which were tested for use in the paper (and some extra utility functions) in the file [densprofiles.py](https://github.com/jmackereth/halo-mass/blob/master/py/densprofiles.py).












