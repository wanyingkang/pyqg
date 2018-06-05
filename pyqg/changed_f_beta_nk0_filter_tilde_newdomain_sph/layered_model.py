from __future__ import print_function
import numpy as np
from numpy import pi
from . import model

try:
    import mkl
    np.use_fastnumpy = True
except ImportError:
    pass

try:
    import pyfftw
    pyfftw.interfaces.cache.enable()
except ImportError:
    pass

class LayeredModel(model.Model):
    r"""Layered quasigeostrophic model.

    This model is meant to represent flows driven by baroclinic instabilty of a
    base-state shear. The potential vorticity anomalies qi are related to the
    streamfunction psii through

    .. math::

       {q_i} = \nabla^2\psi_i + \frac{f_0^2}{H_i} \left(\frac{\psi_{i-1}-
       \psi_i}{g'_{i-1}}- \frac{\psi_{i}-\psi_{i+1}}{g'_{i}}\right)\,,
       \qquad i = 2,\textsf{N}-1\,,

       {q_1} = \nabla^2\psi_1 + \frac{f_0^2}{H_1} \left(\frac{\psi_{2}-
       \psi_1}{g'_{1}}\right)\,,  \qquad i =1\,,

       {q_\textsf{N}} = \nabla^2\psi_\textsf{N} +
       \frac{f_0^2}{H_\textsf{N}} \left(\frac{\psi_{\textsf{N}-1}-
       \psi_\textsf{N}}{g'_{\textsf{N}}}\right) + \frac{f_0}{H_\textsf{N}}h_b\,,
       \qquad i =\textsf{N}\,,

    where the reduced gravity, or buoyancy jump, is

    .. math::

       g'_i \equiv g \frac{\pt_{i+1}-\pt_i}{\pt_i}\,.

    The evolution equations are

    .. math::

       \,{q_{i}}_t + \mathsf{J}\left(\psi_i\,, q_i\right) + \textsf{Q}_y {\psi_i}_x
       - \textsf{Q}_x {\psi_i}_y = \text{ssd} -
       r_{ek} \delta_{i\textsf{N}} \nabla^2 \psi_i\,, \qquad i = 1,\textsf{N}\,,


    where the mean potential vorticy gradients are

    .. math::

       \textsf{Q}_x = \textsf{S}\textsf{V}\,,

    and

    .. math::

       \textsf{Q}_y = \beta\,\textsf{I} - \textsf{S}\textsf{U}\,\,,

    where S is the stretching matrix, I is the identity matrix,
    and the background velocity is

    :math:`\vec{\textsf{V}}(z) = \left(\textsf{U},\textsf{V}\right)`.

    """

    def __init__(
        self,
        g = 9.81,
        beta=1.5e-11,               #? gradient of coriolis parameter
        nz = 4,                     # number of layers
        rd=15000.0,                 # deformation radius
        H = None,                   # layer thickness. If a scalar number, then copy the same H for all layers
        U=None,                     # zonal base state flow. If None, use U=0 for all layers
        V=None,                     # meridional base state flow. If None, use V=0 for all layers
        pt = None,                  # potential temperature
        c2 = None,
        delta = None,               # only used for nz=2, can leave blanck if use multi-layer model
        H0 = 7750,                  # standard atm height scale
        R = 287.,
        kappa = 2./7.,
        tau = 40,                   # time scale for restoring terms, units in day
        **kwargs
        ):
        """
        Parameters
        ----------

        nz : integer number
             Number of layers (> 1)
        beta : number
            Gradient of coriolis parameter. Units: meters :sup:`-1`
            seconds :sup:`-1`
        rd : number
            Deformation radius. Units: meters. Only necessary for
            the two-layer (nz=2) case.
        delta : number
            Layer thickness ratio (H1/H2). Only necessary for the
            two-layer (nz=2) case. Unitless.
        U : list of size nz
            Base state zonal velocity. Units: meters s :sup:`-1`
        V : array of size nz
            Base state meridional velocity. Units: meters s :sup:`-1`
        H : array of size nz
            Layer thickness. Units: meters
        pt: array of size nz.
            Layer Potential Temperature. Units: Kelvin

        """

        # physical
        if U is None:
            U=np.zeros([nz])
        if V is None:
            V=np.zeros([nz])
        if len(np.array(H))==1 and nz!=1:
            H=np.tile(np.array(H),nz)
        self.nz = nz
        self.g = g
        self.beta = beta
        self.rd = rd
        self.delta = delta
        self.R = R
        self.kappa = kappa
        self.tau = tau
        self.Ubg = np.array(U)
        self.Vbg = np.array(V)
        self.Hi = np.array(H)
        self.pti = np.array(pt)
        self.c2 = np.array(c2)
        self.H0 = H0

        super(LayeredModel, self).__init__(nz=nz, **kwargs)

        self.vertical_modes()
        print("nx:{}".format(self.nx))
        print("ny:{}".format(self.ny))
        print("nz:{}".format(self.nz))
    ### PRIVATE METHODS - not meant to be called by user ###


    def _initialize_stretching_matrix(self):
        """ Set up the stretching matrix """

        self.S = np.zeros((self.nz, self.nz))

        if (self.nz==2) and (self.rd) and (self.delta):

            self.del1 = self.delta/(self.delta+1.)
            self.del2 = (self.delta+1.)**-1
            self.Us = self.Ubg[0]-self.Ubg[1]

            self.F1 = self.rd**-2 / (1.+self.delta)
            self.F2 = self.delta*self.F1
            self.S[0,0], self.S[0,1] = -self.F1,  self.F1
            self.S[1,0], self.S[1,1] =  self.F2, -self.F2

        else:

            for i in range(self.nz):
                # Adding other statification terms by Wanying Kang @ Feb 14 2017
                # All following S element, the second half of expression terms 
                # are added to represent stratification 1/H term.
                # Would still have terms represent boundary conditions at top and bottom.
                # q1 = q1 + (self.f*self.g/self.gpi[i]*(1-self.Hi[i]/self.H0/2))*(self.T1(x,y)/self.T0) ,i=0
                # qN = qN + (self.f*self.g/self.gpi[i]*(-1-self.Hi[i]/self.H0/2))*(self.TN(x,y)/self.T0) ,i=nz-1
                # delete the Hi terms at i=0 and i=nz-1 by assuming \psi_zz=0 at top and bottom
                # This assumption means vertical T gradient is zero. T = -f/R*\psi_{z^*}

                if i == 0:
                    # 1. assume \Psi_zz|_{top/bot}=0
                    #self.S[i,i]   = (-self.f2/self.H0/self.gpi[i])
                    #self.S[i,i+1] =  (self.f2/self.H0/self.gpi[i])
                    
                    # 2. assume \Psi_z|_{out_of_range}=0, need to substract constant term to represent the constant temperature when invert \Psi. 
                    # self.S[i,i]   = (-self.f2/self.Hi[i]/self.gpi[i]-
                    #                self.f2/self.H0/self.gpi[i]/2.)
                    # self.S[i,i+1] =  (self.f2/self.Hi[i]/self.gpi[i]+
                    #                 self.f2/self.H0/self.gpi[i]/2.)
                    
                    # 3. transform \Psi -> \tilde \Psi, use BC \Psi_zz|_{top/bot}=0
                    self.S[i,i] = -self.f2*self.c2[i]

                    # 4. transform \Psi -> \tilde \Psi, use BC \Psi_z|_{out_of_range}=0, need to substract constant term when invert \Psi.
                    #self.S[i,i]   = -self.f2/self.Hi[i]/self.gpi[i]-self.f2*self.c2[i]
                    #self.S[i,i+1] = self.f2/self.Hi[i]/self.gpi[i]

                elif i == self.nz-1:
                    # 1. 
                    #self.S[i,i]   = (self.f2/self.H0/self.gpi[i-1])
                    #self.S[i,i-1] =  (-self.f2/self.H0/self.gpi[i-1])
                    # 2.
                    # self.S[i,i]   = (-self.f2/self.Hi[i]/self.gpi[i-1]+
                    #                self.f2/self.H0/self.gpi[i-1]/2.)
                    # self.S[i,i-1] =  (self.f2/self.Hi[i]/self.gpi[i-1]-
                    #                self.f2/self.H0/self.gpi[i-1]/2.)
                    # 3.
                    self.S[i,i] = -self.f2*self.c2[i]
                    # 4.
                    #self.S[i,i]   = -self.f2/self.Hi[i]/self.gpi[i-1]-self.f2*self.c2[i]
                    #self.S[i,i-1] = self.f2/self.Hi[i]/self.gpi[i-1]

                else:
                    # 1. or 2.
                    #self.S[i,i-1] = (self.f2/self.Hi[i]/self.gpi[i-1]-
                    #                self.f2/self.H0/self.gpi[i-1]/2.)
                    #self.S[i,i]   = (-(self.f2/self.Hi[i]/self.gpi[i] +
                    #                    self.f2/self.Hi[i]/self.gpi[i-1])-
                    #                    (self.f2/self.H0/self.gpi[i]/2.-
                    #                    self.f2/self.H0/self.gpi[i-1]/2.))
                    #self.S[i,i+1] = (self.f2/self.Hi[i]/self.gpi[i]+
                    #                self.f2/self.H0/self.gpi[i]/2.)
                    
                    # 3. or 4.
                    self.S[i,i-1] = self.f2/self.Hi[i]/self.gpi[i-1]
                    self.S[i,i]   = (-(self.f2/self.Hi[i]/self.gpi[i] +
                                        self.f2/self.Hi[i]/self.gpi[i-1])
                                        -self.f2*self.c2[i])
                    self.S[i,i+1] = self.f2/self.Hi[i]/self.gpi[i]

    def _initialize_background(self):
        """Set up background state (zonal flow and PV gradients)."""

        self.H = self.Hi.sum()

        if not (self.nz==2):
            #self.gpi = -self.g*(self.pti[1:]-self.pti[:-1])/self.pti[:-1]
            self.gpi = -(self.pti[1:]-self.pti[:-1])/self.H0*self.R*np.exp(-self.kappa/self.H0*np.asarray(self.z[:-1]))
            self.f2gpi = (self.f2/self.gpi)[:,np.newaxis,np.newaxis]

            assert self.gpi.size == self.nz-1, "Invalid size of gpi"

            assert np.all(self.gpi>0.), "Buoyancy jump has negative sign!"

            assert self.Hi.size == self.nz, self.logger.error('size of Hi does not' +
                     'match number of vertical levels nz')

            assert self.pti.size == self.nz, self.logger.error('size of pti does not' +
                     'match number of vertical levels nz')

            assert self.Ubg.size == self.nz, self.logger.error('size of Ubg does not' +
                     'match number of vertical levels nz')

            assert self.Vbg.size == self.nz, self.logger.error('size of Vbg does not' +
                     'match number of vertical levels nz')

        else:
            self.f2gpi = np.array(self.rd**-2 *
                (self.Hi[0]*self.Hi[1])/self.H)[np.newaxis,np.newaxis]

        ## Initialize stretching matrix
        self._initialize_stretching_matrix()

        ## the meridional PV gradients in each layer
        ## Original version
        #self.Qy = self.beta - np.dot(self.S,self.Ubg)
        #self.Qx = np.dot(self.S,self.Vbg)
        ## complex versions, multiplied by k, speeds up computations to precompute
        #self.ikQy = self.Qy[:,np.newaxis,np.newaxis]*1j*self.k
        #self.ilQx = self.Qx[:,np.newaxis,np.newaxis]*1j*self.l
        
        
        ## Set the meridional PV gradients in each layer
        #  Wanying Kang add lat dependent on beta.
        #  Qy is nz*nl*nl matrix, convolution matrix takes the nl*nl dimension
        #  The kernel calculate _ikQy from Qy, instead of using ikQy here. 
        #  _ikQy is originally nz*nk matrix, different from original ikQy which is a nz*nl*nk matrix. After my modificatino, they are the same.
        # This ikQy is used in stability analysis in model.py
        
        #b_lat = np.asarray(self.coslat)**2.*(np.asarray(self.coslat)**2.-2.*np.asarray(self.sinlat)**2.)
        b_lat = np.asarray(self.coslat)**2.
        b_lat[int(self.nl/2):,:] = -b_lat[int(self.nl/2):,:]
        b_lat1 = np.squeeze(b_lat[:,0])
        b_lat = np.tile(b_lat[np.newaxis,:,:], (self.nz,1,1))
        bh_lat = self.fft(b_lat)/(self.nl**2)/(self.nl)
        bh_lat = np.squeeze(bh_lat[0,:,0])      # uniform in x direction, so pick k=0
        
        #Cbh1 = (self.convmtx( bh_lat[:int(self.nl/2)] , self.nl ))[:int(self.nl/2),:]
        #Cbh2 = (self.convmtx( bh_lat[int(self.nl/2):] , self.nl ))[-int(self.nl/2):,:] 
        #Cbh = np.concatenate( [Cbh1, Cbh2] , 0 )
        
        order = np.concatenate([range(int(self.nl/2),self.nl),range(0,int(self.nl/2))])
        Cbh_shift = self.convmtx( bh_lat[order] , self.nl ) 
        Cbh_shift = Cbh_shift[int(self.nl/2):-int(self.nl/2)+1,:]
        Cbh = Cbh_shift[order,:]
        Cbh = Cbh[:,order]

        # Test Wanying Kang's convolution
        #b_test1 = np.arange(self.nl)/2.
        #b_test = np.tile(b_test1[np.newaxis,:,np.newaxis], (self.nz,1,self.nx))
        #bh_test = self.fft(b_test)
        #bh_test1 = np.squeeze(bh_test[0,:,0])
        #b_result = b_test1*b_lat1
        #bh_result = np.dot(Cbh,bh_test1)
        #bh_result = self.ifft(np.tile(bh_result[np.newaxis,:,np.newaxis], (self.nz,1,self.nk)))
        #bh_result = np.squeeze(bh_result[0,:,0])
        #print(b_result)
        #print(bh_result)

        # real space version of Qy Qx:
        #self.Qy = np.tile(self.beta*b_lat1[np.newaxis,:],[self.nz,1]) - np.tile((np.dot(self.S,self.Ubg))[:,np.newaxis],[1,self.nl])
        #self.Qx = np.tile(np.dot(self.S,self.Vbg)[:,np.newaxis],[1,self.nl])

        # spectra space version of Qy Qx:
        self.Qy = np.tile(self.beta*Cbh[np.newaxis,:,:],[self.nz,1,1]) - np.tile((np.dot(self.S,self.Ubg))[:,np.newaxis,np.newaxis],[1,self.nl,self.nl])
        self.Qx = np.dot(self.S,self.Vbg)       
        
        # complex versions, multiplied by k, speeds up computations to precompute
        # Wanying Kang: add lat dependent on beta. ikQy is nz*nl*nl*nk matrix
        self.ikQy = self.Qy[:,:,:,np.newaxis]*1j*self.kk[np.newaxis,np.newaxis,np.newaxis,:]
        self.ilQx = self.Qx[:,np.newaxis,np.newaxis]*1j*self.l  #Original version
    

        ## lat-dependent restoring terms
        g_lat1 = (1.+50.*(1-np.tanh(7*self.sinlat1)))
        #      multiply sinlat1*coslat1**2 like qM, sinlat1 will weaken the drag, so maybe not..
        g_lat = np.tile(g_lat1[np.newaxis,:,np.newaxis], (self.nz,1,self.nx))
        gh_lat = self.fft(g_lat)/(self.nl**2)/(self.nl)
        gh_lat = np.squeeze(gh_lat[0,:,0]) 
        Cgh_shift = self.convmtx( gh_lat[order] , self.nl ) 
        Cgh_shift = Cgh_shift[int(self.nl/2):-int(self.nl/2)+1,:]
        Cgh = Cgh_shift[order,:]
        Cgh = Cgh[:,order]
        
        self.gamma = np.tile(1./self.tau/86400.*Cgh[np.newaxis,:,:],[self.nz,1,1])
        
#    def _initialize_inversion_matrix(self):
#        # Original Version
#        a = np.ma.zeros((self.nz, self.nz, self.nl, self.nk), np.dtype('float64'))
#
#        if (self.nz==2):
#            det_inv =  np.ma.masked_equal(
#                    ( (self.S[0,0]-self.wv2)*(self.S[1,1]-self.wv2) -\
#                            self.S[0,1]*self.S[1,0] ), 0.)**-1
#            a[0,0] = (self.S[1,1]-self.wv2)*det_inv
#            a[0,1] = -self.S[0,1]*det_inv
#            a[1,0] = -self.S[1,0]*det_inv
#            a[1,1] = (self.S[0,0]-self.wv2)*det_inv
#        else:
#            I = np.eye(self.nz)[:,:,np.newaxis,np.newaxis]
#            M = self.S[:,:,np.newaxis,np.newaxis]-I*self.wv2
#            M[:,:,0,0] = np.nan  # avoids singular matrix in inv()
#            a = np.linalg.inv(M.T).T
#        print(a[a!=0])
#        self.a = np.ma.masked_invalid(a).filled(0.)

    def _initialize_inversion_matrix(self):

        # Wanying Kang: Do convolution if f has lat-stucture as 
        # f=f0*cos(lat)*sin(lat), f2=f0^2*cos^2(lat)*sin^2(lat)
        a = np.ma.zeros((self.nz, self.nz, self.nl, self.nl, self.nk0), np.dtype(np.complex128))

        if (self.nz==2):
            Ij = np.eye(self.nl)[np.newaxis,np.newaxis,:,:,np.newaxis]
            det_inv =  np.ma.masked_equal(
                    ( (self.S[0,0]-self.wv2)*(self.S[1,1]-self.wv2) -\
                            self.S[0,1]*self.S[1,0] ), 0.)**-1
            for j in range(self.nl): 
                a[0,0,j,j] = (self.S[1,1]-self.wv2)*det_inv
                a[0,1,j,j] = -self.S[0,1]*det_inv
                a[1,0,j,j] = -self.S[1,0]*det_inv
                a[1,1,j,j] = (self.S[0,0]-self.wv2)*det_inv
        else:
            Izl = np.multiply.outer(np.eye(self.nz),np.eye(self.nl))
            Iz = np.eye(self.nz)
            
            # Wanying Kang: Do convolution if f has lat-stucture as 
            # f=f0*cos(lat)*sin(lat), f2=f0^2*cos^2(lat)*sin^2(lat)
            f_lat = np.asarray(self.coslat)**2.*np.asarray(self.sinlat)**2.
            f_lat = np.tile(f_lat[np.newaxis,:,:], (self.nz,1,1))

            fh_lat = self.fft(f_lat)/(self.nl**2)/(self.nl)
            fh_lat = np.squeeze(fh_lat[0,:,0])      # uniform in x direction, so pick k=0
            #Cfh1 = (self.convmtx( fh_lat[:int(self.nl/2)] , self.nl ))[:int(self.nl/2),:]
            #Cfh2 = (self.convmtx( fh_lat[int(self.nl/2):] , self.nl ))[-int(self.nl/2):,:] 
            #Cfh = np.concatenate( [Cfh1, Cfh2] , 0 )
            #Cfh = np.eye(self.nl)                  # compare with non-lat dependent case
            
            order = np.concatenate([range(int(self.nl/2),self.nl),range(0,int(self.nl/2))])
            Cfh_shift = self.convmtx( fh_lat[order] , self.nl ) 
            Cfh_shift = Cfh_shift[int(self.nl/2):-int(self.nl/2)+1,:]
            Cfh = Cfh_shift[order,:]
            Cfh = Cfh[:,order]

            # Wanying Kang: Add spherical term -cos^2(lat)/sin(lat)*Psi_y
            sp_lat=-np.asarray(self.coslat1)**2./np.asarray(self.sinlat1)
            sp_lat = np.tile(sph_lat[np.newaxis,:,np.newaxis],(self.nz,1,self.nx))
            sph_lat = self.fft(sp_lat)/(self.nl**2)/(self.nl)
            sph_lat = np.squeeze(sph_lat[0,:,0])

            Csph_shift = self.convmtx( sph_lat[order] , self.nl ) 
            Csph_shift = Csph_shift[int(self.nl/2):-int(self.nl/2)+1,:]
            Csph = Csph_shift[order,:]
            Csph = Csph[:,order]
            Csph = Csph*1j*self.ll

            
            # Wanying Kang: Make up poisson operator, M
            M =( (np.multiply.outer(self.S,Cfh))[:,:,:,:,np.newaxis]
                 -Izl[:,:,:,:,np.newaxis]*self.wv2[np.newaxis,np.newaxis,np.newaxis,:,:]
                 + Izl[:,:,:,:,np.newaxis]*Csph[np.newaxis,np.newaxis,:,:,np.newaxis])

            # Wanying Kang: Add BC by modifying the poisson operator M,
            # give up the equation for high k wavenumber, need totally nz*nk0+nz*(nk0-1) slots. 
            #    1. NP: p|_{NP}=0
            #        For all k, wave#k component has no Amp at NP.
            #M[:,:,int(self.nl/2),:,0:self.nk0]=(Iz[:,:,np.newaxis,np.newaxis])*((np.exp(2*pi*1j*self.ll*(self.ny-1)/self.nl)/self.nl)[np.newaxis,np.newaxis,:,np.newaxis])
            #M[:,:,int(self.nl/2),int(self.nl/2),0:self.nk0]=0.
            #M[:,:,int(self.nl/2),int(self.nl*3/4),0:self.nk0]=0.
            #    2. SP: p_x|_{SP}=0. 1j*k*ph|_{SP}=0 where ph is fourier transformed p in x dir.
            #        For k=0, the equation is automatically satisfied; For k/=0, this means wave#k component has no Amp at SP.
            #M[:,:,int(self.nl*3/4),:,1:self.nk0]=(Iz[:,:,np.newaxis,np.newaxis])*(1/self.nl)*self.kk[1:self.nk0]
            #M[:,:,int(self.nl/2),int(self.nl/2),1:self.nk0]=0.
            #M[:,:,int(self.nl/2),int(self.nl*3/4),1:self.nk0]=0.
          
            # Wanying Kang: calculate matrix inversion
            Mt = np.ascontiguousarray(np.transpose(M,[4,0,2,1,3]))
            Mt.shape=(self.nk,self.nz*self.nl,self.nz*self.nl)
            #Mt[0,:,:]=np.nan  # avoids singular matrix in inv(), however filterred out k=0 components.
            for ik in range(self.nk0):
                at = np.linalg.inv(Mt[ik,:,:])
                at.shape = (self.nz,self.nl,self.nz,self.nl)
                a[:,:,:,:,ik] = np.transpose(at,[0,2,1,3])
            a[:,:,0,0,0]=0.
        #self.a = np.ma.masked_invalid(a).filled(0.)
        self.a = a
        
        # Wanying Kang add b matrix to invert k=0 component,
        # now this is not necessary since I changed the way I calculate a above.
        #Mb = np.multiply.outer(self.S,Cfh)-Izl*(self.ll**2)
        #Mb[:,:,int(self.nl/2)-1,:]=(Iz[:,:,np.newaxis])*((np.exp(2*pi*1j*self.ll*(self.ny-1)/self.nl)/self.nl)[np.newaxis,np.newaxis,:])
        #Mb = M[:,:,:,:,0]
        #Mb[:,:,0,0]=np.nan
        #Mbt = np.ascontiguousarray(np.transpose(Mb,[0,2,1,3]))
        #Mbt.shape=(self.nl*self.nz,self.nl*self.nz)
        #bt = np.linalg.inv(Mbt)
        #bt.shape = (self.nz,self.nl,self.nz,self.nl)
        #b = np.transpose(bt,[0,2,1,3])
        #b [:,:,0,0]=0.+0j
        #self.a[:,:,:,:,0]=b
        
    def _initialize_forcing(self):
        pass
        #"""Set up frictional filter."""
        # this defines the spectral filter (following Arbic and Flierl, 2003)
        # cphi=0.65*pi
        # wvx=np.sqrt((self.k*self.dx)**2.+(self.l*self.dy)**2.)
        # self.filtr = np.exp(-self.filterfac*(wvx-cphi)**4.)
        # self.filtr[wvx<=cphi] = 1.

    ### All the diagnostic stuff follows. ###
    def _calc_cfl(self):
        return np.abs(
            np.hstack([self.u + self.Ubg[:,np.newaxis,np.newaxis], self.v])
        ).max()*self.dt/self.dx

    # calculate KE: this has units of m^2 s^{-2}
    #   (should also multiply by H1 and H2...)
    def _calc_ke(self):
        ke = 0.
        for j in range(self.nz):
            ke += .5*self.Hi[j]*self.spec_var(self.wv*self.ph[j])
        return ke.sum() / self.H

    # calculate eddy turn over time
    # (perhaps should change to fraction of year...)
    def _calc_eddy_time(self):
        """ estimate the eddy turn-over time in days """
        ens = 0.
        for j in range(self.nz):
            ens = .5*self.Hi[j] * self.spec_var(self.wv2*self.ph[j])

        return 2.*pi*np.sqrt( self.H / ens.sum() ) / 86400

    def _calc_derived_fields(self):

        self.p = self.ifft(self.ph)
        self.xi =self.ifft(-self.wv2*self.ph)
        self.Jpxi = self._advect(self.xi, self.u, self.v)
        self.Jq = self._advect(self.q, self.u, self.v)

        self.Sph = np.einsum("ij,jkl->ikl",self.S,self.ph)
        self.Sp = self.ifft(self.Sph)
        self.JSp = self._advect(self.Sp,self.u,self.v)

        self.phn = self.modal_projection(self.ph)


    def _initialize_model_diagnostics(self):
        """ Extra diagnostics for layered model """

        self.add_diagnostic('entspec',
                description='barotropic enstrophy spectrum',
                function= (lambda self:
                    np.abs((self.Hi[:,np.newaxis,np.newaxis]*self.qh).sum(axis=0))**2/self.H) )

        self.add_diagnostic('KEspec_modal',
                description='modal KE spectra',
                function= (lambda self:
                    self.wv2*(np.abs(self.phn)**2)/self.M**2 ))

        self.add_diagnostic('PEspec_modal',
                description='modal PE spectra',
                function= (lambda self:
                    self.kdi2[1:,np.newaxis,np.newaxis]*(np.abs(self.phn[1:,:,:])**2)/self.M**2 ))

        self.add_diagnostic('APEspec',
                description='available potential energy spectrum',
                function= (lambda self:
                           (self.f2gpi*
                            np.abs(self.ph[:-1]-self.ph[1:])**2).sum(axis=0)/self.H))

        self.add_diagnostic('KEflux',
                    description='spectral divergence of flux of kinetic energy',
                    function =(lambda self: (self.Hi[:,np.newaxis,np.newaxis]*
                               (self.ph.conj()*self.Jpxi).real).sum(axis=0)/self.H))

        self.add_diagnostic('APEflux',
                    description='spectral divergence of flux of available potential energy',
                    function =(lambda self: (self.Hi[:,np.newaxis,np.newaxis]*
                               (self.ph.conj()*self.JSp).real).sum(axis=0)/self.H))

        self.add_diagnostic('APEgenspec',
                    description='the spectrum of the rate of generation of available potential energy',
                    function =(lambda self: (self.Hi[:,np.newaxis,np.newaxis]*
                                (self.Ubg[:,np.newaxis,np.newaxis]*self.k +
                                 self.Vbg[:,np.newaxis,np.newaxis]*self.l)*
                                (1j*self.ph.conj()*self.Sph).real).sum(axis=0)/self.H))

        self.add_diagnostic('ENSflux',
                 description='barotropic enstrophy flux',
                 function = (lambda self: (-self.Hi[:,np.newaxis,np.newaxis]*
                              (self.qh.conj()*self.Jq).real).sum(axis=0)/self.H))

#        # Wanying Kang: this function cannot be used since I change the dimension of ikQy
#        self.add_diagnostic('ENSgenspec',
#                    description='the spectrum of the rate of generation of barotropic enstrophy',
#                    function = (lambda self:
#                            -(self.Hi[:,np.newaxis,np.newaxis]*((self.ikQy -
#                            self.ilQx)*(self.Sph.conj()*self.ph)).real).sum(axis=0)/self.H))
