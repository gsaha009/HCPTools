import numpy as np
import awkward as ak
from coffea.nanoevents.methods import vector
import matplotlib.pyplot as plt

from util import *


class PolarimetricA1:
    def __init__(self,
                 p4_tau: ak.Array, 
                 p4_os_pi: ak.Array, p4_ss1_pi: ak.Array, p4_ss2_pi: ak.Array,
                 taucharge: int,
                 decayChannel: str) -> None:
        """
          Calculate Polarimetric vector for tau to a1 decay
          args:
            p4_tau, p4_os_pi, p4_ss1_pi, p4_ss2_pi : awkward array of lorentz vectors
            taucharge
            decayChannel
          returns:
            None
        """
        print("Configure PolarimetricA1 : Start")

        self.P   = p4_tau
        self.p1  = p4_os_pi
        self.p2  = p4_ss1_pi
        self.p3  = p4_ss2_pi
        self.taucharge = taucharge
        self.decayChannel = decayChannel
        
        self.numResonances  = 7
        # define mass of tau lepton, charged and neutral pion;
        # values are taken from Prog. Theor. Exp. Phys. 2022 (2022) 083C01 (PDG)
        self.m_tau          = 1.7769   # [GeV]
        self.m_chargedPi    = 0.139570 # [GeV]
        self.m_neutralPi    = 0.134977 # [GeV]
        
        # define mass and width of a1(1260) meson;
        # values are taken from column "nominal fit" in Table VI of Phys.Rev.D 61 (2000) 012002
        self.m0_a1          = 1.331    # [GeV]
        self.Gamma0_a1      = 0.814    # [GeV]

        # define parameters specifying "running" of a1 width;
        # the values of Gamma_a1 as function of s have been taken from Fig. 9 (b) of Phys.Rev.D 61 (2000) 012002
        Gamma_a1_vs_s = [
            ( 0.00, 0.000 ), ( 0.20, 0.000 ), ( 0.40, 0.000 ), ( 0.50, 0.005 ), ( 0.60, 0.020 ),
            ( 0.65, 0.040 ), ( 0.70, 0.055 ), ( 0.75, 0.075 ), ( 0.80, 0.110 ), ( 0.85, 0.160 ),
            ( 0.90, 0.205 ), ( 0.95, 0.250 ), ( 1.00, 0.295 ), ( 1.05, 0.340 ), ( 1.10, 0.375 ),
            ( 1.15, 0.410 ), ( 1.20, 0.450 ), ( 1.25, 0.475 ), ( 1.30, 0.515 ), ( 1.35, 0.555 ),
            ( 1.40, 0.595 ), ( 1.45, 0.630 ), ( 1.50, 0.660 ), ( 1.55, 0.690 ), ( 1.60, 0.720 ),
            ( 1.65, 0.750 ), ( 1.70, 0.780 ), ( 1.75, 0.815 ), ( 1.80, 0.845 ), ( 1.85, 0.875 ),
            ( 1.90, 0.905 ), ( 1.93, 0.930 ), ( 1.95, 0.980 ), ( 2.00, 1.060 ), ( 2.05, 1.125 ),
            ( 2.10, 1.185 ), ( 2.15, 1.245 ), ( 2.20, 1.300 ), ( 2.25, 1.355 ), ( 2.30, 1.415 ),
            ( 2.35, 1.470 ), ( 2.37, 1.485 ), ( 2.40, 1.520 ), ( 2.45, 1.575 ), ( 2.50, 1.640 ),
            ( 2.55, 1.705 ), ( 2.60, 1.765 ), ( 2.65, 1.835 ), ( 2.70, 1.900 ), ( 2.75, 1.970 ),
            ( 2.80, 2.050 ), ( 2.85, 2.130 ), ( 2.90, 2.205 ), ( 2.95, 2.285 ), ( 3.00, 2.380 ),
            ( 3.05, 2.470 ), ( 3.10, 2.570 ), ( 3.15, 2.690 ) 
        ]
        self.Gamma_a1_vs_s = [list(tup) for tup in Gamma_a1_vs_s]

        # define masses and widths of intermediate rho(770), rho(1450), f2(1270), sigma, f0(1370) resonances;
        # values are taken from Table I of Phys.Rev.D 61 (2000) 012002
        self.m0_rho770      = 0.774    # [GeV]
        self.Gamma0_rho770  = 0.149    # [GeV]
        self.m0_rho1450     = 1.370    # [GeV]
        self.Gamma0_rho1450 = 0.386    # [GeV]
        self.m0_f2          = 1.275    # [GeV]
        self.Gamma0_f2      = 0.185    # [GeV]
        self.m0_sigma       = 0.860    # [GeV]
        self.Gamma0_sigma   = 0.880    # [GeV]
        self.m0_f0          = 1.186    # [GeV]
        self.Gamma0_f0      = 0.350    # [GeV]

        # define coefficients specifying the contribution of meson resonances to the hadronic current J
        # values are taken from Table III of Phys.Rev.D 61 (2000) 012002
        self.beta_moduli = [  1.00,  0.12,  0.37,  0.87,  0.71,  2.10,  0.77 ]
        self.beta_phases = [  0.00,  0.99, -0.15,  0.53,  0.56,  0.23, -0.54 ]
        assert len(self.beta_moduli) == self.numResonances and len(self.beta_phases) == self.numResonances

        self.beta = [TComplex(self.beta_moduli[i], self.beta_phases[i]*np.pi, True) for i in range(len(self.numResonances))]

        # metric tensor g^{mu,nu}
        self.g = np.array([[-1.0, -1.0, -1.0, -1.0],
                           [ 0.0, -1.0,  0.0, -1.0],
                           [ 0.0,  0.0, -1.0, -1.0],
                           [ 0.0,  0.0,  0.0,  1.0]])
        

    def Boost(self,p,frame):
        boostvec = frame.boostvec
        return p.boost(boostvec.negative())
        
        
    def operator(self) -> ak.Array:
        #P = setp4(name="PtEtaPhiMLorentzVector",
        #          ak.zeros_like(self.P.pt),
        #          ak.zeros_like(self.P.eta),
        #          ak.zeros_like(self.P.phi),
        #          self.P.mass)
        P  = self.Boost(self.P, self.P)
        p1 = self.Boost(self.p1, self.P)
        p2 = self.Boost(self.p2, self.P)
        p3 = self.Boost(self.p3, self.P)
        
        N = P - (p1 + p2 + p3)

        # p1, p2, p3, decayChannel
        J   = self.comp_J(p1, p2, p3)

        Pi  = self.comp_Pi(J, N)

        # charge
        Pi5 = self.comp_Pi5(J, N)

        # CV: Standard Model value, cf. text following Eq. (3.15) in Comput.Phys.Commun. 64 (1991) 275
        gammaVA = 1.0

        # CV: sign of terms proportional to gammaVA differs for tau+ and tau-,
        #     cf. text following Eq. (3.16) in Comput.Phys.Commun. 64 (1991) 275
        sign = 0.0
        if    self.charge == +1: sign = -1.
        elif  se;f.charge == -1: sign = +1.
        else assert(0)

        omega = P.dot(Pi - sign*gammaVA*Pi5)
    
        H = (1./(omega*P.mass))*(np.power(P.mass, 2)*(Pi5 - sign*gammaVA*Pi) - P.dot(Pi5 - sign*gammaVA*Pi)*P)

        retVal = H.pvec.unit

        return retVal


    def star(self, p) -> ak.Array:
        """
          Components must be awkward array.
          Numpy array is not allowed here.
          returns:
            awkward array of Lorentz vectors [each components are ak.array of complex conjugate objects]
        """
        out = ak.zip(
            {
                "x": p.x.Conjugate(),
                "y": p.y.Conjugate(),
                "z": p.z.Conjugate(),
                "t": P.t.Conjugate()
            },
            with_name="LorentzVector",
            behavior=vector.behavior
        )
        return out


    def get_component(self, p, mu):
        """
          @brief Return certain component of given four-vector
          @param p given four-vector
          @param mu component to be returned [1]
          
          [1] use 0 for px
              use 1 for py
              use 2 for pz
              use 3 for energy
          
          @return p[mu]
        """
        if   mu == 0:    return p.x
        elif mu == 1:    return p.y
        elif mu == 2:    return p.z
        elif mu == 3:    return p.t
        else: assert False


    def convert_to_cLorentzVector(self, p):
        """
          Convert four-vector of type LorentzVector to type cLorentzVector
          p four-vector [ak.Array]
          return: cLorentzVector(p)
        """
        vec = ak.zip(
            {
                "x": TComplex(p.x),
                "y": TComplex(p.y),
                "z": TComplex(p.z),
                "t": TComplex(p.t),
            },
            with_name="LorentzVector",
            behavior=vector.behavior,
        )
        return vec


    def convert_to_LorentzVector(self, p):
        """
          @brief Convert four-vector of type cLorentzVector to type LorentzVector. 
          Note: The imaginary part of the four-vector given as function argument is discarded in the conversion.
          @param p four-vector
          @return LorentzVector(p)    
        """
        LV = ak.zip(
            {
                "x": p.x.Re(),
                "y": p.y.Re(),
                "z": p.z.Re(),
                "t": p.t.Re(),
            },
            with_name = "LorentzVector",
            behavior = vector.behavior,
        )
        return LV

    
    def comp_Pi(self, J, N):
        """
          J: awkward array of complex LorentzVector
          N: awkward array of LorentzVector
        """
        cN = self.convert_to_cLorentzVector(N) # LV -> complex LV
        Jstar = self.star(J)

        JstarXN = Jstar.dot(self.g*cN)
        JXN     = J.dot(self.g*cN)
        JstarXJ = Jstar.dot(self.g*J)

        retVal = self.convert_to_LorentzVector(2.*(JstarXN*J + JXN*Jstar - JstarXJ*cN))  
        return retVal


    def sgn(self, x):
        """
          @brief Return sign of integer value given as function argument
          @param x function argument
          @return sign(x)
        """
        if x > 0: return +1
        if x < 0: return -1
        return 0


    def get_epsilon(self, mu, nu, rho, sigma):
        """
          @brief Compute Levi-Civita symbol epsilon^{mu,nu,rho,sigma},
                 cf. https://en.wikipedia.org/wiki/Levi-Civita_symbol
                 (Section "Levi-Civita tensors")
          @params mu, nu,rho, sigma indices of Levi-Civita symbol
          @return epsilon^{mu,nu,rho,sigma}
          CV: formula for computation of four-dimensional Levi-Civita symbol taken from
               https://en.wikipedia.org/wiki/Levi-Civita_symbol
             (Section "Definition" -> "Generalization to n dimensions")
        """
        a = 0*np.array[4]
        a[0] = mu
        a[1] = nu
        a[2] = rho
        a[3] = sigma
        epsilon = 1.
        for i in range(4):
            for j in range(4):
                epsilon *= self.sgn(a[j] - a[i])
        return epsilon

    
    def comp_Pi5(self, J, N):
        """
          J: awkward array of complex LorentzVector
          N: awkward array of LorentzVector
        """
        cN = self.convert_to_cLorentzVector(N)
        Jstar = self.star(J)

        J_     = [J.x, J.y, J.z, J.t]
        Jstar_ = [Jstar.x, Jstar.y, Jstar.z, Jstar.t]
        cN_    = [cN.x, cN.y, cN.z, cN.t]

        nev = ak.to_numpy(J.x).shape[0]
        vProd = np.zeros((4, nev, 1), dtype=TComplex)
        for mu in range(4):
            for nu in range(4):
                for rho in range(4):
                    for sigma in range(4):
                        epsilon = self.get_epsilon(mu, nu, rho, sigma)
                        vProd[mu] += epsilon*Jstar[nu]*J[rho]*cN[sigma]
        #vProd = self.g*vProd
        vProd = np.sum(self.g * vProd.T, axis=1)[:,:,np.newaxis]
        retval = ak.zip(
            {
                "x": 2.*vProd[0].Im(),
                "y": 2.*vProd[1].Im(),
                "z": 2.*vProd[2].Im(),
                "t": 2.*vProd[3].Im()
            },
            with_name="LorentzVector",
            behavior=vector.behavior,            
        )
        return retval


    def kdash(self, si, mj, mk):
        """
         @brief Compute "decay momentum",
                given by bottom line of Eq. (A6) in Phys.Rev.D 61 (2000) 012002
         @param si (mass of two-pion system that forms resonance)^2
         @param mj mass of first  pion that forms resonance [1]
         @param mk mass of second pion that forms resonance [1]
        
         [1] the ordering of the two pions does not matter
        
         @return k'_i
        """
        retVal = np.sqrt((si - np.power(mj + mk, 2))*(si - np.power(mj - mk, 2)))/(2.0*np.sqrt(si))
        return retVal;


    def Gamma(self, m0, Gamma0, si, mj, mk, L):
        """
        # @brief Compute "running width" of intermediate rho(770), rho(1450), f2(1270), sigma, and f0(1370)
        # resonances,
        #        given by bottom line of Eq. (A7) in Phys.Rev.D 61 (2000) 012002
        # @param m0 nominal mass of resonance
        # @param Gamma0 nominal width of resonance
        # @param si (mass of two-pion system that forms resonance)^2
        # @param mj mass of first  pion that forms resonance [1]
        # @param mk mass of second pion that forms resonance [1]
        # @param L angular momentum of resonance (s-wave = 0, p-wave=1, d-wave=2)
        #
        #  [1] the ordering of the two pions does not matter
        #
        # @return Gamma^{Y,L}(s_i)
        """
        kdashi = self.kdash(si, mj, mk)
        kdash0 = self.kdash(np.power(m0, 2), mj, mk)
        
        retVal = Gamma0*np.power(kdashi/kdash0, 2*L + 1)*m0/np.sqrt(si)
        return retVal


    def BreitWigner(self, m0, Gamma0, si, mj, mk, L) -> TComplex:
        """
          @brief Compute Breit-Wigner function of intermediate rho(770), rho(1450), f2(1270), sigma, and f0(1370) resonances,
                 given by top line of Eq. (A7) in Phys.Rev.D 61 (2000) 012002
          @param m0 nominal mass of resonance
          @param Gamma0 nominal width of resonance
          @param si (mass of two-pion system that forms resonance)^2
          @param mj mass of first  pion that forms resonance [1]
          @param mk mass of second pion that forms resonance [1]
          @param L angular momentum of resonance (s-wave = 0, p-wave=1, d-wave=2)
         
           [1] the ordering of the two pions does not matter
         
          @return B^{L}_{Y}(s_i)
        args:
          m0: scalar
          Gamma0: Scalar
          si: awkward array [mass]
          mj: scalar
          mk: scalar
          L: scalar
        return:
          Array of TComplex objects [awkward or numpy??? probably should be numpy :| ]
        """
        num = -np.power(m0, 2)
        denom = TComplex(si - np.power(m0, 2), m0*self.Gamma(m0, Gamma0, si, mj, mk, L))
        #denom = TComplex(ak.to_numpy(si - np.power(m0, 2)), ak.to_numpy(m0*self.Gamma(m0, Gamma0, si, mj, mk, L)))
        
        retVal = TComplex(num, 0.)/denom
        return retVal



    def comp_J(self, p1: ak.Array, p2: ak.Array, p3: ak.Array) -> TComplex:
        """
        Args:
          p1, p2, p3: awkward array of Lorentz Vectors

        Return:
          Awkward array of TComplex objects
        """
        q1 = p2 - p3
        q2 = p3 - p1
        q3 = p1 - p2

        h1 = p2 + p3
        Q1 = h1 - p1
        s1 = h1.mass2
        h2 = p1 + p3
        Q2 = h2 - p2
        s2 = h2.mass2
        h3 = p1 + p2
        Q3 = h3 - p3
        s3 = h3.mass2

        m1, m2, m3 = 0., 0., 0.
        if self.decayChannel == "k3ChargedPi":
            m1 = self.m_chargedPi
            m2 = self.m_chargedPi
            m3 = self.m_chargedPi
        elif self.decayChannel == "kChargedPi2NeutralPi":
            m1 = self.m_neutralPi
            m2 = self.m_neutralPi
            m3 = self.m_chargedPi
        else:
            raise RuntimeError(f"Error in <PolarimetricVectorTau2a1::comp_J>: Invalid parameter 'decayChannel' = {self.decayChannel}")

  
        a = p1 + p2 + p3
        s = a.mass2

        T = np.zeros((4,4), dtype=TComplex)
        for mu in range(4):
            for nu in range(4):
                # CV: all vectors without explicit mu indices are assumed to have the mu index as subscript,
                #     hence multiplication with the product of metric tensors g^{mu,mu}*g^{nu,nu} is required to transform
                #     the return value of the functions get_component(a, mu)*get_component(a, nu) into the expression a^{mu}*a^{nu}
                #     when computing T^{mu,nu} = g^{mu,nu} - a^{mu}a^{nu}/a^2,
                #     as described in text following Eq. (A2) in Phys.Rev.D 61 (2000) 012002
                #comp1np = ak.to_numpy(self.get_component(a, mu))
                #comp2np = ak.to_numpy(self.get_component(a, nu))
                #snp     = ak.to_numpy(s)
                #T[mu][nu] = TComplex(self.g[mu][nu] - self.g[mu][mu]*self.g[nu][nu]*self.get_component(a, mu)*self.get_component(a, nu)/s)
                T[mu][nu] = TComplex(ak.flatten(self.g[mu][nu] - self.g[mu][mu]*self.g[nu][nu]*self.get_component(a, mu)*self.get_component(a, nu)/s))
                #T[mu][nu] = TComplex((self.g[mu][nu] - self.g[mu][mu]*self.g[nu][nu]*comp1np*comp2np/snp).flatten())
                

        # compute amplitudes for individual resonances according to Eq. (A3) in Phys.Rev.D 61 (2000) 012002
        # Note: all the factors F_R in Eq. (A3) are equal to one, as the nominal fit assumes the a1 size parameter R to be zero
        #j = [np.zeros((4,snp.shape[0],1), dtype=TComplex) for _ in range(self.numResonances)]
        j = [np.zeros(4, dtype=TComplex) for _ in range(self.numResonances)]

        cq1 = self.convert_to_cLorentzVector(q1)
        cq2 = self.convert_to_cLorentzVector(q2)
        #j[0] = T@(self.BreitWigner(self.m0_rho770,  self.Gamma0_rho770,  s1, m2, m3, 1)*cq1 - self.BreitWigner(self.m0_rho770,  self.Gamma0_rho770,  s2, m1, m3, 1)*cq2)
        #j[1] = T@(self.BreitWigner(self.m0_rho1450, self.Gamma0_rho1450, s1, m2, m3, 1)*cq1 - self.BreitWigner(self.m0_rho1450, self.Gamma0_rho1450, s2, m1, m3, 1)*cq2)
        j0   = self.BreitWigner(self.m0_rho770,  self.Gamma0_rho770,  s1, m2, m3, 1)*cq1 - self.BreitWigner(self.m0_rho770,  self.Gamma0_rho770,  s2, m1, m3, 1)*cq2 
        j[0] = np.sum(T * j0.T, axis=1)
        j1   = self.BreitWigner(self.m0_rho1450, self.Gamma0_rho1450, s1, m2, m3, 1)*cq1 - self.BreitWigner(self.m0_rho1450, self.Gamma0_rho1450, s2, m1, m3, 1)*cq2
        j[1] = np.sum(T * j1.T, axis=1)

        aXq1 = a.dot(q1)
        cQ1 = self.convert_to_cLorentzVector(Q1)
        aXq2 = a.dot(q2)
        cQ2 = self.convert_to_cLorentzVector(Q2)
        #j[2] = T@(aXq1*self.BreitWigner(self.m0_rho770,  self.Gamma0_rho770,  s1, m2, m3, 1)*cQ1 - aXq2*self.BreitWigner(self.m0_rho770,  self.Gamma0_rho770,  s2, m1, m3, 1)*cQ2)
        #j[3] = T@(aXq1*self.BreitWigner(self.m0_rho1450, self.Gamma0_rho1450, s1, m2, m3, 1)*cQ1 - aXq2*self.BreitWigner(self.m0_rho1450, self.Gamma0_rho1450, s2, m1, m3, 1)*cQ2)
        j2   = aXq1*self.BreitWigner(self.m0_rho770,  self.Gamma0_rho770,  s1, m2, m3, 1)*cQ1 - aXq2*self.BreitWigner(self.m0_rho770,  self.Gamma0_rho770,  s2, m1, m3, 1)*cQ2
        j[2] = np.sum(T * j2.T, axis=1)
        j3   = aXq1*self.BreitWigner(self.m0_rho1450, self.Gamma0_rho1450, s1, m2, m3, 1)*cQ1 - aXq2*self.BreitWigner(self.m0_rho1450, self.Gamma0_rho1450, s2, m1, m3, 1)*cQ2
        j[3] = np.sum(T * j3.T, axis=1)

        aXq3 = a.dot(q3)
        cq3 = self.convert_to_cLorentzVector(q3)
        q3Xq3 = q3.mass2
        ca = self.convert_to_cLorentzVector(a)
        h3Xa = h3.dot(a)
        ch3 = self.convert_to_cLorentzVector(h3)
        #j[4] = T@(self.BreitWigner(self.m0_f2, self.Gamma0_f2, s3, m1, m2, 2)*(aXq3*cq3 - (q3Xq3/3.)*(ca - (h3Xa/s3)*ch3)))
        j4   = self.BreitWigner(self.m0_f2, self.Gamma0_f2, s3, m1, m2, 2)*(aXq3*cq3 - (q3Xq3/3.)*(ca - (h3Xa/s3)*ch3))
        j[4] = np.sum(T * j4.T, axis=1)
        
        cQ3 = self.convert_to_cLorentzVector(Q3)
        #j[5] = T@(self.BreitWigner(self.m0_sigma, self.Gamma0_sigma, s3, m1, m2, 0)*cQ3)
        #j[6] = T@(self.BreitWigner(self.m0_f0, self.Gamma0_f0, s3, m1, m2, 0)*cQ3)
        j5   = self.BreitWigner(self.m0_sigma, self.Gamma0_sigma, s3, m1, m2, 0)*cQ3
        j[5] = np.sum(T * j5.T, axis=1)[:,:,np.newaxis]
        j6   = self.BreitWigner(self.m0_f0, self.Gamma0_f0, s3, m1, m2, 0)*cQ3
        j[6] = np.sum(T * j6.T, axis=1)[:,:,np.newaxis]

        #retVal = np.zeros((4, snp.shape[0], 1) dtype=TComplex)
        retVal = np.zeros(4, dtype=TComplex)
        for idx in range(self.numResonances):
            retVal += self.beta[idx]*j[idx]

        retVal *= self.BreitWigner_a1(s)
        
        # CV: multiply with metric tensor in order to transform J^{mu} into J_{mu},
        #     as required to insert J^{mu} given by Eq. (A2) of Phys.Rev.D 61 (2000) 012002 into
        #     Eq. (3.15) of Comput.Phys.Commun. 64 (1991) 275
        retVal = self.g@retVal
        return retVal
    


    def m_a1(self, s: ak.Array) -> float:
        return self.m0_a1


    def Gamma_a1(self, s: ak.Array) -> ak.Array:
        mask1       = (s - self.Gamma_a1_vs_s[0][0]) <= 0
        mask1result = self.Gamma_a1_vs_s[0][1]*ak.ones_like(s)
        mask2       = (s - self.Gamma_a1_vs_s_[-1][0]) >= 0
        mask2result = self.Gamma_a1_vs_s_[-1][1]*ak.ones_like(s) 
        
        gamma_left  = ak.Array([a[0] for a in self.Gamma_a1_vs_s])
        gamma_right = ak.Array([a[1] for a in self.Gamma_a1_vs_s])

        gammaS, gammaL = ak.broadcast_arrays(ak.to_numpy(s), ak.to_numpy(gamma_left))
        _     , gammaR = ak.broadcast_arrays(ak.to_numpy(s), ak.to_numpy(gamma_right))

        gammaS = ak.from_regular(gammaS)
        gammaL = ak.from_regular(gammaL)
        gammaR = ak.from_regular(gammaR)

        diffLS = gammaS - gammaL
        mask_low  = diffLS >= 0.0
        mask_high = diffLS <= 0.0
        
        idx_low    = ak.local_index(gammaL)[mask_low]
        idx_high   = ak.local_index(gammaL)[mask_high]

        s_lo = gammaL[idx_low][:,-1][:,None]
        s_hi = gammaL[idx_high][:,1][:,None]
        Gamma_lo = gammaR[idx_low][:,-1][:,None]
        Gamma_hi = gammaR[idx_high][:,1][:,None]
        
        retVal = ak.where(mask1,
                          mask1result,
                          ak.where(mask2,
                                   mask2result,
                                   (Gamma_lo*(s_hi - s) + Gamma_hi*(s - s_lo)) / (s_hi - s_lo)
                                   )
                          )
        
        return retVal


    def BreitWigner_a1(self, s: ak.Array) -> TComplex:
        """
          return: awkward array of TComplex objects
        """
        m = self.m_a1(s)
        Gamma = self.Gamma_a1(s)

        num = -np.power(m, 2)
        denom = TComplex(s - np.power(m, 2), self.m0_a1*Gamma)

        retVal = TComplex(num, 0.)/denom
        return retVal
