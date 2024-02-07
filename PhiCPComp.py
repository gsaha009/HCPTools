import os
import numpy as np
import awkward as ak
from PhiCPBase import PhiCPBase
#from PolarimetricA1 import PolarimetricA1
#from PolarimetricA1_CV_KS import PolarimetricVectorA1


class PhiCPComp(PhiCPBase):
    def __init__(self,
                 cat: str = "",
                 taum: ak.Array = None, 
                 taup: ak.Array = None, 
                 taum_decay: ak.Array = None, 
                 taup_decay: ak.Array = None):
        """
            cat: category to compute on
                 possible values would be "pipi", "pirho", "rhorho", "a1rho", "a1a1" [for now]
            Resources:
                 https://coffeateam.github.io/coffea/api/coffea.nanoevents.methods.vector.LorentzVector.html#coffea.nanoevents.methods.vector.LorentzVector.boost
        """
        super(PhiCPComp, self).__init__(taum=taum, taup=taup, taum_decay=taum_decay, taup_decay=taup_decay)
        if cat not in ["pipi", "pirho", "rhorho", "a1pi", "a1rho", "a1a1"]:
            raise RuntimeError ("category must be defined !")
        self.cat = cat


    def gethvecs_pipi(self):
        boostvec, p4_taum_pi, p4_taup_pi, p4_taum_hrest, p4_taup_hrest = self.get_evtinfo_pipi()

        # Polarimetric vectors will be along the direction of the respective pions
        h1raw = self.gethvec_pi(boostvec, p4_taum_pi)
        h2raw = self.gethvec_pi(boostvec, p4_taup_pi)
        #h1raw.absolute(), h2raw.absolute()

        return h1raw, h2raw, p4_taum_hrest, p4_taup_hrest

    
    def gethvecs_pirho_leg1(self):
        h_boostvec, p4_taum, p4_taum_pi, p4_taup_pi, p4_taum_pi0, p4_taum_nu, p4_taum_hrest, p4_taup_hrest = self.get_evtinfo_pirho_leg1()
        
        # Polarimetric vectors will be along the direction of the respective pions
        h1raw = self.gethvec_rho(h_boostvec, p4_taum, p4_taum_pi, p4_taum_pi0, p4_taum_nu)
        h2raw = self.gethvec_pi(h_boostvec, p4_taup_pi)
        #h1raw.absolute(), h2raw.absolute()
        
        return h1raw, h2raw, p4_taum_hrest, p4_taup_hrest

    
    def gethvecs_pirho_leg2(self):
        h_boostvec, p4_taup, p4_taum_pi, p4_taup_pi, p4_taup_pi0, p4_taup_nu, p4_taum_hrest, p4_taup_hrest = self.get_evtinfo_pirho_leg2()
        # Polarimetric vectors will be along the direction of the respective pions
        h1raw = self.gethvec_pi(h_boostvec, p4_taum_pi)
        h2raw = self.gethvec_rho(h_boostvec, p4_taup, p4_taup_pi, p4_taup_pi0, p4_taup_nu)
        #h1raw.absolute(), h2raw.absolute()
        
        return h1raw, h2raw, p4_taum_hrest, p4_taup_hrest
        
        
    def gethvecs_rhorho(self):
        h_boostvec, p4_taum, p4_taup, p4_taum_pi, p4_taup_pi, p4_taum_pi0, p4_taup_pi0, p4_taum_nu, p4_taup_nu, p4_taum_hrest, p4_taup_hrest = self.get_evtinfo_rhorho()

        # Polarimetric vectors will be along the direction of the respective pions
        h1raw = self.gethvec_rho(h_boostvec, p4_taum, p4_taum_pi, p4_taum_pi0, p4_taum_nu)
        h2raw = self.gethvec_rho(h_boostvec, p4_taup, p4_taup_pi, p4_taup_pi0, p4_taup_nu)
        #h1raw.absolute(), h2raw.absolute()
        
        return h1raw, h2raw, p4_taum_hrest, p4_taup_hrest
            
    
    def gethvecs_a1rho(self):
        pass
    
    def gethvecs_a1pi_leg1(self):

        h_boostvec, p4_taum, p4_taum_os_pi, p4_taum_ss1_pi, p4_taum_ss2_pi, p4_taup_pi, p4_taum_hrest, p4_taup_hrest = self.get_evtinfo_a1pi_leg1()
        
        # Polarimetric vectors will be along the direction of the respective pions
        h1raw = self.gethvec_a1(h_boostvec,
                                p4_taum,
                                p4_taum_os_pi,
                                p4_taum_ss1_pi,
                                p4_taum_ss2_pi,
                                -1)
        h2raw = self.gethvec_pi(h_boostvec, p4_taup_pi)
        
        #h1raw.absolute(), h2raw.absolute()
        
        return h1raw, h2raw, p4_taum_hrest, p4_taup_hrest         

    def gethvecs_a1pi_leg2(self):
        h_boostvec, p4_taup, p4_taup_os_pi, p4_taup_ss1_pi, p4_taup_ss2_pi, p4_taum_pi, p4_taum_hrest, p4_taup_hrest = self.get_evtinfo_a1pi_leg2()
        
        # Polarimetric vectors will be along the direction of the respective pions
        h1raw = self.gethvec_a1(h_boostvec,
                                p4_taup,
                                p4_taup_os_pi,
                                p4_taup_ss1_pi,
                                p4_taup_ss2_pi,
                                1)
        h2raw = self.gethvec_pi(h_boostvec, p4_taum_pi)
        
        #h1raw.absolute(), h2raw.absolute()
        
        return h1raw, h2raw, p4_taum_hrest, p4_taup_hrest


    def gethvecs_a1a1(self):
        h_boostvec, p4_taum, p4_taup, p4_taum_os_pi, p4_taup_os_pi, p4_taum_ss1_pi, p4_taup_ss1_pi, p4_taum_ss2_pi, p4_taup_ss2_pi, p4_taum_hrest, p4_taup_hrest = self.get_evtinfo_a1a1()
        
        # Polarimetric vectors will be along the direction of the respective pions
        h1raw = self.gethvec_a1(p4_taum+p4_taup,
                                h_boostvec,
                                p4_taum,
                                p4_taum_os_pi,
                                p4_taum_ss1_pi,
                                p4_taum_ss2_pi,
                                -1)
        h2raw = self.gethvec_a1(p4_taum+p4_taup,
                                h_boostvec,
                                p4_taup,
                                p4_taup_os_pi,
                                p4_taup_ss1_pi,
                                p4_taup_ss2_pi,
                                +1)
        #h1raw.absolute(), h2raw.absolute()
        
        return h1raw, h2raw, p4_taum_hrest, p4_taup_hrest
        
    
    def gethvecs(self) -> ak.Array:
        print(" --- gethvecs --- ")
        h1 = None
        h2 = None
        p4_taum_hrest = None
        p4_taup_hrest = None
        
        if self.cat == "pipi":
            h1, h2, p4_taum_hrest, p4_taup_hrest = self.gethvecs_pipi()
        elif self.cat == "pirho":
            h1_1, h2_1, p4_taum_hrest_1, p4_taup_hrest_1 = self.gethvecs_pirho_leg1()
            h1_2, h2_2, p4_taum_hrest_2, p4_taup_hrest_2 = self.gethvecs_pirho_leg2()
            h1 = ak.concatenate([h1_1, h1_2], axis=0)
            h2 = ak.concatenate([h2_1, h2_2], axis=0)
            print(p4_taum_hrest_1.pt.type, p4_taum_hrest_2.pt.type)
            p4_taum_hrest = ak.concatenate([p4_taum_hrest_1, p4_taum_hrest_2], axis=0)
            p4_taup_hrest = ak.concatenate([p4_taup_hrest_1, p4_taup_hrest_2], axis=0)
        elif self.cat == "rhorho":
            h1, h2, p4_taum_hrest, p4_taup_hrest = self.gethvecs_rhorho()            
        elif self.cat == "a1a1":
            # create pol_a1 object
            # call configure
            # call gethvec
            h1, h2, p4_taum_hrest, p4_taup_hrest = self.gethvecs_a1a1()
        elif self.cat == "a1pi":
            h1_1, h2_1, p4_taum_hrest_1, p4_taup_hrest_1 = self.gethvecs_a1pi_leg1()
            h1_2, h2_2, p4_taum_hrest_2, p4_taup_hrest_2 = self.gethvecs_a1pi_leg2()
            h1 = ak.concatenate([h1_1, h1_2], axis=0)
            h2 = ak.concatenate([h2_1, h2_2], axis=0)
            print(p4_taum_hrest_1.pt.type, p4_taum_hrest_2.pt.type)
            p4_taum_hrest = ak.concatenate([p4_taum_hrest_1, p4_taum_hrest_2], axis=0)
            p4_taup_hrest = ak.concatenate([p4_taup_hrest_1, p4_taup_hrest_2], axis=0)

        else:
            raise RuntimeError ("Give right category name")
            
        return h1, h2, p4_taum_hrest, p4_taup_hrest


    
    def comp_phiCP(self):
        print(" --- comp_phiCP --- ")
        h1,h2,p4_taum_hrest,p4_taup_hrest = self.gethvecs()
        h1 = h1.unit  # new
        h2 = h2.unit  # new
        print("h1: unitvec")
        self.printinfoP3(h1)
        print("h2: unitvec")
        self.printinfoP3(h2)

        taum_hrest_pvec_unit = p4_taum_hrest.pvec.unit
        taup_hrest_pvec_unit = p4_taup_hrest.pvec.unit
        print("tau- hrest unit")
        self.printinfoP3(taum_hrest_pvec_unit)
        print("tau+ hrest unit")
        self.printinfoP3(taup_hrest_pvec_unit)
        
        #k1raw = h1.cross(p4_taum_hrest.pvec)
        #k2raw = h2.cross(p4_taup_hrest.pvec)
        k1raw = h1.cross(taum_hrest_pvec_unit) # new
        k2raw = h2.cross(taup_hrest_pvec_unit) # new
        #k1raw.absolute(), k2raw.absolute()

        #print("k1raw")
        #self.printinfoP3(k1raw, log=True)
        #print("k2raw")
        #self.printinfoP3(k2raw, log=True)
        
        k1 = k1raw.unit
        k2 = k2raw.unit

        print("k1: unitvec")
        self.printinfoP3(k1)
        print("k2: unitvec")
        self.printinfoP3(k2)
        
        angle = (h1.cross(h2)).dot(p4_taum_hrest.pvec.unit)
        print(f"Angle: {angle}")
        self.plotit(arrlist=[ak.ravel(angle).to_numpy()], bins=50, log=True)
        
        #temp = np.arccos(k1.dot(k2))
        #temp2 = 2*np.pi - temp
        temp = np.arctan2((k1.cross(k2)).absolute(), k1.dot(k2)) # new
        temp2 = (2*np.pi - temp)                                 # new
        self.plotit(arrlist=[ak.ravel(temp).to_numpy(),
                             ak.ravel(temp2).to_numpy()],
                    bins=9,
                    log=False,
                    dim=(1,2))
        phicp = ak.where(angle <= 0.0, temp, temp2)

        print(f"PhiCP: {phicp}")
        self.plotit(arrlist=[ak.ravel(phicp).to_numpy()], bins=9)
        
        return phicp

    
        
    def geta1vecsDP(self, p4_os_pi, p4_ss1_pi, p4_ss2_pi):
        #print(f"{p4_ss1_pi.pdgId}, {p4_ss2_pi.pdgId}")
        Minv1 = (p4_os_pi + p4_ss1_pi).mass
        Minv2 = (p4_os_pi + p4_ss2_pi).mass
        
        Pi = ak.where(np.abs(0.77526-Minv1) < np.abs(0.77526-Minv2), p4_ss1_pi, p4_ss2_pi)
        PiZero = p4_os_pi
        
        return Pi, PiZero

    
    def getPhiCP_DP(self, PiPlus, PiMinus, PiZeroPlus, PiZeroMinus):

        ZMF = PiPlus + PiMinus
        boostv = ZMF.boostvec
        # minus side
        PiMinus_ZMF = PiMinus.boost(boostv.negative())
        PiZeroMinus_ZMF = PiZeroMinus.boost(boostv.negative())
        vecPiMinus = PiMinus_ZMF.pvec.unit
        vecPiZeroMinus = PiZeroMinus_ZMF.pvec.unit
        print(f"vecPiZeroMinus: {vecPiZeroMinus}")
        print(f"vecPiMinus: {vecPiMinus}")
        vecPiZeroMinustransv = (vecPiZeroMinus - vecPiMinus*(vecPiMinus.dot(vecPiZeroMinus))).unit
        # plus side
        PiPlus_ZMF = PiPlus.boost(boostv.negative())
        PiZeroPlus_ZMF = PiZeroPlus.boost(boostv.negative())
        vecPiPlus = PiPlus_ZMF.pvec.unit
        vecPiZeroPlus = PiZeroPlus_ZMF.pvec.unit
        print(f"vecPiZeroPlus: {vecPiZeroPlus}")
        print(f"vecPiPlus: {vecPiPlus}")
        vecPiZeroPlustransv = (vecPiZeroPlus - vecPiPlus*(vecPiPlus.dot(vecPiZeroPlus))).unit
        # Y variable
        Y1 = (PiMinus.energy - PiZeroMinus.energy)/(PiMinus.energy + PiZeroMinus.energy)
        Y2 = (PiPlus.energy - PiZeroPlus.energy)/(PiPlus.energy + PiZeroPlus.energy)
        Y  = Y1*Y2
        # angle
        acop_DP_1 = np.arccos(vecPiZeroPlustransv.dot(vecPiZeroMinustransv))
        sign_DP   = vecPiMinus.dot(vecPiZeroPlustransv.cross(vecPiZeroMinustransv))
        sign_mask = sign_DP < 0.0 
        acop_DP_2 = ak.where(sign_mask, 2*np.pi - acop_DP_1, acop_DP_1)
        Y_mask    = Y < 0.0
        acop_DP_3 = ak.where(Y_mask, acop_DP_2 + np.pi, acop_DP_2)
        mask      = Y_mask & (acop_DP_3 > 2*np.pi)
        acop_DP   = ak.where(mask, acop_DP_3 - 2*np.pi, acop_DP_3)

        #self.plotit(arrlist=[ak.ravel(acop_DP).to_numpy()], bins=9)
            
        
        return acop_DP

    
    def comp_PhiCP_DP(self):
        """
        Acoplanarity angle in decay plane method
        """
        PiMinus = None
        PiZeroMinus = None
        PiPlus = None
        PiZeroPlus = None

        if self.cat == "rhorho":
            h_boostvec, p4_taum, p4_taup, p4_taum_pi, p4_taup_pi, p4_taum_pi0, p4_taup_pi0, p4_taum_nu, p4_taup_nu, p4_taum_hrest, p4_taup_hrest = self.get_evtinfo_rhorho()
            PiMinus = p4_taum_pi
            PiZeroMinus = p4_taum_pi0
            PiPlus = p4_taup_pi
            PiZeroPlus = p4_taup_pi0

        elif self.cat == "a1rho":
            h_boostvec, p4_taup, p4_taup_pi, p4_taup_pi0, p4_taup_nu, p4_taum_os_pi, p4_taum_ss1_pi, p4_taum_ss2_pi, p4_taum_hrest, p4_taup_hrest = self.get_evtinfo_a1rho_leg1() # tau- > a1, tau+ > rho
            PiPlus_1 = p4_taup_pi
            PiZeroPlus_1 = p4_taup_pi0
            PiMinus_1, PiZeroMinus_1 = self.geta1vecsDP(p4_taum_os_pi, p4_taum_ss1_pi, p4_taum_ss2_pi) 

            h_boostvec, p4_taum, p4_taum_pi, p4_taum_pi0, p4_taum_nu, p4_taup_os_pi, p4_taup_ss1_pi, p4_taup_ss2_pi, p4_taum_hrest, p4_taup_hrest = self.get_evtinfo_a1rho_leg2() # tau+ > a1, tau- > rho
            PiMinus_2 = p4_taum_pi
            PiZeroMinus_2 = p4_taum_pi0
            PiPlus_2, PiZeroPlus_2 = self.geta1vecsDP(p4_taup_os_pi, p4_taup_ss1_pi, p4_taup_ss2_pi) 

            PiMinus = ak.concatenate([PiMinus_1, PiMinus_2], axis=0)
            PiZeroMinus = ak.concatenate([PiZeroMinus_1, PiZeroMinus_2], axis=0)
            PiPlus = ak.concatenate([PiPlus_1, PiPlus_2], axis=0)
            PiZeroPlus = ak.concatenate([PiZeroPlus_1, PiZeroPlus_2], axis=0)
            
        elif self.cat == "a1a1":
            h_boostvec, p4_taum, p4_taup, p4_taum_os_pi, p4_taup_os_pi, p4_taum_ss1_pi, p4_taup_ss1_pi, p4_taum_ss2_pi, p4_taup_ss2_pi, p4_taum_hrest, p4_taup_hrest = self.get_evtinfo_a1a1()
            PiMinus, PiZeroMinus = self.geta1vecsDP(p4_taum_os_pi, p4_taum_ss1_pi, p4_taum_ss2_pi)
            PiPlus, PiZeroPlus   = self.geta1vecsDP(p4_taup_os_pi, p4_taup_ss1_pi, p4_taup_ss2_pi)
        else:
            raise RuntimeWarning (f"{self.cat} is not suitable for DP method")

        phicp = self.getPhiCP_DP(PiPlus, PiMinus, PiZeroPlus, PiZeroMinus)
        self.plotit(arrlist=[ak.ravel(phicp).to_numpy()], bins=9)

        return phicp

    
    def getIP(self, p4, p3):
        dirvec = p4.pvec
        proj = p3.dot(dirvec.unit)
        return p3 - dirvec*proj


    def comp_PhiCP_IP(self):
        if self.cat == "pipi":
            boostvec, p4_taum_pi, p4_taup_pi, p4_taum_hrest, p4_taup_hrest = self.get_evtinfo_pipi()
            
        else:
            raise RuntimeError(f"{self.cat} is not suitable for DP method")


        #self.plotit(arrlist=[ak.ravel(acop_IP).to_numpy()], bins=9)
        #return acop_IP
        pass
