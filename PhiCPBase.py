import os
import numpy as np
import awkward as ak
from TComplex import TComplex
from PolarimetricA1 import PolarimetricA1
#from PolarimetricA1_CV_KS import PolarimetricVectorA1
from coffea.nanoevents.methods import vector
import matplotlib.pyplot as plt

class PhiCPBase:

    def __init__(self, 
                 taum: ak.Array = None, 
                 taup: ak.Array = None, 
                 taum_decay: ak.Array = None, 
                 taup_decay: ak.Array = None):
        self.taum = taum
        self.taup = taup
        self.taum_decay = taum_decay
        self.taup_decay = taup_decay

        
    def selcols(self, mask) -> ak.Array:
        #mask = ak.flatten(mask)
        sel_taum = self.taum[mask]
        sel_taup = self.taup[mask]
        sel_taum_decay = self.taum_decay[mask]
        sel_taup_decay = self.taup_decay[mask]
        
        return {'taum': sel_taum, 
                'taup': sel_taup, 
                'taum_decay': sel_taum_decay, 
                'taup_decay': sel_taup_decay}

    """
    def getp4(self, genarr: ak.Array, setmass=False) -> ak.Array:
        if setmass:
            return ak.zip(
                {
                    "pt": genarr.pt,
                    "eta": genarr.eta,
                    "phi": genarr.phi,
                    "mass": genarr.mass + 1.78,
                    #"pdgId": genarr.pdgId
                },
                with_name="PtEtaPhiMLorentzVector",
                behavior=vector.behavior
            )    

        else:
            return ak.zip(
                {
                    "pt": genarr.pt,
                    "eta": genarr.eta,
                    "phi": genarr.phi,
                    "mass": genarr.mass,
                    #"pdgId": genarr.pdgId
                },
                with_name="PtEtaPhiMLorentzVector",
                behavior=vector.behavior
            )    
    """

    
    def setp4(self, name="LorentzVector", *args):
        if len(args) < 4:
            raise RuntimeError ("Need at least four components")

        if name == "PtEtaPhiMLorentzVector":
            return ak.zip(
                {
                    "pt": args[0],
                    "eta": args[1],
                    "phi": args[2],
                    "mass": args[3],
                },
                with_name=name,
                behavior=vector.behavior
            )

        else:
            return ak.zip(
                {
                    "x": args[0],
                    "y": args[1],
                    "z": args[2],
                    "t": args[3],
                },
                with_name=name,
                behavior=vector.behavior
            )



    def getp4(self, genarr: ak.Array) -> ak.Array:
        return ak.zip(
            {
                "pt": genarr.pt,
                "eta": genarr.eta,
                "phi": genarr.phi,
                "mass": genarr.mass,
                #"pdgId": genarr.pdgId
            },
            with_name="PtEtaPhiMLorentzVector",
            behavior=vector.behavior
        )



    """
    def setp4_ptetaphim(self, *args):
        if len(args) < 4:
            raise RuntimeError ("Need at least four components")

        return ak.zip(
            {
                "pt": args[0],
                "eta": args[1],
                "phi": args[2],
                "mass": args[3],
            },
            with_name="PtEtaPhiMLorentzVector",
            behavior=vector.behavior
        )
    """
        
        
    def plotit(self, arrlist=[], bins=100, log=False, dim=(1,1)):
        print(" ---> Plotting ---> ")
        if len(arrlist) > 1 :
            fig, axs = plt.subplots(dim[0], dim[1], figsize=(2.5*dim[1], 2.5))
            for i,arr in enumerate(arrlist):
                axs[i].hist(arr, bins, density=True, log=log, alpha=0.7)
        else:
            fig, ax = plt.subplots()
            ax.hist(arrlist[0], bins, density=True, log=log, alpha=0.7)

        fig.tight_layout()
        plt.show()

    
    def printinfo(self, p4, type="p", plot=True):
        if type=="p":
            print(f"    px: {p4.px}")
            print(f"    py: {p4.py}")
            print(f"    pz: {p4.pz}")
            print(f"    E: {p4.energy}")
            if plot:
                self.plotit(arrlist=[ak.ravel(p4.px).to_numpy(),
                                     ak.ravel(p4.py).to_numpy(),
                                     ak.ravel(p4.pz).to_numpy(),
                                     ak.ravel(p4.energy).to_numpy()],
                            bins=50,
                            log=False,
                            dim=(1,4))
        elif type=="pt":
            print(f"    pt: {p4.pt}")
            print(f"    eta: {p4.eta}")
            print(f"    phi: {p4.phi}")
            print(f"    M: {p4.mass}")
            if plot:
                self.plotit(arrlist=[ak.ravel(p4.pt).to_numpy(),
                                     ak.ravel(p4.eta).to_numpy(),
                                     ak.ravel(p4.phi).to_numpy(),
                                     ak.ravel(p4.mass).to_numpy()],
                            bins=50,
                            log=False,
                            dim=(1,4))
        else:
            #print(f"    px: {p4.x}")
            #print(f"    py: {p4.y}")
            #print(f"    pz: {p4.z}")
            #print(f"    E: {p4.t}")
            print(f"    x: {p4.x}")
            print(f"    y: {p4.y}")
            print(f"    z: {p4.z}")
            print(f"    t: {p4.t}")
            if plot:
                self.plotit(arrlist=[ak.ravel(p4.x).to_numpy(),
                                     ak.ravel(p4.y).to_numpy(),
                                     ak.ravel(p4.z).to_numpy(),
                                     ak.ravel(p4.t).to_numpy()],
                            bins=50,
                            log=False,
                            dim=(1,4))


    def printinfoP3(self, p3, plot=True, log=False):
        print(f"    x: {p3.x}")
        print(f"    y: {p3.y}")
        print(f"    z: {p3.z}")
        if plot:
            self.plotit(arrlist=[ak.ravel(p3.x).to_numpy(),
                                 ak.ravel(p3.y).to_numpy(),
                                 ak.ravel(p3.z).to_numpy()],
                        bins=50,
                        log=log,
                        dim=(1,3))
    
    
    def inspect(self, arr):
        print("....ispecting....")
        print(f"  dimension: {arr.ndim}")
        print(f"  type: {arr.type}")
        ne = np.count(arr) if arr.ndim == 1 else np.sum(arr, axis=0)
        print(f"  nEntries: {ne}")
        
        
    def Mag2(self, vect):
        vect4 = vect
        vect3 = vect.pvec
        return ((vect4.t)*(vect4.t) - (vect3.absolute())*(vect3.absolute()))

        
    def taum_pinu(self) -> ak.Array:
        mask = ((ak.num(self.taum_decay.pdgId, axis=-1) == 2) 
                 & (
                     (ak.sum(self.taum_decay.pdgId == -211, axis=-1) == 1)
                     | (ak.sum(self.taum_decay.pdgId == -311, axis=-1) == 1)
                 ) & (ak.sum(self.taum_decay.pdgId == 16, axis=-1) == 1))
        return mask

    def taup_pinu(self) -> ak.Array:
        mask = ((ak.num(self.taup_decay.pdgId, axis=-1) == 2) 
                 & (
                     (ak.sum(self.taup_decay.pdgId == 211, axis=-1) == 1)
                     | (ak.sum(self.taup_decay.pdgId == 311, axis=-1) == 1)
                 ) & (ak.sum(self.taup_decay.pdgId == -16, axis=-1) == 1))
        return mask
    
    def taum_rho(self) -> ak.Array:
        mask = ((ak.num(self.taum_decay.pdgId, axis=-1) == 3) 
                 & (
                     (ak.sum(self.taum_decay.pdgId == -211, axis=-1) == 1)
                     | (ak.sum(self.taum_decay.pdgId == -311, axis=-1) == 1)
                 ) & (ak.sum(self.taum_decay.pdgId == 111, axis=-1) == 1)
                 & (ak.sum(self.taum_decay.pdgId == 16, axis=-1) == 1))
        return mask
    
    def taup_rho(self) -> ak.Array:
        mask = ((ak.num(self.taup_decay.pdgId, axis=-1) == 3)
                 & (
                     (ak.sum(self.taup_decay.pdgId == 211, axis=-1) == 1)
                     | (ak.sum(self.taup_decay.pdgId == 311, axis=-1) == 1)
                 ) & (ak.sum(self.taup_decay.pdgId == 111, axis=-1) == 1)
                 & (ak.sum(self.taup_decay.pdgId == -16, axis=-1) == 1))
        return mask

    def taum_a1(self) -> ak.Array:
        # pid 311 will make things complicated
        mask = ((ak.num(self.taum_decay.pdgId, axis=-1) == 4) 
                 & (ak.sum(self.taum_decay.pdgId == -211, axis=-1) == 2)
                 & (ak.sum(self.taum_decay.pdgId == 211, axis=-1) == 1)
                 & (ak.sum(self.taum_decay.pdgId == 16, axis=-1) == 1))
        return mask
    
    def taup_a1(self) -> ak.Array:
        # pid 311 will make things complicated
        mask = ((ak.num(self.taup_decay.pdgId, axis=-1) == 4) 
                 & (ak.sum(self.taup_decay.pdgId == 211, axis=-1) == 2)
                 & (ak.sum(self.taup_decay.pdgId == -211, axis=-1) == 1)
                 & (ak.sum(self.taup_decay.pdgId == -16, axis=-1) == 1))
        return mask
    
    def getframe(self, sel_taum: ak.Array, sel_taup: ak.Array):
        p4taum = self.getp4(sel_taum)
        p4taup = self.getp4(sel_taup)
        return p4taum+p4taup

    def get_evtinfo_pipi(self):
        print(" --- gethvecs_pipi ---")
        # get the mask to select tau-tau legs
        mask = self.taum_pinu() & self.taup_pinu()
        print(f"Selection of pi-pi pair [mask]: {mask}")
        print(f"n total events: {ak.count(mask)}")
        print(f"n selected events: {ak.sum(mask)}")
        selcoldict = self.selcols(mask)
        
        p4_taum = self.getp4(selcoldict['taum'])
        p4_taup = self.getp4(selcoldict['taup'])
        print("tau- in lab frame:")
        self.printinfo(p4_taum, "p")
        print("tau+ in lab frame:")
        self.printinfo(p4_taup, "p")  

        ispim = (selcoldict['taum_decay'].pdgId == -211) | (selcoldict['taum_decay'].pdgId == -311)
        ispip = (selcoldict['taup_decay'].pdgId == 211) | (selcoldict['taup_decay'].pdgId == 311)

        p4_taum_nu = self.getp4(selcoldict['taum_decay'][selcoldict['taum_decay'].pdgId == 16]) # tau- nu
        p4_taum_pi = self.getp4(selcoldict['taum_decay'][ispim]) # tau- pi-

        p4_taup_nu = self.getp4(selcoldict['taup_decay'][selcoldict['taup_decay'].pdgId == -16]) # tau+ nu
        p4_taup_pi = self.getp4(selcoldict['taup_decay'][ispip]) # tau- pi-

        frame = self.getframe(p4_taum_pi, p4_taup_pi)
        print("pip p4 + pim p4 ...")
        self.printinfo(frame, "p")

        boostvec = frame.boostvec
        print("Boost vec: ")
        self.printinfoP3(boostvec)        
        
        p4_taum_hrest = p4_taum.boost(boostvec.negative()) # Get the tau- in the h rest frame
        p4_taup_hrest = p4_taup.boost(boostvec.negative()) # Get the tau+ in the h rest frame

        print("tau- in rest frame:")
        self.printinfo(p4_taum_hrest, "p")
        print("tau+ in rest frame:")
        self.printinfo(p4_taup_hrest, "p")  

        return boostvec, p4_taum_pi, p4_taup_pi, p4_taum_hrest, p4_taup_hrest


    def get_evtinfo_pirho_leg1(self):
        print(" --- gethvecs_rhopi --- ")
        mask = self.taum_rho() & self.taup_pinu()
        print(f"Selection of rho-pi pair [mask]: {mask}")
        print(f"n total events: {ak.count(mask)}")
        print(f"n selected events: {ak.sum(mask)}")
        selcoldict = self.selcols(mask)
        
        p4_taum = self.getp4(selcoldict['taum'])
        p4_taup = self.getp4(selcoldict['taup'])
        print("tau- in lab frame:")
        self.printinfo(p4_taum, "p")
        print("tau+ in lab frame:")
        self.printinfo(p4_taup, "p")  

        ispim = (selcoldict['taum_decay'].pdgId == -211) | (selcoldict['taum_decay'].pdgId == -311)
        ispip = (selcoldict['taup_decay'].pdgId == 211) | (selcoldict['taup_decay'].pdgId == 311)

        p4_taum_nu = self.getp4(selcoldict['taum_decay'][selcoldict['taum_decay'].pdgId == 16]) # tau- nu
        p4_taum_pi = self.getp4(selcoldict['taum_decay'][ispim]) # tau- pi-
        p4_taum_pi0 = self.getp4(selcoldict['taum_decay'][selcoldict['taum_decay'].pdgId == 111]) # tau- pi-

        p4_taup_nu = self.getp4(selcoldict['taup_decay'][selcoldict['taup_decay'].pdgId == -16]) # tau+ nu
        p4_taup_pi = self.getp4(selcoldict['taup_decay'][ispip]) # tau+ pi+

        p4_h = self.getframe(p4_taum, p4_taup)
        print("taup p4 + taum p4 ...")
        self.printinfo(p4_h, "p")
        h_boostvec = p4_h.boostvec # Get the boost of Higgs
        print("Higgs boost vec: ")
        self.printinfoP3(h_boostvec)

        p4_taum_hrest = p4_taum.boost(h_boostvec.negative()) # Get the tau- in the h rest frame
        p4_taup_hrest = p4_taup.boost(h_boostvec.negative()) # Get the tau+ in the h rest frame
        #print(p4_taum_hrest.delta_phi(p4_taup_hrest)) # check if they are opposite
        print("tau- in rest frame:")
        self.printinfo(p4_taum_hrest, "p")
        print("tau+ in rest frame:")
        self.printinfo(p4_taup_hrest, "p")  

        return h_boostvec, p4_taum, p4_taum_pi, p4_taup_pi, p4_taum_pi0, p4_taum_nu, p4_taum_hrest, p4_taup_hrest
        

    def get_evtinfo_pirho_leg2(self):
        print(" --- gethvecs_pirho_leg2 --- ")
        mask = self.taum_pinu() & self.taup_rho()
        print(f"Selection of pi-rho pair [mask]: {mask}")
        print(f"n total events: {ak.count(mask)}")
        print(f"n selected events: {ak.sum(mask)}")
        selcoldict = self.selcols(mask)
        
        p4_taum = self.getp4(selcoldict['taum'])
        p4_taup = self.getp4(selcoldict['taup'])

        ispim = (selcoldict['taum_decay'].pdgId == -211) | (selcoldict['taum_decay'].pdgId == -311)
        ispip = (selcoldict['taup_decay'].pdgId == 211) | (selcoldict['taup_decay'].pdgId == 311)
        
        p4_taum_nu = self.getp4(selcoldict['taum_decay'][selcoldict['taum_decay'].pdgId == 16]) # tau- nu
        p4_taum_pi = self.getp4(selcoldict['taum_decay'][ispim]) # tau- pi-

        p4_taup_nu = self.getp4(selcoldict['taup_decay'][selcoldict['taup_decay'].pdgId == -16]) # tau+ nu
        p4_taup_pi = self.getp4(selcoldict['taup_decay'][ispip]) # tau+ pi+
        p4_taup_pi0 = self.getp4(selcoldict['taup_decay'][selcoldict['taup_decay'].pdgId == 111]) # tau+ pi+

        p4_h = self.getframe(p4_taum, p4_taup)
        print("taup p4 + taum p4 ...")
        self.printinfo(p4_h, "p")
        h_boostvec = p4_h.boostvec # Get the boost of Higgs
        print("Higgs boost vec: ")
        self.printinfoP3(h_boostvec)

        p4_taum_hrest = p4_taum.boost(h_boostvec.negative()) # Get the tau- in the h rest frame
        p4_taup_hrest = p4_taup.boost(h_boostvec.negative()) # Get the tau+ in the h rest frame
        #print(p4_taum_hrest.delta_phi(p4_taup_hrest)) # check if they are opposite

        return h_boostvec, p4_taup, p4_taum_pi, p4_taup_pi, p4_taup_pi0, p4_taup_nu, p4_taum_hrest, p4_taup_hrest

        
    def get_evtinfo_rhorho(self):
        print(" --- gethvecs_rhorho --- ")
        mask = self.taum_rho() & self.taup_rho()
        print(f"Selection of rho-rho pair [mask]: {mask}")
        print(f"n total events: {ak.count(mask)}")
        print(f"n selected events: {ak.sum(mask)}")
        selcoldict = self.selcols(mask)
        
        p4_taum = self.getp4(selcoldict['taum'])
        p4_taup = self.getp4(selcoldict['taup'])

        ispim = (selcoldict['taum_decay'].pdgId == -211) | (selcoldict['taum_decay'].pdgId == -311)
        ispip = (selcoldict['taup_decay'].pdgId == 211) | (selcoldict['taup_decay'].pdgId == 311)        
        
        p4_taum_nu = self.getp4(selcoldict['taum_decay'][selcoldict['taum_decay'].pdgId == 16]) # tau- nu
        p4_taum_pi = self.getp4(selcoldict['taum_decay'][ispim]) # tau- pi-
        p4_taum_pi0 = self.getp4(selcoldict['taum_decay'][selcoldict['taum_decay'].pdgId == 111]) # tau- pi0

        p4_taup_nu = self.getp4(selcoldict['taup_decay'][selcoldict['taup_decay'].pdgId == -16]) # tau+ nu
        p4_taup_pi = self.getp4(selcoldict['taup_decay'][ispip]) # tau+ pi+
        p4_taup_pi0 = self.getp4(selcoldict['taup_decay'][selcoldict['taup_decay'].pdgId == 111]) # tau+ pi0

        p4_h = self.getframe(p4_taum, p4_taup)
        print("taup p4 + taum p4 ...")
        self.printinfo(p4_h, "p")
        h_boostvec = p4_h.boostvec # Get the boost of Higgs
        print("Higgs boost vec: ")
        self.printinfoP3(h_boostvec)

        p4_taum_hrest = p4_taum.boost(h_boostvec.negative()) # Get the tau- in the h rest frame
        p4_taup_hrest = p4_taup.boost(h_boostvec.negative()) # Get the tau+ in the h rest frame
        #print(p4_taum_hrest.delta_phi(p4_taup_hrest)) # check if they are opposite

        return h_boostvec, p4_taum, p4_taup, p4_taum_pi, p4_taup_pi, p4_taum_pi0, p4_taup_pi0, p4_taum_nu, p4_taup_nu, p4_taum_hrest, p4_taup_hrest


    def get_evtinfo_a1a1(self):
        print(" --- gethvecs_a1a1 --- ")
        # Prepare the collections first
        mask = self.taum_a1() & self.taup_a1()
        print(f"Selection of a1-a1 pair [mask]: {mask}")
        print(f"n total events: {ak.count(mask)}")
        print(f"n selected events: {ak.sum(mask)}")
        selcoldict = self.selcols(mask)
        
        #p4_taum = self.getp4(selcoldict['taum'], True)
        #p4_taup = self.getp4(selcoldict['taup'], True)
        p4_taum = self.getp4(selcoldict['taum'])
        p4_taup = self.getp4(selcoldict['taup'])
        p4_taum = self.setp4("PtEtaPhiMLorentzVector",
                             p4_taum.pt,
                             p4_taum.eta,
                             p4_taum.phi,
                             1.777+p4_taum.mass)
        p4_taup = self.setp4("PtEtaPhiMLorentzVector",
                             p4_taup.pt,
                             p4_taup.eta,
                             p4_taup.phi,
                             1.777+p4_taup.mass)
        
        p4_taum_os_pi = self.getp4(selcoldict['taum_decay'][selcoldict['taum_decay'].pdgId == 211]) # tau- pi+
        p4_taum_ss_pi = self.getp4(selcoldict['taum_decay'][selcoldict['taum_decay'].pdgId == -211]) # tau- pi-
        p4_taum_ss1_pi = p4_taum_ss_pi[:,0:1]
        p4_taum_ss1_pi = self.setp4("PtEtaPhiMLorentzVector",
                                    p4_taum_ss1_pi.pt,
                                    p4_taum_ss1_pi.eta,
                                    p4_taum_ss1_pi.phi,
                                    0.1396*ak.ones_like(p4_taum_ss1_pi.pt))
        p4_taum_ss2_pi = p4_taum_ss_pi[:,1:2]
        p4_taum_ss2_pi = self.setp4("PtEtaPhiMLorentzVector",
                                    p4_taum_ss2_pi.pt,
                                    p4_taum_ss2_pi.eta,
                                    p4_taum_ss2_pi.phi,
                                    0.1396*ak.ones_like(p4_taum_ss2_pi.pt))

        #print(taum_nu_cat_tautau_pinu.pdgId)
        #print(taum_pi_cat_tautau_pinu.pdgId)

        p4_taup_os_pi = self.getp4(selcoldict['taup_decay'][selcoldict['taup_decay'].pdgId == -211]) # tau+ pi+
        p4_taup_ss_pi = self.getp4(selcoldict['taup_decay'][selcoldict['taup_decay'].pdgId == 211]) # tau+ pi0
        p4_taup_ss1_pi = p4_taup_ss_pi[:,0:1]
        p4_taup_ss1_pi = self.setp4("PtEtaPhiMLorentzVector",
                                    p4_taup_ss1_pi.pt,
                                    p4_taup_ss1_pi.eta,
                                    p4_taup_ss1_pi.phi,
                                    0.1396*ak.ones_like(p4_taup_ss1_pi.pt))
        p4_taup_ss2_pi = p4_taup_ss_pi[:,1:2]
        p4_taup_ss2_pi = self.setp4("PtEtaPhiMLorentzVector",
                                    p4_taup_ss2_pi.pt,
                                    p4_taup_ss2_pi.eta,
                                    p4_taup_ss2_pi.phi,
                                    0.1396*ak.ones_like(p4_taup_ss2_pi.pt))
        
        #print(taup_nu_cat_tautau_pinu.pdgId)
        #print(taup_pi_cat_tautau_pinu.pdgId)

        p4_h = self.getframe(p4_taum, p4_taup)
        h_boostvec = p4_h.boostvec # Get the boost of Higgs
        p4_taum_hrest = p4_taum.boost(h_boostvec.negative()) # Get the tau- in the h rest frame
        p4_taup_hrest = p4_taup.boost(h_boostvec.negative()) # Get the tau+ in the h rest frame

        print(f"Opposite ?????? {p4_taum_hrest.delta_phi(p4_taup_hrest)}")
        self.plotit(arrlist=[ak.ravel(p4_taum_hrest.delta_phi(p4_taup_hrest)).to_numpy()])

        return h_boostvec, p4_taum, p4_taup, p4_taum_os_pi, p4_taup_os_pi, p4_taum_ss1_pi, p4_taup_ss1_pi, p4_taum_ss2_pi, p4_taup_ss2_pi, p4_taum_hrest, p4_taup_hrest
        

    def get_evtinfo_a1pi_leg1(self):
        print(" --- gethvecs_a1pi_leg1 --- ")
        # Prepare the collections first
        mask = self.taum_a1() & self.taup_pinu()
        print(f"Selection of a1-pi pair [mask]: {mask}")
        print(f"n total events: {ak.count(mask)}")
        print(f"n selected events: {ak.sum(mask)}")
        selcoldict = self.selcols(mask)

        ispip = (selcoldict['taup_decay'].pdgId == 211) | (selcoldict['taup_decay'].pdgId == 311)

        p4_taum = self.getp4(selcoldict['taum'])
        p4_taup = self.getp4(selcoldict['taup'])

        p4_taum_os_pi = self.getp4(selcoldict['taum_decay'][selcoldict['taum_decay'].pdgId == 211]) # tau- pi+
        p4_taum_ss_pi = self.getp4(selcoldict['taum_decay'][selcoldict['taum_decay'].pdgId == -211]) # tau- pi-
        p4_taum_ss1_pi = p4_taum_ss_pi[:,0:1]
        p4_taum_ss2_pi = p4_taum_ss_pi[:,1:2]

        #print(taum_nu_cat_tautau_pinu.pdgId)
        #print(taum_pi_cat_tautau_pinu.pdgId)
        
        p4_taup_pi = self.getp4(selcoldict['taup_decay'][ispip]) # tau+ pi+
        p4_taup_nu = self.getp4(selcoldict['taup_decay'][selcoldict['taup_decay'].pdgId == -16]) # tau+ nu
        
        #print(taup_nu_cat_tautau_pinu.pdgId)
        #print(taup_pi_cat_tautau_pinu.pdgId)

        p4_h = self.getframe(p4_taum, p4_taup)
        h_boostvec = p4_h.boostvec # Get the boost of Higgs
        p4_taum_hrest = p4_taum.boost(h_boostvec.negative()) # Get the tau- in the h rest frame
        p4_taup_hrest = p4_taup.boost(h_boostvec.negative()) # Get the tau+ in the h rest frame

        return h_boostvec, p4_taum, p4_taum_os_pi, p4_taum_ss1_pi, p4_taum_ss2_pi, p4_taup_pi, p4_taum_hrest, p4_taup_hrest

    def get_evtinfo_a1pi_leg2(self):
        print(" --- gethvecs_a1pi_leg2 --- ")
        # Prepare the collections first
        mask = self.taup_a1() & self.taum_pinu()
        print(f"Selection of a1-pi pair [mask]: {mask}")
        print(f"n total events: {ak.count(mask)}")
        print(f"n selected events: {ak.sum(mask)}")
        selcoldict = self.selcols(mask)

        ispim = (selcoldict['taum_decay'].pdgId == -211) | (selcoldict['taum_decay'].pdgId == -311)

        p4_taum = self.getp4(selcoldict['taum'])
        p4_taup = self.getp4(selcoldict['taup'])

        p4_taup_os_pi = self.getp4(selcoldict['taup_decay'][selcoldict['taup_decay'].pdgId == -211]) # tau- pi+
        p4_taup_ss_pi = self.getp4(selcoldict['taup_decay'][selcoldict['taup_decay'].pdgId == 211]) # tau- pi-
        p4_taup_ss1_pi = p4_taup_ss_pi[:,0:1]
        p4_taup_ss2_pi = p4_taup_ss_pi[:,1:2]

        #print(taum_nu_cat_tautau_pinu.pdgId)
        #print(taum_pi_cat_tautau_pinu.pdgId)
        
        p4_taum_pi = self.getp4(selcoldict['taum_decay'][ispim]) # tau+ pi+
        p4_taum_nu = self.getp4(selcoldict['taum_decay'][selcoldict['taum_decay'].pdgId == 16]) # tau+ nu
        
        #print(taup_nu_cat_tautau_pinu.pdgId)
        #print(taup_pi_cat_tautau_pinu.pdgId)

        p4_h = self.getframe(p4_taum, p4_taup)
        h_boostvec = p4_h.boostvec # Get the boost of Higgs
        p4_taum_hrest = p4_taum.boost(h_boostvec.negative()) # Get the tau- in the h rest frame
        p4_taup_hrest = p4_taup.boost(h_boostvec.negative()) # Get the tau+ in the h rest frame

        return h_boostvec, p4_taup, p4_taup_os_pi, p4_taup_ss1_pi, p4_taup_ss2_pi, p4_taum_pi, p4_taum_hrest, p4_taup_hrest

    def get_evtinfo_a1rho_leg1(self):
        print(" --- gethvecs_a1rho_leg2 --- ")
        mask = self.taup_rho() & self.taum_a1()
        print(f"Selection of rho-pi pair [mask]: {mask}")
        print(f"n total events: {ak.count(mask)}")
        print(f"n selected events: {ak.sum(mask)}")
        selcoldict = self.selcols(mask)
        
        p4_taum = self.getp4(selcoldict['taum'])
        p4_taup = self.getp4(selcoldict['taup'])
        print("tau- in lab frame:")
        self.printinfo(p4_taum, "p")
        print("tau+ in lab frame:")
        self.printinfo(p4_taup, "p")  

        ispim = (selcoldict['taum_decay'].pdgId == -211) | (selcoldict['taum_decay'].pdgId == -311)
        ispip = (selcoldict['taup_decay'].pdgId == 211) | (selcoldict['taup_decay'].pdgId == 311)

        p4_taup_nu = self.getp4(selcoldict['taup_decay'][selcoldict['taup_decay'].pdgId == -16]) # tau- nu
        p4_taup_pi = self.getp4(selcoldict['taup_decay'][ispip]) # tau- pi-
        p4_taup_pi0 = self.getp4(selcoldict['taup_decay'][selcoldict['taup_decay'].pdgId == 111]) # tau- pi-


        p4_taum_os_pi = self.getp4(selcoldict['taum_decay'][selcoldict['taum_decay'].pdgId == 211]) # tau- pi+
        p4_taum_ss_pi = self.getp4(selcoldict['taum_decay'][selcoldict['taum_decay'].pdgId == -211]) # tau- pi-
        p4_taum_ss1_pi = p4_taum_ss_pi[:,0:1]
        p4_taum_ss2_pi = p4_taum_ss_pi[:,1:2]


        p4_h = self.getframe(p4_taum, p4_taup)
        print("taup p4 + taum p4 ...")
        self.printinfo(p4_h, "p")
        h_boostvec = p4_h.boostvec # Get the boost of Higgs
        print("Higgs boost vec: ")
        self.printinfoP3(h_boostvec)

        p4_taum_hrest = p4_taum.boost(h_boostvec.negative()) # Get the tau- in the h rest frame
        p4_taup_hrest = p4_taup.boost(h_boostvec.negative()) # Get the tau+ in the h rest frame
        #print(p4_taum_hrest.delta_phi(p4_taup_hrest)) # check if they are opposite
        print("tau- in rest frame:")
        self.printinfo(p4_taum_hrest, "p")
        print("tau+ in rest frame:")
        self.printinfo(p4_taup_hrest, "p")  

        return h_boostvec, p4_taup, p4_taup_pi, p4_taup_pi0, p4_taup_nu, p4_taum_os_pi, p4_taum_ss1_pi, p4_taum_ss2_pi, p4_taum_hrest, p4_taup_hrest


    def get_evtinfo_a1rho_leg2(self):
        print(" --- gethvecs_a1rho_leg2 --- ")
        mask = self.taum_rho() & self.taup_a1()
        print(f"Selection of rho-pi pair [mask]: {mask}")
        print(f"n total events: {ak.count(mask)}")
        print(f"n selected events: {ak.sum(mask)}")
        selcoldict = self.selcols(mask)
        
        p4_taum = self.getp4(selcoldict['taum'])
        p4_taup = self.getp4(selcoldict['taup'])
        print("tau- in lab frame:")
        self.printinfo(p4_taum, "p")
        print("tau+ in lab frame:")
        self.printinfo(p4_taup, "p")  

        ispim = (selcoldict['taum_decay'].pdgId == -211) | (selcoldict['taum_decay'].pdgId == -311)
        ispip = (selcoldict['taup_decay'].pdgId == 211) | (selcoldict['taup_decay'].pdgId == 311)

        p4_taum_nu = self.getp4(selcoldict['taum_decay'][selcoldict['taum_decay'].pdgId == 16]) # tau- nu
        p4_taum_pi = self.getp4(selcoldict['taum_decay'][ispim]) # tau- pi-
        p4_taum_pi0 = self.getp4(selcoldict['taum_decay'][selcoldict['taum_decay'].pdgId == 111]) # tau- pi-


        p4_taup_os_pi = self.getp4(selcoldict['taup_decay'][selcoldict['taup_decay'].pdgId == -211]) # tau- pi+
        p4_taup_ss_pi = self.getp4(selcoldict['taup_decay'][selcoldict['taup_decay'].pdgId == 211]) # tau- pi-
        p4_taup_ss1_pi = p4_taup_ss_pi[:,0:1]
        p4_taup_ss2_pi = p4_taup_ss_pi[:,1:2]


        p4_h = self.getframe(p4_taum, p4_taup)
        print("taup p4 + taum p4 ...")
        self.printinfo(p4_h, "p")
        h_boostvec = p4_h.boostvec # Get the boost of Higgs
        print("Higgs boost vec: ")
        self.printinfoP3(h_boostvec)

        p4_taum_hrest = p4_taum.boost(h_boostvec.negative()) # Get the tau- in the h rest frame
        p4_taup_hrest = p4_taup.boost(h_boostvec.negative()) # Get the tau+ in the h rest frame
        #print(p4_taum_hrest.delta_phi(p4_taup_hrest)) # check if they are opposite
        print("tau- in rest frame:")
        self.printinfo(p4_taum_hrest, "p")
        print("tau+ in rest frame:")
        self.printinfo(p4_taup_hrest, "p")  

        return h_boostvec, p4_taum, p4_taum_pi, p4_taum_pi0, p4_taum_nu, p4_taup_os_pi, p4_taup_ss1_pi, p4_taup_ss2_pi, p4_taum_hrest, p4_taup_hrest

    
    def gethvec_pi(self, boostvec, p4_pi):
        print(" --- gethvec_pi --- ")
        print(f"Pi: in lab frame: ")
        self.printinfo(p4_pi, "p")

        p4_pi_hrest = p4_pi.boost(boostvec.negative())

        print(f"Pi: in rest frame: ")
        self.printinfo(p4_pi_hrest, "p")
        
        return p4_pi_hrest.pvec
    
        
    def gethvec_rho(self, 
                    boostvec: ak.Array, 
                    p4_tau: ak.Array, 
                    p4_pi: ak.Array, 
                    p4_pi0: ak.Array, 
                    p4_nu: ak.Array):
        print(" --- gethvec_rho --- ")
        print("Lab frame: ")
        print("  Tau: ")
        self.printinfo(p4_tau, "p")
        print("  Pi: ")
        self.printinfo(p4_pi, "p")
        print("  Pi0: ")
        self.printinfo(p4_pi0, "p")
        print("  Nu: ")
        self.printinfo(p4_nu, "p")
        
        Tau = p4_tau.boost(boostvec.negative())
        pi  = p4_pi.boost(boostvec.negative())
        pi0 = p4_pi0.boost(boostvec.negative())
        q   = pi.subtract(pi0)
        P   = Tau
        N   = P.subtract(pi.add(pi0))
        #N   = p4_nu.boost(boostvec.negative())

        print("Rest frame: ")
        print("  Tau: ")
        self.printinfo(Tau)
        print("  Pi: ")
        self.printinfo(pi)
        print("  Pi0: ")
        self.printinfo(pi0)
        print("  Pi-Pi0: ")
        self.printinfo(q)

        
        #print("Detail info about the polarimetric vector for tau to rho decay")
        #print(f"\tTau: {P}")
        #print(f"\tPi: {pi}")
        #print(f"\tPi0: {pi0}")
        #print(f"\tq: Pi - Pi0: {q}")
        #print(f"\tN: Nu: {N}")


        #norm = (1/ (2*(q.dot(N))*(q.dot(P)) - ((q.pvec.absolute())**2)*(N.dot(P))))
        #print(f"\thnorm: {norm}")
        #out = P.mass*(((2*(q.dot(N))*q.pvec).subtract(((q.pvec.absolute())**2)*N.pvec)))*norm
        #out = (((2*(q.dot(N))*q.pvec).subtract(self.Mag2(q)*N.pvec))).unit
        out = (((2*(q.dot(N))*q.pvec).subtract(self.Mag2(q)*N.pvec)))
        print(f"hvec raw: {out}")
        self.printinfoP3(out)

        print(f"mag: {out.absolute()}")
        self.plotit(arrlist=[ak.ravel(out.absolute()).to_numpy()],
                    bins=100)
        
        return out
    
    
    def gethvec_a1(self,
                   p4_h: ak.Array,
                   boostvec: ak.Array,
                   p4_tau: ak.Array, 
                   p4_os_pi: ak.Array, 
                   p4_ss1_pi: ak.Array, 
                   p4_ss2_pi: ak.Array,
                   charge: int):
        print(" --- gethvec_a1 --- ")
        print("  ===> tau in lab frame ===>")
        self.printinfo(p4_tau)
        self.printinfo(p4_tau, "pt")
        print("  ===> os pion in lab frame ===>")
        self.printinfo(p4_os_pi)
        self.printinfo(p4_os_pi, "pt")
        print("  ===> ss pion1 in lab frame ===>")
        self.printinfo(p4_ss1_pi)
        self.printinfo(p4_ss1_pi, "pt")
        print("  ===> ss pion2 in lab frame ===>")
        self.printinfo(p4_ss2_pi)
        self.printinfo(p4_ss2_pi, "pt")

        print("  ===> Boostvec: Higgs Rest Frame ===>")
        self.printinfoP3(boostvec)


        print("  -ve boost applied on tau and its decay products")
        p4_tau_HRF = p4_tau.boost(boostvec.negative())
        p4_os_pi_HRF = p4_os_pi.boost(boostvec.negative())
        p4_ss1_pi_HRF = p4_ss1_pi.boost(boostvec.negative())
        p4_ss2_pi_HRF = p4_ss2_pi.boost(boostvec.negative())

        print("  ===> tau in Higgs rest frame ===>")
        self.printinfo(p4_tau_HRF)
        self.printinfo(p4_tau_HRF, "pt")
        print("  ===> os pion in Higgs rest frame ===>")
        self.printinfo(p4_os_pi_HRF)
        self.printinfo(p4_os_pi_HRF, "pt")
        print("  ===> ss pion1 in Higgs rest frame ===>")
        self.printinfo(p4_ss1_pi_HRF)
        self.printinfo(p4_ss1_pi_HRF, "pt")
        print("  ===> ss pion2 in Higgs rest frame ===>")
        self.printinfo(p4_ss2_pi_HRF)
        self.printinfo(p4_ss2_pi_HRF, "pt")



        """
        boost_ttrf = boostvec
        tauP4_ttrf = self.getP4_rf(p4_tau, boost_ttrf)        
        r, n, k = self.getHelicityAxes(boost_ttrf, tauP4_ttrf, p4_h, p4_tau)
        print("tau in helicity basis")
        tauP4_hf   = self.getP4_hf(tauP4_ttrf, r, n, k)
        boost_trf  = tauP4_hf.boostvec
        print("    boost_trf: ")
        self.printinfoP3(boost_trf)

        
        # boost on decay products
        print(f" -----> Tau")
        p4_tau_trf = self.getP4_ttrf_hf_trf(p4_tau, boost_ttrf, r, n, k, boost_trf)
        print(f" -----> OS Pion")
        p4_os_pi_trf = self.getP4_ttrf_hf_trf(p4_os_pi, boost_ttrf, r, n, k, boost_trf)
        #p4_os_pi_trf = self.setp4_ptetaphim(p4_os_pi_trf.pt, p4_os_pi_trf.eta, p4_os_pi_trf.phi, 0.1396*ak.ones_like(p4_os_pi_trf.pt))
        #p4_os_pi_trf = self.setp4("PtEtaPhiMLorentzVector",
        #                          p4_os_pi_trf.pt,
        #                          p4_os_pi_trf.eta,
        #                          p4_os_pi_trf.phi,
        #                          0.1396*ak.ones_like(p4_os_pi_trf.pt))
        
        print(f" -----> SS1 Pion")
        p4_ss1_pi_trf = self.getP4_ttrf_hf_trf(p4_ss1_pi, boost_ttrf, r, n, k, boost_trf)
        #p4_ss1_pi_trf = self.setp4_ptetaphim(p4_ss1_pi_trf.pt, p4_ss1_pi_trf.eta, p4_ss1_pi_trf.phi, 0.1396*ak.ones_like(p4_ss1_pi_trf.pt))
        #p4_ss1_pi_trf = self.setp4("PtEtaPhiMLorentzVector",
        #                           p4_ss1_pi_trf.pt,
        #                           p4_ss1_pi_trf.eta,
        #                           p4_ss1_pi_trf.phi,
        #                           0.1396*ak.ones_like(p4_ss1_pi_trf.pt))

        print(f" -----> SS2 Pion")
        p4_ss2_pi_trf = self.getP4_ttrf_hf_trf(p4_ss2_pi, boost_ttrf, r, n, k, boost_trf)
        #p4_ss2_pi_trf = self.setp4_ptetaphim(p4_ss2_pi_trf.pt, p4_ss2_pi_trf.eta, p4_ss2_pi_trf.phi, 0.1396*ak.ones_like(p4_ss2_pi_trf.pt))
        #p4_ss2_pi_trf = self.setp4("PtEtaPhiMLorentzVector",
        #                           p4_ss2_pi_trf.pt,
        #                           p4_ss2_pi_trf.eta,
        #                           p4_ss2_pi_trf.phi,
        #                           0.1396*ak.ones_like(p4_ss2_pi_trf.pt))
        """
        
        a1pol  = PolarimetricA1(p4_tau_HRF, p4_os_pi_HRF, p4_ss1_pi_HRF, p4_ss2_pi_HRF, charge)
        #a1pol = PolarimetricA1(p4_tau_trf, p4_os_pi_trf, p4_ss1_pi_trf, p4_ss2_pi_trf, charge)

        #a1pol = PolarimetricVectorA1()
        #out = a1pol.Vector(p4_os_pi, p4_ss1_pi, p4_ss2_pi, p4_tau, charge, "k3ChargedPi")
        
        out = -a1pol.PVC().pvec
        print(f"  hvec : {out}")
        self.printinfoP3(out, plot=False)
        return out

    def getP4_ttrf_hf_trf(self, p4, boost_ttrf, r, n, k, boost_trf):
        p4_ttrf = self.getP4_rf(p4, boost_ttrf)
        p4_hf   = self.getP4_hf(p4_ttrf, r, n, k)
        p4_trf  = self.getP4_rf(p4_hf, boost_trf)

        return p4_trf
        

    def getHelicityAxes(self, boost_ttrf, tauP4_ttrf, p4_h, p4tauLF):
        # get helicity basis: r, n, k
        # get_localCoordinateSystem(evt.tauMinusP4(), &higgsP4, &boost_ttrf, hAxis_, collider_, r, n, k, verbosity_, cartesian_);
        # https://github.com/veelken/tautau-Entanglement/blob/main/src/PolarimetricVectorAlgoThreeProng0Pi0.cc#L141
        print("  --- getHelicityAxes ---")

        k = self.get_k(p4tauLF, boost_ttrf)
        h = self.get_h_higgsAxis(p4_h, boost_ttrf)
        #h = self.get_h_beamAxis(boost_ttrf)
        r = self.get_r(k, h)
        n = self.get_n(k, r)

        print(f"n.r: {n.dot(r)}")
        print(f"n.k: {n.dot(k)}")
        print(f"r.k: {r.dot(k)}")

        print(f"  plot: n.r, n.k, r.k")
        self.plotit(arrlist=[ak.ravel(n.dot(r)).to_numpy(),
                             ak.ravel(n.dot(k)).to_numpy(),
                             ak.ravel(r.dot(k)).to_numpy()],
                    dim=(1,3))
        
        return r, n, k
        

    def get_k(self, taup4_LF, boost_ttrf):
        print("     --- get_k ---")
        # -------------------------------- T E S T --------------------------------- #
        k = taup4_LF.boost(boost_ttrf.negative())
        #k = taup4_LF.boost(taup4_LF.boostvec.negative())
        out = k.pvec.unit
        self.printinfoP3(out)

        return out

    
    def get_h_higgsAxis(self, recoilP4, boost_ttrf):
        # CV: this code does not depend on the assumption that the tau pair originates from a Higgs boson decay;
        # it also works for tau pairs originating from Z/gamma* -> tau+ tau- decays
        print("     --- get_h_higgsAxis ---")
        sf = 1.01
        higgsPx = sf*recoilP4.px
        higgsPy = sf*recoilP4.py
        higgsPz = sf*recoilP4.pz
        higgsE = np.sqrt(higgsPx**2 + higgsPy**2 + higgsPz**2 + recoilP4.mass2)

        higgsP4 = self.setp4("LorentzVector", higgsPx, higgsPy, higgsPz, higgsE)
        self.printinfo(higgsP4)
        self.printinfo(higgsP4, "pt")
        
        #higgsP4_ttrf = higgsP4.boost(boost_ttrf.negative())
        #h = higgsP4_ttrf.pvec.unit
        # -------------------------------- T E S T --------------------------------- #
        h = higgsP4.pvec.unit
        print("     boost: ")

        self.printinfoP3(h)
        
        return h


    def get_h_beamAxis(self, boost_ttrf):
        # https://github.com/veelken/tautau-Entanglement/blob/main/src/get_localCoordinateSystem.cc#L32
        print("     --- get_h_beamAxis ---")
        dummybeam = ak.ones_like(boost_ttrf.x)
        beamE = 6500*dummybeam             # 6.5 TeV
        mBeamParticle = 0.938272*dummybeam # proton mass[GeV]

        beamPx = 0.0*dummybeam
        beamPy = 0.0*dummybeam
        beamPz = np.sqrt(beamE**2 - mBeamParticle**2)

        beamP4 = self.setp4("LorentzVector", beamPx, beamPy, beamPz, beamE)
        print("\t\t beamP4 before boost")
        self.printinfo(beamP4)
        
        beamP4 = beamP4.boost(boost_ttrf.negative())
        print("\t\t beamP4 after boost")
        self.printinfo(beamP4)
        
        h = beamP4.pvec.unit

        print("     boost: ")
        self.printinfoP3(h)
        
        return h

    
    
    def get_r(self, k, h):
        print("     --- get_r ---")
        costheta = k.dot(h)
        sintheta = np.sqrt(1. - costheta*costheta)
        self.plotit(arrlist=[ak.ravel(costheta).to_numpy(),
                             ak.ravel(sintheta).to_numpy()], dim=(1,2))
        r = (h - k*costheta)*(1./sintheta)

        self.printinfoP3(r)
        return r


    def get_n(self, k, r):
        print(f"     --- get_n ---")
        n = r.cross(k)
        self.printinfoP3(n)
        
        return n

    def getP4_rf(self, p4, boostv):
        print("   --- getP4_rf ---")
        out = p4.boost(boostv.negative())
        self.printinfo(out)
        self.printinfo(out, "pt", False)
        return out

    
    def getP4_hf(self, p4, r, n, k):
        # CV: rotate given four-vector to helicity frame (hf)
        print("     --- getP4_hf ---")
        p3 = p4.pvec
        pr = p3.dot(r)
        pn = p3.dot(n)
        pk = p3.dot(k)

        out = self.setp4("LorentzVector", pr, pn, pk, p4.energy)

        self.printinfo(out)
        return out
