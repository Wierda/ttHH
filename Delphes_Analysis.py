"""
Behold: a python analysis to rival ROOT's convolutedness!

This script allows you to analyse multiple root files automagically with 
dileptonic, semileptonic and full hadronic analyses. You can choose whether 
you want the results of the analyses to be plotted separately or summed over, 
and you can loop over files and plot those results together.

Is it fast? Somewhat, it can do about 300-400 events per second. 

Is it flexible? Hell naw. If you want to change the analyses themselves you're 
going to have to mess around with the Select_contents and Analysis functions. 
But since that's honestly not so different from Rivet, I won't be open to 
constructive criticism (/hj)(tone indicator, google them they're nice). 

I'll try and write docstrings for everything, but for now that's a work in progress

Cheers,
A. Renske A. C. Wierda
"""
#%%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import awkward as aw
import uproot

from itertools import combinations
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.backends.backend_pdf import PdfPages

#%%

class FourMomentum():

    def __init__(self, vector, pT = False):
        if pT == False:
            self.E = vector[0]
            self.p = np.array(vector[1:])

            self.PT = (self.p[0]**2 + self.p[1]**2)**0.5
            if self.p[0] == 0:
                self.phi = np.pi/2
                if self.PT == 0: self.eta = np.infty
                else: self.eta = np.arcsinh(self.p[2]/self.PT)
            else:
                self.eta = np.arcsinh(self.p[2]/self.PT)
                self.phi = np.arctan(self.p[1]/self.p[0])

        elif pT == True:
            m = vector[0]
            self.PT = vector[1]
            self.eta = vector[2]
            self.phi = vector[3]

            self.E = (m**2 + self.PT**2*np.cosh(self.eta)**2)**0.5
            self.p = self.PT*np.array([np.cos(self.phi), np.sin(self.phi), np.sinh(self.eta)])
        
        else:
            print('Failed to construct four-momentum, change the pT flag')
            self.E = self.PT = self.eta = self.phi = np.nan
            self.p = np.array([np.nan, np.nan, np.nan])


    def __str__(self):
        return "FourMomentum([{}, {}, {}, {}])".format(self.E, *self.p)

    def __repr__(self):
        return "FourMomentum([{}, {}, {}, {}])".format(self.E, *self.p)

    def __add__(self, other):
        return FourMomentum([self.E + other.E, *(self.p + other.p)])

    def __sub__(self, other):
        return FourMomentum([self.E - other.E, *(self.p - other.p)])

    def mass(self):
        return (self.E**2 - np.dot(self.p, self.p))**0.5

    def deltaEta(self, other):
        return np.abs(self.eta - other.eta)

    def deltaPhi(self, other):
        return np.mod(self.phi - other.phi, np.pi)

    def deltaR(self, other):
        return (self.deltaEta(other)**2 + self.deltaPhi(other)**2)**0.5

""" class Histogram():

    def __init__(self, bin_edges, nsets = 1):
        if bin_edges.ndim == 1:
            bin_edges.reshape(1, len(bin_edges))
        shape = np.array(bin_edges.shape)
        self.__bin_edges = np.full((nsets, *shape), bin_edges)
        self.__hist_heights = np.zeros((nsets, *shape - 1))
        print(self.__bin_edges.shape)

    def fill(self, values, layer):
        if values.ndim == 1:
            values = values.reshape(len(values), 1)
        hist = np.histogramdd(values, bins = self.__bin_edges[layer])
        self.__hist_heights[layer] += hist[0]

    def integral(self):
        return np.sum(self.__hist_heights * np.diff(self.__bin_edges), axis = tuple(range(-self.__hist_heights.ndim + 1, 0)))

    def get(self):
        return self.__bin_edges, self.__hist_heights

    def get_normalised_idiot(self): # Used to be get_normalised, but I use the phrase "get (...) idiot" way too much, so here we are
        integral = np.full(tuple(reversed(self.__hist_heights.shape)), self.integral()).T
        return self.__bin_edges, self.__hist_heights/integral """

class Histogram1D():

    def __init__(self, bin_edges, nsets = 1):
        self.bin_edges = bin_edges
        self.data = [[] for i in range(nsets)]

    def fill(self, values, layer):
        self.data[layer].extend(values.tolist())

    def get(self, layer):
        hist = np.histogram(self.data[layer], bins = self.bin_edges)
        return hist

    def get_normalised_idiot(self, layer): # Used to be get_normalised, but I use the phrase "get (...) idiot" way too much, so here we are
        hist = np.histogram(self.data[layer], bins = self.bin_edges, density = True)
        return hist

class Histogram2D():

    def __init__(self, bin_edges, nsets = 1):
        self.bin_edges = bin_edges
        self.data = [[] for i in range(nsets)]

    def fill(self, values, layer):
        self.data[layer].extend(values.tolist())

    def get(self, layer):
        hist = np.histogram2d(np.array(self.data[layer])[:, 0], np.array(self.data[layer])[:, 1], 
                                bins = self.bin_edges)
        return hist

    def get_normalised_idiot(self, layer): # Used to be get_normalised, but I use the phrase "get (...) idiot" way too much, so here we are
        hist = np.histogram2d(np.array(self.data[layer])[:, 0], np.array(self.data[layer])[:, 1], 
                                bins = self.bin_edges, density = True)
        return hist

def Open_file(path, reco = True):

    file = uproot.open(path)

    if reco == True:
        electron_keys = file.keys(filter_name = ['Electron*'])
        electron_keys.remove('Electron/Electron.EhadOverEem')
        electron_keys.remove("Electron/Electron.fBits")
        electron_keys.remove("Electron/Electron.Particle")
        muon_keys = file.keys(filter_name = ['Muon*'])
        muon_keys.remove("Muon/Muon.fBits")
        muon_keys.remove("Muon/Muon.Particle")
        jet_keys = file.keys(filter_name = ['Jet*'])
        jet_keys.remove('Jet/Jet.fBits')

        while jet_keys[-1] != 'Jet/Jet.PTD':
            jet_keys.pop()

        electron_data = file.arrays(filter_name = electron_keys)
        muon_data = file.arrays(filter_name = muon_keys)
        jet_data = file.arrays(filter_name = jet_keys)

    else:
        particle_keys = file.keys(filter_name = ['Particle*'])
        particle_keys.remove("Particle/Particle.fBits")
        jet_keys = file.keys(filter_name = ['GenJet*'])
        jet_keys.remove('GenJet/GenJet.fBits')

        while jet_keys[-1] != 'GenJet/GenJet.PTD':
            jet_keys.pop()

        particle_data = file.arrays(filter_name = particle_keys)
        jet_data = file.arrays(filter_name = jet_keys)

    event_keys = ['Event/Event.CrossSection']
    ht_keys = ['ScalarHT/ScalarHT.HT']
        
    event_data = file.arrays(filter_name = event_keys)
    ht_data = file.arrays(filter_name = ht_keys)
    event_data = aw.zip([event_data, ht_data])

    file.close()

    if reco == True:
        return electron_data, muon_data, jet_data, event_data
    else:
        return particle_data, jet_data, event_data


def Find_free_particles(particles, contents_dataframe):
    free_particle_indices = []
    idx = pd.IndexSlice
    for i in particles.index.get_level_values(0).unique():
        event_particles = particles.loc[idx[i, :]]
        event_jets = contents_dataframe.loc[idx[i, :, :]]

        # For every particle in the event, check that it isn't
        # in any of the event's jets
        for row in event_particles.itertuples(name = None):
            index = row[0]
            mask = event_jets['refs'] != row[2] 

            if mask.all(axis = None):
                free_particle_indices.append((i, index))

    return particles.loc[free_particle_indices]


def Select_contents(particle_data, jet_data, event_data, arange, reco = True):
    """
    Selects the batch data and returns the relevant subset as Pandas dataframes
    Input:
        - particle_data, jet_data, jet_contents_data: awkward arrays containing
          the data from the particles, the jets and the jet contents

        - arange: numpy ndarray with the indices for the batch

    Outputs the dataframes for leptons, bjets and general event data.

    Modify this function whenever new particles need to be included
    or different cuts need to be applied.
    """

    if reco == True:
        electrons = aw.to_pandas(particle_data[0][arange])
        muons = aw.to_pandas(particle_data[1][arange])

        # Reset the keys for easy concatenation
        electrons = electrons.rename(columns={key: key.replace('Electron', 'Lepton') for key in electrons.columns})
        muons = muons.rename(columns={key: key.replace('Muon', 'Lepton') for key in muons.columns})

    else:
        particles = aw.to_pandas(particle_data[arange])
        particles = particles.rename(columns={key: key.replace('Particle', 'Lepton') for key in particles.columns})
        electrons = particles[particles['Lepton.PID'].isin([11, -11])]
        muons = particles[particles['Lepton.PID'].isin([13, -13])]

    # Define cuts on electrons and muons
    electron_cuts = np.logical_and(electrons["Lepton.PT"] >= 10, electrons["Lepton.Eta"].abs() < 2.47)
    muon_cuts = np.logical_and(muons["Lepton.PT"] >= 10, muons["Lepton.Eta"].abs() < 2.7)

    # Group into a single lepton dataframe
    leptons = pd.concat([electrons[electron_cuts], muons[muon_cuts]], axis = 0)
    leptons = leptons.sort_values(['Lepton.PT'], ascending = False)
    leptons = leptons.sort_index(level = 0, sort_remaining = False)

    jets = aw.to_pandas(jet_data[arange])
    if reco == False:
        jets = jets.rename(columns={key: key.replace('GenJet', 'Jet') for key in jets.columns})

    events = aw.to_pandas(event_data[arange])
    events = events.droplevel(0, axis = 1)
    events = events.reset_index(level = 1, drop = True)
    events = events.rename(columns={'ScalarHT.HT': 'Event.HT'})

    # Define cuts on jets
    # if reco == True:
    jet_cuts = np.logical_and(jets["Jet.PT"] >= 30, jets["Jet.Eta"].abs() < 2.5)
    jets = jets[jet_cuts]

    # Find bjets
    if reco == True:
        bjet_mask = jets['Jet.BTag'] == 1
    else:
        bjet_mask = jets['Jet.Flavor'] == 5
    bjets = jets[bjet_mask]

    # Sort bjets by PT per event
    bjets = bjets.sort_values(['Jet.PT'], ascending = False)
    bjets = bjets.sort_index(level = 0, sort_remaining = False)

    # Repeat for all jets that are not bjets
    notbjets = jets[np.invert(bjet_mask)]
    notbjets = notbjets.sort_values(['Jet.PT'], ascending = False)
    notbjets = notbjets.sort_index(level = 0, sort_remaining = False)

    # jets is now ordered as bjets(high PT -> low PT) + notbjets(high PT -> low PT)
    jets = pd.concat([bjets, notbjets], axis = 0)
    jets = jets.sort_index(level = 0, sort_remaining = False)

    # Count number of leptons, jets, bjets and nonbjets
    events.assign(nLeptons = 0, nJets = 0, nBjets = 0, nNotbjets = 0)
    events['nLeptons'] = leptons.groupby(['entry']).count().iloc[:, 0]
    events = events.fillna(0)

    events['nJets'] = jets.groupby(['entry']).count().iloc[:, 0]

    events['nBjets'] = bjets.groupby(['entry']).count().iloc[:, 0]
    events = events.fillna(0)

    events['nNotbjets'] = events['nJets'] - events['nBjets']

    events.assign(Event_type = np.nan, Bjet_group = 0)
    
    # Determine event type (fullhad = 0, semilep = 1, dilep = 2)
    events.loc[events['nLeptons'] == 0, 'Event_type'] = 0
    events.loc[events['nLeptons'] == 1, 'Event_type'] = 1
    events.loc[events['nLeptons'] == 2, 'Event_type'] = 2 

    # Requirements on number of jets per event type (2, 1, 0). First column is
    # number of bjets, second is number of other jets
    reqs = np.array([[[4, 2], [5, 1], [6, 0]],
                     [[4, 4], [5, 3], [6, 2]],
                     [[4, 6], [5, 5], [6, 4]]])

    for i in range(3):
        mask = events['Event_type'] == i

        for j in range(3):
            if j < 2: maskb = np.logical_and(mask, events['nBjets'] == reqs[2 - i, j, 0])
            else: maskb = np.logical_and(mask, events['nBjets'] >= reqs[2 - i, j, 0])
            masknb = np.logical_and(maskb, events['nNotbjets'] >= reqs[2 - i, j, 1])
            events.loc[masknb, 'Bjet_group'] = j + 4

    # Calculate the energies and momenta of the jets for later use
    kinematics = jets[['Jet.Mass', 'Jet.PT', 'Jet.Eta', 'Jet.Phi']].to_numpy()
    four_momenta = [FourMomentum(kinematics[j], pT = True) for j in range(len(jets))]
    jets.insert(7, 'Jet.FourMomentum', four_momenta)

    return jets, leptons, events


def Jet_combinations(ls, jet_momenta, target_mass_1, target_mass_2):

    jet_combos = combinations(ls, 4)

    # Check each combination, and save that combination if it
    # is closer to the Higgs mass than the previous closest
    inds, dm = ([999, 999, 999, 999], np.infty)
    for a, b, c, d in jet_combos:
        mX_12 = (jet_momenta[a] + jet_momenta[b]).mass()
        mX_34 = (jet_momenta[c] + jet_momenta[d]).mass()
        chi_squared = 10*(1 - target_mass_1/mX_12)**2 + 10*(1 - target_mass_2/mX_34)**2
        if chi_squared < dm:
            dm = chi_squared
            inds[0] = a
            inds[1] = b
            inds[2] = c
            inds[3] = d

    if jet_momenta[inds[0]].PT < jet_momenta[inds[1]].PT:
        inds = [inds[1], inds[0], inds[2], inds[3]]

    if jet_momenta[inds[2]].PT < jet_momenta[inds[3]].PT:
        inds = [inds[0], inds[1], inds[3], inds[2]]

    return inds, chi_squared


# def Jet_combinations(ls, jet_momenta, target_mass):

#     jet_combos = combinations(ls, 2)

#     # Check each combination, and save that combination if it
#     # is closer to the Higgs mass than the previous closest
#     inds, dm = ([999, 999], np.infty)
#     for a, b in jet_combos:
#         momentum_12 = jet_momenta[a] + jet_momenta[b]
#         massdiff = np.sqrt((momentum_12.mass() - target_mass)**2)
#         if massdiff < dm:
#             dm = massdiff
#             inds[0] = a
#             inds[1] = b

#     if jet_momenta[inds[0]].PT < jet_momenta[inds[1]].PT:
#         inds = [inds[1], inds[0]]

#     return inds


def Make_Higgs_bosons(jet_momenta, nbjets):
    """
    Take combinations of all bjets and compute their invariant masses, then 
    take the two combinations that have masses closest to the Higgs mass to
    be the Higgs bosons. In this context, 'leading' refers to the jet combination
    with invariant mass closest to the Higgs mass.
    Input:
        - jet_momenta:  list/array of FourMomentum objects, all the bjets first,
                        followed by possible notbjets.
        - nbjets:       number of bjets in jet_momenta

    Output:
        - leading_mass:     invariant mass of the leading bjet combination
        - subleading_mass:  invariant mass of the subleading bjet combination
        - hh_mass:          invariant mass of the 4 above bjets
        - trailing_mass:    invariant mass of all the leftover bjets
    """
    # if nbjets > 6: nbjets = 6

    ## Combinatoric statistics

    ls = list(range(nbjets)) 
    inds_HH, chi_HH = Jet_combinations(ls, jet_momenta, 125, 125)
    inds_HZ, chi_HZ = Jet_combinations(ls, jet_momenta, 125, 91)
    inds_ZZ, chi_ZZ = Jet_combinations(ls, jet_momenta, 91, 91)

    ind1, ind2, ind3, ind4 = inds_HH
    # Remove the four indices from the list and
    ls.remove(ind1)
    ls.remove(ind2)
    ls.remove(ind3)
    ls.remove(ind4)

    ind5, ind6 = (999, 999)
    if nbjets == 4:
        ind5, ind6 = (4, 5)
    if nbjets == 5:
        ind5, ind6 = (*ls, 5)
    if nbjets >= 6:
        ind5, ind6 = ls[:2]


    combos = [(ind1, ind2), (ind3, ind4), (ind5, ind6)]

    momentum_12 = jet_momenta[ind1] + jet_momenta[ind2]
    momentum_34 = jet_momenta[ind3] + jet_momenta[ind4]
    momentum_56 = jet_momenta[ind5] + jet_momenta[ind6]

    momenta_ab = [momentum_12, momentum_34, momentum_56]
    momenta_abcd = [momentum_12 + momentum_34, 
                    momentum_12 + momentum_56,
                    momentum_34 + momentum_56]

    mass_ab = [momentum.mass() for momentum in momenta_ab]
    mass_abcd = [momentum.mass() for momentum in momenta_abcd]
    deltaEta = [jet_momenta[a].deltaEta(jet_momenta[b]) for a, b in combos]
    deltaPhi = [jet_momenta[a].deltaPhi(jet_momenta[b]) for a, b in combos]
    deltaR_ab = [jet_momenta[a].deltaR(jet_momenta[b]) for a, b in combos]
    deltaR_abcd = [momenta_ab[a].deltaR(momenta_ab[b]) for a, b in combinations(range(3), 2)]
    pT_ab = [momentum.PT for momentum in momenta_ab]

    mass_HZ = [(jet_momenta[inds_HZ[2*i]] + jet_momenta[inds_HZ[2*i + 1]]).mass() for i in [0, 1]]
    mass_ZZ = [(jet_momenta[inds_ZZ[2*i]] + jet_momenta[inds_ZZ[2*i + 1]]).mass() for i in [0, 1]]

    mass_XX = [mass_ab, mass_HZ, mass_ZZ]
    chi_XX = [chi_HH, chi_HZ, chi_ZZ]

    ## Summary statistics
    combis = combinations(list(range(6)), 2)
    masses = np.array([(jet_momenta[a] + jet_momenta[b]).mass() for a, b in combis])
    mass_stats = [masses.min(), masses.mean(), masses.max()]
    combis = combinations(list(range(6)), 2)
    deltaEtas = np.array([jet_momenta[a].deltaEta(jet_momenta[b]) for a, b in combis])
    deltaEta_stats = [deltaEtas.min(), deltaEtas.mean(), deltaEtas.max()]

    return chi_XX, mass_XX, deltaR_ab, deltaEta, deltaPhi, pT_ab, mass_abcd, deltaR_abcd, mass_stats, deltaEta_stats, combos


def SingleTopness(jet_momenta, bjet_combos):

    bjet_indices = np.array(bjet_combos).flatten()
    ls = list(range(len(jet_momenta)))
    for i in bjet_indices[:4]: ls.remove(i)
    xs = []
    for i in bjet_indices[:4]:
        combis = combinations(ls, 2)
        for b, c in combis:
            mW = (jet_momenta[b] + jet_momenta[c]).mass()
            mt = (jet_momenta[i] + jet_momenta[b] + jet_momenta[c]).mass()
            xWt = np.sqrt(100*(1 - 80/mW)**2 + 100*(1 - 173/mt)**2)
            xs.append(xWt)

    if len(xs) == 0: return np.nan
    else: return np.sort(xs)[0]



def Analysis(jets, free_leptons, events, event_type, hist_dict, layer, nbatch): 
    """
    Analyses the provided data for the given event_type, and fills the histograms at the given layer.
    Input:
        - jets:        Jets in the batch, Pandas dataframe.

        - free_leptons: Free leptons in the batch, Pandas dataframe

        - events:       General event information, Pandas dataframe

        - hist_dict:    Dictionary of Histograms that will be filled

        - event_type:   Either 0, 1 or 2; selects the analysis to be performed.

        - layer:        The layer to write the histograms to.

    As of right now, the same analysis is performed on each event type. 
    """

    hist_dict['cutflows'].fill(np.full(nbatch, 0), layer = layer)

    # Select only events with the given event type
    selected_events = events[events['Event_type'] == event_type]
    if len(selected_events) == 0:
        return events
    hist_dict['cutflows'].fill(np.full(len(selected_events), 1), layer = layer)

    # Fill the histograms for the number of jets before any further selection
    hist_dict['n_jets'].fill(selected_events['nJets'].to_numpy(), layer = layer)
    hist_dict['n_bjets'].fill(selected_events['nBjets'].to_numpy(), layer = layer)
    hist_dict['n_notbjets'].fill(selected_events['nNotbjets'].to_numpy(), layer = layer)

    selected_events = selected_events[selected_events['Bjet_group'] > 0]
    if len(selected_events) == 0:
        return events
    hist_dict['cutflows'].fill(np.full(len(selected_events), 2), layer = layer)

    hist_dict['yields'].fill(selected_events['Bjet_group'].to_numpy(), layer = layer)

    # Select corresponding leptons, .isin checks which indices of free leptons are also
    # in the selected events
    lepton_mask = free_leptons.index.get_level_values(0).isin(selected_events.index.get_level_values(0))
    # Fill with the transverse momentum of all the selected leptons
    hist_dict['lep_pt'].fill(free_leptons.loc[lepton_mask, 'Lepton.PT'].to_numpy(), layer = layer)

    # Take the first 6 jets to be bjets, and select the bjets in the same way as the leptons
    bjets = jets.groupby(['entry']).nth(list(range(6)))
    bjet_mask = bjets.index.get_level_values(0).isin(selected_events.index.get_level_values(0))
    # Group the bjets by 'entry', which is the first index of the MultiIndex. This gives access
    # to the nth member function for easy selection of the leeading and subleading jets.
    selected_bjets = bjets[bjet_mask].groupby(['entry'])
    for i in range(6):
        hist_dict['b_{}_pt'.format(i+1)].fill(selected_bjets.nth(i)['Jet.PT'].to_numpy(), layer = layer)
        hist_dict['b_{}_eta'.format(i+1)].fill(selected_bjets.nth(i)['Jet.Eta'].abs().to_numpy(), layer = layer)

        selected_events.loc[:, 'pT_{}'.format(i+1)] = selected_bjets.nth(i)['Jet.PT']
        selected_events.loc[:, 'eta_{}'.format(i+1)] = selected_bjets.nth(i)['Jet.Eta']

    jet_mask = jets.index.get_level_values(0).isin(selected_events.index.get_level_values(0))
    selected_jets = jets[jet_mask].groupby(['entry'])
    selected_contents = pd.concat((jets.loc[jet_mask, 'Jet.PT'], free_leptons.loc[lepton_mask, 'Lepton.PT'])).groupby(['entry'])
    hist_dict['HT_total'].fill(selected_events['Event.HT'].to_numpy(), layer = layer)
    hist_dict['HT_select'].fill(selected_contents.sum().to_numpy(), layer = layer)
    selected_events.loc[:, 'HT_select'] = selected_contents.sum().to_numpy()


    for name, group in selected_jets:
        nbjets = int(selected_events.loc[name, 'nBjets'].item())

        bjets = group.iloc[:6]
        kinematics = list(Make_Higgs_bosons(group['Jet.FourMomentum'].to_numpy(), nbjets))
        selected_events.loc[name, ['x_HH', 'x_HZ', 'x_ZZ']] = kinematics[0]
        selected_events.loc[name, ['m_12', 'm_34', 'm_56']] = kinematics[1][0]
        selected_events.loc[name, ['m_HZ_1', 'm_HZ_2']] = kinematics[1][1]
        selected_events.loc[name, ['m_ZZ_1', 'm_ZZ_2']] = kinematics[1][2]
        selected_events.loc[name, ['dR_12', 'dR_34', 'dR_56']] = kinematics[2]
        selected_events.loc[name, ['dEta_12', 'dEta_34', 'dEta_56']] = kinematics[3]
        selected_events.loc[name, ['dPhi_12', 'dPhi_34', 'dPhi_56']] = kinematics[4]
        selected_events.loc[name, ['pT_12', 'pT_34', 'pT_56']] = kinematics[5]
        selected_events.loc[name, 'm_1234'] = kinematics[6][0]
        selected_events.loc[name, ['dR_1234', 'dR_1256', 'dR_3456']] = kinematics[7]
        selected_events.loc[name, ['m_min', 'm_mean', 'm_max']] = kinematics[8]
        selected_events.loc[name, ['dEta_min', 'dEta_mean', 'dEta_max']] = kinematics[9]

        xWt = SingleTopness(group['Jet.FourMomentum'].to_numpy(), kinematics[10])
        selected_events.loc[name, 'xWt1'] = xWt


    events.loc[selected_events.index] = selected_events

    hist_dict['bb_x_HH'].fill(selected_events['x_HH'].to_numpy(), layer = layer)
    hist_dict['bb_x_HZ'].fill(selected_events['x_HZ'].to_numpy(), layer = layer)
    hist_dict['bb_x_ZZ'].fill(selected_events['x_ZZ'].to_numpy(), layer = layer)

    hist_dict['bb_12_mass'].fill(selected_events['m_12'].to_numpy(), layer = layer)
    hist_dict['bb_34_mass'].fill(selected_events['m_34'].to_numpy(), layer = layer)
    hist_dict['bb_56_mass'].fill(selected_events['m_56'].to_numpy(), layer = layer)

    hist_dict['bb_HZ_1_mass'].fill(selected_events['m_HZ_1'].to_numpy(), layer = layer)
    hist_dict['bb_HZ_2_mass'].fill(selected_events['m_HZ_2'].to_numpy(), layer = layer)
    hist_dict['bb_ZZ_1_mass'].fill(selected_events['m_ZZ_1'].to_numpy(), layer = layer)
    hist_dict['bb_ZZ_2_mass'].fill(selected_events['m_ZZ_2'].to_numpy(), layer = layer)

    hist_dict['bb_12_dR'].fill(selected_events['dR_12'].to_numpy(), layer = layer)
    hist_dict['bb_34_dR'].fill(selected_events['dR_34'].to_numpy(), layer = layer)
    hist_dict['bb_56_dR'].fill(selected_events['dR_56'].to_numpy(), layer = layer)

    hist_dict['bb_12_dEta'].fill(selected_events['dEta_12'].to_numpy(), layer = layer)
    hist_dict['bb_34_dEta'].fill(selected_events['dEta_34'].to_numpy(), layer = layer)
    hist_dict['bb_56_dEta'].fill(selected_events['dEta_56'].to_numpy(), layer = layer)

    hist_dict['bb_12_dPhi'].fill(selected_events['dPhi_12'].to_numpy(), layer = layer)
    hist_dict['bb_34_dPhi'].fill(selected_events['dPhi_34'].to_numpy(), layer = layer)
    hist_dict['bb_56_dPhi'].fill(selected_events['dPhi_56'].to_numpy(), layer = layer)

    hist_dict['bb_12_pT'].fill(selected_events['pT_12'].to_numpy(), layer = layer)
    hist_dict['bb_34_pT'].fill(selected_events['pT_34'].to_numpy(), layer = layer)
    hist_dict['bb_56_pT'].fill(selected_events['pT_56'].to_numpy(), layer = layer)

    hist_dict['xWt1'].fill(selected_events['xWt1'].to_numpy(), layer = layer)
    hist_dict['4j_1234_mass'].fill(selected_events['m_1234'].to_numpy(), layer = layer)

    hist_dict['4j_1234_dR'].fill(selected_events['dR_1234'].to_numpy(), layer = layer)
    hist_dict['4j_1256_dR'].fill(selected_events['dR_1256'].to_numpy(), layer = layer)
    hist_dict['4j_3456_dR'].fill(selected_events['dR_3456'].to_numpy(), layer = layer)

    hist_dict['m_min'].fill(selected_events['m_min'].to_numpy(), layer = layer)
    hist_dict['m_mean'].fill(selected_events['m_mean'].to_numpy(), layer = layer)
    hist_dict['m_max'].fill(selected_events['m_max'].to_numpy(), layer = layer)

    hist_dict['dEta_min'].fill(selected_events['dEta_min'].to_numpy(), layer = layer)
    hist_dict['dEta_mean'].fill(selected_events['dEta_mean'].to_numpy(), layer = layer)
    hist_dict['dEta_max'].fill(selected_events['dEta_max'].to_numpy(), layer = layer)

    hist_dict['bb_12_pT_dR_2D'].fill(selected_events[['pT_12', 'dR_12']].to_numpy(), layer = layer)
    hist_dict['bb_34_pT_dR_2D'].fill(selected_events[['pT_34', 'dR_34']].to_numpy(), layer = layer)
    hist_dict['bb_56_pT_dR_2D'].fill(selected_events[['pT_56', 'dR_56']].to_numpy(), layer = layer)

    index_mask = events.index.isin(selected_events.index.values)
    events.loc[index_mask] = selected_events

    return events


def Fill_histograms(jets, free_leptons, events, hist_dict, layer = None, analyses = [0, 1, 2], nbatch = 100):
    """
    Fill the histograms with the Analysis routine for each analysis. Passes all arguments to Analysis,
    but implements the layer = None behaviour.
    """

    events = events.assign( pT_1 = 0, eta_1 = 0, 
                            pT_2 = 0, eta_2 = 0, 
                            pT_3 = 0, eta_3 = 0, 
                            pT_4 = 0, eta_4 = 0, 
                            pT_5 = 0, eta_5 = 0, 
                            pT_6 = 0, eta_6 = 0, 
                            x_HH = 0, x_HZ = 0, x_ZZ = 0,
                            m_12 = 0, m_34 = 0, m_56 = 0, 
                            m_HZ_1 = 0, m_HZ_2 = 0,
                            m_ZZ_1 = 0, m_ZZ_2 = 0,
                            dR_12 = 0, dR_34 = 0, dR_56 = 0, 
                            dEta_12 = 0, dEta_34 = 0, dEta_56 = 0,
                            dPhi_12 = 0, dPhi_34 = 0, dPhi_56 = 0,
                            pT_12 = 0, pT_34 = 0, pT_56 = 0,
                            xWt1 = 0, HT_select = 0, m_1234 = 0,
                            dR_1234 = 0, dR_1256 = 0, dR_3456 = 0,
                            m_min = 0, m_mean = 0, m_max = 0,
                            dEta_min = 0, dEta_mean = 0, dEta_max = 0)

    for i in range(len(analyses)):
        # analyses[i] provides the event type being selected for the analysis.
        if layer == None:
            print('Analysing event type {}, filling layer {}'.format(analyses[i], i))
            events = Analysis(jets, free_leptons, events, analyses[i], hist_dict, i, nbatch)
        else:
            print('Analysing event type {}, filling layer {}'.format(analyses[i], layer))
            events = Analysis(jets, free_leptons, events, analyses[i], hist_dict, layer, nbatch)

    return events


def Analyse_file(path, analyses = [0, 1, 2], reco = True, nsets = 1, layer = None, hist_dict = None, ntot = 10000, nbatch = 100):
    """
    Analyse a given file with the following settings:
        - path:         Path to the .root file that is to be analysed

        - analyses:     Which analyses to run. 0 = full hadronic, 1 = semileptonic, 2 = dileptonic.
                        Give a list with a selection of any of these, i.e. analyses = [1] or [0, 1, 2]

        - nsets:        Number of histograms to keep track of, can be any integer. Each Histogram instance
                        has bin contents in the shape of (nsets, nbins), to keep track of the results of 
                        different analyses and/or files

        - layer:        Which of the sets of Histograms to write to (which row). Can be either None or an
                        integer. None means each analysis gets its own layer (make sure that nsets is large enough).

        - hist_dict:    If None, create a new dictionary with Histograms. If given an already existing dictionary,
                        continue to fill those histograms.
        
        - ntot:         Total number of events to be analysed.

        - nbatch:       Batch size (number of events to be analysed at the same time) (I advise against changing
                        this because of the memory size of the involved Pandas dataframes).

        Outputs the filled hist_dict.

        Example of the use of the layer and hist_dict kwargs: 
        hist_dict = Analyse_file(path, analyses = [0, 1, 2], nsets = 3, layer = None) has the same output as:
        `hist_dict = None
         for i in range(3):
             hist_dict = Analyse_file(path, analyses = [i], nsets = 3, layer = i, hist_dict = hist_dict)`
    """

    # Initiate the histogram dictionary if it hasn't been passed as a kwarg.
    # New histograms have to be added here.
    if hist_dict == None:
        hist_dict = {
        'cutflows': Histogram1D(np.linspace(-0.5, 2.5, 4), nsets = nsets),
        'n_jets': Histogram1D(np.linspace(-0.5, 16.5, 18), nsets = nsets),
        'n_bjets': Histogram1D(np.linspace(-0.5, 16.5, 18), nsets = nsets),
        'n_notbjets': Histogram1D(np.linspace(-0.5, 16.5, 18), nsets = nsets),
        'yields': Histogram1D(np.linspace(3.5, 6.5, 4), nsets = nsets),
        'b_1_pt': Histogram1D(np.linspace(0, 700, 36), nsets = nsets),
        'b_1_eta': Histogram1D(np.linspace(0, 2.5, 26), nsets = nsets),
        'b_2_pt': Histogram1D(np.linspace(0, 700, 36), nsets = nsets),
        'b_2_eta': Histogram1D(np.linspace(0, 2.5, 26), nsets = nsets),
        'b_3_pt': Histogram1D(np.linspace(0, 700, 36), nsets = nsets),
        'b_3_eta': Histogram1D(np.linspace(0, 2.5, 26), nsets = nsets),
        'b_4_pt': Histogram1D(np.linspace(0, 700, 36), nsets = nsets),
        'b_4_eta': Histogram1D(np.linspace(0, 2.5, 26), nsets = nsets),
        'b_5_pt': Histogram1D(np.linspace(0, 700, 36), nsets = nsets),
        'b_5_eta': Histogram1D(np.linspace(0, 2.5, 26), nsets = nsets),
        'b_6_pt': Histogram1D(np.linspace(0, 700, 36), nsets = nsets),
        'b_6_eta': Histogram1D(np.linspace(0, 2.5, 26), nsets = nsets),
        'bb_x_HH': Histogram1D(np.linspace(0, 15, 25), nsets = nsets),
        'bb_x_HZ': Histogram1D(np.linspace(0, 15, 25), nsets = nsets),
        'bb_x_ZZ': Histogram1D(np.linspace(0, 15, 25), nsets = nsets),
        'bb_12_mass': Histogram1D(np.linspace(0, 700, 36),nsets = nsets),
        'bb_34_mass': Histogram1D(np.linspace(0, 700, 36), nsets = nsets),
        'bb_56_mass': Histogram1D(np.linspace(0, 700, 36), nsets = nsets),
        'bb_HZ_1_mass': Histogram1D(np.linspace(0, 700, 36), nsets = nsets),
        'bb_HZ_2_mass': Histogram1D(np.linspace(0, 700, 36), nsets = nsets),
        'bb_ZZ_1_mass': Histogram1D(np.linspace(0, 700, 36), nsets = nsets),
        'bb_ZZ_2_mass': Histogram1D(np.linspace(0, 700, 36), nsets = nsets),
        'bb_12_dR': Histogram1D(np.linspace(0, 7, 25), nsets = nsets),
        'bb_34_dR': Histogram1D(np.linspace(0, 7, 25), nsets = nsets),
        'bb_56_dR': Histogram1D(np.linspace(0, 7, 25), nsets = nsets),
        'bb_12_dEta': Histogram1D(np.linspace(0, np.pi, 25), nsets = nsets),
        'bb_34_dEta': Histogram1D(np.linspace(0, np.pi, 25), nsets = nsets),
        'bb_56_dEta': Histogram1D(np.linspace(0, np.pi, 25), nsets = nsets),
        'bb_12_dPhi': Histogram1D(np.linspace(0, np.pi, 25), nsets = nsets),
        'bb_34_dPhi': Histogram1D(np.linspace(0, np.pi, 25), nsets = nsets),
        'bb_56_dPhi': Histogram1D(np.linspace(0, np.pi, 25), nsets = nsets),
        'bb_12_pT': Histogram1D(np.linspace(0, 700, 36), nsets = nsets),
        'bb_34_pT': Histogram1D(np.linspace(0, 700, 36), nsets = nsets),
        'bb_56_pT': Histogram1D(np.linspace(0, 700, 36), nsets = nsets),
        'xWt1': Histogram1D(np.linspace(0, 15, 25), nsets = nsets),
        '4j_1234_mass': Histogram1D(np.linspace(120, 2000, 48), nsets = nsets),
        '4j_1234_dR': Histogram1D(np.linspace(0, 7, 25), nsets = nsets),
        '4j_1256_dR': Histogram1D(np.linspace(0, 7, 25), nsets = nsets),
        '4j_3456_dR': Histogram1D(np.linspace(0, 7, 25), nsets = nsets),
        'lep_pt': Histogram1D(np.linspace(0, 700, 36), nsets = nsets),
        'HT_total': Histogram1D(np.linspace(320, 2000, 43), nsets = nsets),
        'HT_select': Histogram1D(np.linspace(320, 2000, 43), nsets = nsets),
        'm_min': Histogram1D(np.linspace(0, 300, 31), nsets = nsets),
        'm_mean': Histogram1D(np.linspace(0, 700, 36), nsets = nsets),
        'm_max': Histogram1D(np.linspace(120, 2000, 48), nsets = nsets),
        'dEta_min': Histogram1D(np.linspace(0, np.pi/4, 25), nsets = nsets),
        'dEta_mean': Histogram1D(np.linspace(0, np.pi, 25), nsets = nsets),
        'dEta_max': Histogram1D(np.linspace(0, 2*np.pi, 25), nsets = nsets),
        'bb_12_pT_dR_2D': Histogram2D([np.linspace(0, 700, 36), np.linspace(0, 7, 25)], nsets = nsets),
        'bb_34_pT_dR_2D': Histogram2D([np.linspace(0, 700, 36), np.linspace(0, 7, 25)], nsets = nsets),
        'bb_56_pT_dR_2D': Histogram2D([np.linspace(0, 700, 36), np.linspace(0, 7, 25)], nsets = nsets)
        }

    # Open the .root file and store the contents in 3 different awkward arrays
    print("\n\nOpening file\n")
    data = Open_file(path, reco = reco)

    events_list = []

    for i in range(ntot//nbatch):

        print('\nBatch {}'.format(i))

        # Select the batch data and store it in Pandas dataframes
        print('Filling dataframes')
        if reco == True:
            jets, leptons, events = Select_contents((data[0], data[1]), data[2], data[3], np.arange(i*nbatch, (i+1)*nbatch),
                                                    reco = reco)
        else:
            jets, leptons, events = Select_contents(data[0], data[1], data[2], np.arange(i*nbatch, (i+1)*nbatch), 
                                                    reco = reco)
        # Fill the histograms with the results of the analyses of the batch data
        print('Filling histograms')
        events = Fill_histograms(jets, leptons, events, hist_dict, layer = layer, analyses = analyses, nbatch = nbatch)
        events_list.append(events[np.logical_and(events['Bjet_group'] > 0, events['Event_type'].isin(analyses))])

    events = pd.concat(events_list, ignore_index = True)
    return hist_dict, events

# %%

decays = ['fullhad']
file_path_ttHH = ['/Users/renske/Documents/CERN/ttHH_data/ttHH_'+decay+'/tag_1_delphes_events.root:Delphes' for decay in decays]
file_path_ttHjj = ['/Users/renske/Documents/CERN/ttHH_data/ttHjj_'+decay+'/tag_1_delphes_events.root:Delphes' for decay in decays]
file_path_ttbbbb = ['/Users/renske/Documents/CERN/ttHH_data/ttbbbb_'+decay+'/tag_1_delphes_events.root:Delphes' for decay in decays]

files = [*file_path_ttbbbb, *file_path_ttHjj, *file_path_ttHH]

n_events = 10000

xs_decays = {'fullhad': 0.6741**2,
             'semilep': 2*0.2134*0.6741,
             'dilep': 0.2134**2}

xs_ttHH = 0.000744457
xs_ttHjj = 0.45982401751254454
xs_ttbbbb = 0.00012963823574710002

lumi = 140*10**3

nsets = 3
keys = [*['ttbbbb {}'.format(decay) for decay in decays], *['ttHjj {}'.format(decay) for decay in decays], *['ttHH {}'.format(decay) for decay in decays]]
analyses = [0]
hist_dict = None
event_list = []

for i in range(nsets):
    hist_dict, events = Analyse_file(files[i], layer = i, analyses = analyses, reco = True, nsets = nsets, hist_dict = hist_dict, nbatch = 500, ntot = n_events)
    if len(events) != 0:
        event_list.append(events)

events = pd.concat(event_list, keys = keys)
significance = pd.DataFrame(index = pd.Index([0, 1, 2], name = 'nLeptons'), columns = pd.Index([4, 5, 6], name = 'nBjets'))
for event_type in [0, 1, 2]:
    for group in [4, 5, 6]:
        mask1 = events['Event_type'] == event_type
        mask2 = events['Bjet_group'] == group
        selected_events = events[np.logical_and(mask1, mask2)]

        labels = np.asarray(selected_events.index.get_level_values(0).unique(), dtype = str)
        if np.all(np.char.find(labels, 'ttHjj') == -1.) and np.all(np.char.find(labels, 'ttbbbb') == -1.):
            continue
        
        signal_yields = 0
        background_yields = 0
        for decay in decays:
            if 'ttHH '+decay in labels:
                signal_yields += xs_ttHH*xs_decays[decay]*len(selected_events.loc['ttHH '+decay])/n_events
            if 'ttHjj '+decay in labels:
                background_yields += xs_ttHjj*xs_decays[decay]*len(selected_events.loc['ttHjj '+decay])/n_events
            if 'ttbbbb '+decay in labels:
                background_yields += xs_ttHjj*xs_decays[decay]*len(selected_events.loc['ttbbbb '+decay])/n_events

        if background_yields == 0: continue
        significance.loc[event_type, group] = signal_yields/background_yields**0.5

print(lumi*significance)

events.to_csv('Results_analysis_{}.csv'.format(''.join(map(str, analyses))))


#%%
colours = ['r', 'g', 'b', 'c']
hist_kwargs = pd.DataFrame({
        'cutflows': ['cutflow'],
        'n_jets': ['number of jets (j)'],
        'n_bjets': ['number of bjets (b)'],
        'n_notbjets': ['number of other jets (nb)'],
        'yields': ['yields per number of bjets (b)'],
        'b_1_pt': [r'$p_T$ of $j_1$'],
        'b_1_eta': [r'$\eta$ of $j_1$'],
        'b_2_pt': [r'$p_T$ of $j_2$'],
        'b_2_eta': [r'$\eta$ of $j_2$'],
        'b_3_pt': [r'$p_T$ of $j_3$'],
        'b_3_eta': [r'$\eta$ of $j_3$'],
        'b_4_pt': [r'$p_T$ of $j_4$'],
        'b_4_eta': [r'$\eta$ of $j_4$'],
        'b_5_pt': [r'$p_T$ of $j_5$'],
        'b_5_eta': [r'$\eta$ of $j_5$'],
        'b_6_pt': [r'$p_T$ of $j_6$'],
        'b_6_eta': [r'$\eta$ of $j_6$'],
        'bb_x_HH': [r'$\chi^2_{HH}$'],
        'bb_x_HZ': [r'$\chi^2_{HZ}$'],
        'bb_x_ZZ': [r'$\chi^2_{ZZ}$'],
        'bb_12_mass': [r'$m_{bb, 12}$'],
        'bb_34_mass': [r'$m_{bb, 34}$'],
        'bb_56_mass': [r'$m_{jj, 56}$'],
        'bb_HZ_1_mass': [r'$m_{HZ, 12}$'],
        'bb_HZ_2_mass': [r'$m_{HZ, 34}$'],
        'bb_ZZ_1_mass': [r'$m_{ZZ, 12}$'],
        'bb_ZZ_2_mass': [r'$m_{ZZ, 34}$'],
        'bb_12_dR': [r'$\Delta R_{bb, 12}$'],
        'bb_34_dR': [r'$\Delta R_{bb, 34}$'],
        'bb_56_dR': [r'$\Delta R_{jj, 56}$'],
        'bb_12_dEta': [r'$\Delta\eta_{bb, 12}$'],
        'bb_34_dEta': [r'$\Delta\eta_{bb, 34}$'],
        'bb_56_dEta': [r'$\Delta\eta_{jj, 56}$'],
        'bb_12_dPhi': [r'$\Delta\phi_{bb, 12}$'],
        'bb_34_dPhi': [r'$\Delta\phi_{bb, 34}$'],
        'bb_56_dPhi': [r'$\Delta\phi_{jj, 56}$'],
        'bb_12_pT': [r'$p_{T, bb, 12}$'],
        'bb_34_pT': [r'$p_{T, bb, 34}$'],
        'bb_56_pT': [r'$p_{T, jj, 56}$'],
        'xWt1': [r'$X_{Wt}$'],
        '4j_1234_mass': [r'$m_{4j, 1234}$'],
        '4j_1234_dR': [r'$\Delta R_{4j, 1234}$'],
        '4j_1256_dR': [r'$\Delta R_{4j, 1256}$'],
        '4j_3456_dR': [r'$\Delta R_{4j, 3456}$'],
        'lep_pt': [r'lepton $p_T$'],
        'HT_total': [r'$HT$ event'],
        'HT_select': [r'$HT$ jets + leptons'],
        'm_min': [r'$m_{min}$'],
        'm_mean': [r'$m_{mean}$'],
        'm_max': [r'$m_{max}$'],
        'dEta_min': [r'$\Delta\eta_{min}$'],
        'dEta_mean': [r'$\Delta\eta_{mean}$'],
        'dEta_max': [r'$\Delta\eta_{max}$'],
        'bb_12_pT_dR_2D': [r'$p_{T, bb, 12}$', r'$\Delta R_{bb, 12}$'],
        'bb_34_pT_dR_2D': [r'$p_{T, bb, 34}$', r'$\Delta R_{bb, 34}$'],
        'bb_56_pT_dR_2D': [r'$p_{T, bb, 56}$', r'$\Delta R_{bb, 56}$']
        }, index = ['xlabel', 'ylabel'])

with PdfPages('Results.pdf') as pdf:
    for key in hist_kwargs.columns.values:
        hist = hist_dict[key]
        if '2D' in key:
            fig, axs = plt.subplots(nrows = nsets, figsize = (7, nsets*7))
            for j in range(nsets):
                heights, xedges, yedges = hist.get(j)
                xv, yv = np.meshgrid(xedges, yedges)
                axs[j].pcolormesh(xv, yv, heights.T, cmap = 'plasma')
                axs[j].set_xlabel(hist_kwargs.loc['xlabel', key])
                axs[j].set_ylabel(hist_kwargs.loc['ylabel', key])

        else: 
            fig, ax = plt.subplots(figsize = (7, 7))
            if key in ['cutflows', 'yields']: 
                get = hist.get
            else: 
                get = hist.get_normalised_idiot
            divider = make_axes_locatable(ax)
            ax_ratio = divider.append_axes("bottom", 1.2, pad=0.1, sharex=ax)
            ax.xaxis.set_tick_params(labelbottom=False)
            for j in range(nsets):
                heights, bin_edges = get(j)
                ax.stairs(heights, bin_edges, color = colours[j], 
                        alpha = 0.6, label = keys[j])
                ax_ratio.stairs(heights/get(0)[0], bin_edges, color = colours[j], 
                        alpha = 0.6, label = keys[j])
                ax_ratio.set_xlabel(hist_kwargs.loc['xlabel', key])
                if key not in ['xWt1']: ax.set_yscale('log')
                ax.legend()

            if key in ['bb_12_mass', 'bb_34_mass', 'bb_56_mass']:
                ax.axvline(125, c = 'k', lw = 2)

        pdf.savefig(fig)
        plt.show()


# %%

file_path = file_path_ttHH[0]
file = uproot.open(file_path)

jet_keys = file.keys(filter_name = 'GenJet*')
jet_keys.remove('GenJet/GenJet.fBits')
while jet_keys[-1] != 'GenJet/GenJet.PTD':
    jet_keys.pop()
genjet_data = file.arrays(filter_name = jet_keys)
# genjets = aw.to_pandas(genjet_data[:100])

# particle_keys = file.keys(filter_name = 'Particle*')
# particle_keys.remove('Particle/Particle.fBits')
# particle_data = file.arrays(filter_name = particle_keys)
# particles = aw.to_pandas(particle_data[:100])

# leptons = particles[particles['Particle.PID'].isin([11, 13, -11, -13])]

# selected_contents = pd.concat((genjets['GenJet.PT'], leptons['Particle.PT'])).groupby(['entry'])

# file.close()


# %%

