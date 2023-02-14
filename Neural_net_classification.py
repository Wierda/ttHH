#%%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import pyhf
import gc
pyhf.set_backend("numpy")
torch.random.manual_seed(0)
np.random.seed(0)

from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.colors import LinearSegmentedColormap
from SALib.sample import saltelli
from SALib.analyze import sobol

cdict = {   'dark pink': '#F2385A',
            'dark blue': '#343844',
            'dark turquoise': '#36B1Bf',
            'light turquoise': '#4AD9D9',
            'off white': '#E9F1DF',
            'dark yellow': '#FDC536',
            'light green': '#BCD979',
            'dark green': '#9DAD6F',
            'lilac': '#BD93D8'}

colours = [cdict['dark blue'], cdict['dark pink'], cdict['dark yellow'], cdict['off white']]
cmap_red = LinearSegmentedColormap.from_list('DiHiggs_red', colors = colours, N = 256)
colours = [cdict['dark blue'], cdict['dark turquoise'], cdict['dark yellow'], cdict['off white']]
cmap_blue = LinearSegmentedColormap.from_list('DiHiggs_blue', colors = colours, N = 256)
#%%

n_events = {'ttHH': 10**6,
            'ttHZ': 10**6,
            'ttHjj': 10**6,
            'ttbbbb': 10**6,
            'ttZjj': 3*10**5,
            'ttZZ': 3*10**5,
            'tttt': 3*10**5,
            'tt': 3*10**5}

xs_decays = {'fullhad': 0.6741**2,
             'semilep': 2*0.2134*0.6741,
             'dilep': 0.2134**2,
             'inclusive': (0.6741 + 0.2134)**2}

xs = {  'ttHH': 0.78*1.11*1.07*0.34,
        'ttHjj': 329*1.13*1.07*0.58,
        'ttHZ': 1.2*1.11*1.07*0.58,
        'ttZjj': 428*1.20*0.7,
        'ttZZ': 1.5*1.20*0.7**2,
        'ttbbbb': 150,
        'tttt': 5.3*1.19*1.11,
        'tt': 834E3*1.11*1.06}

lumi = 3000

decays = ['fullhad', 'semilep', 'dilep']

#%%

file_path = '/Users/renske/Documents/CERN/Results_Analysis_012.csv'

start = file_path.find('is_')
stop = file_path.find('.csv')
analyses = list(map(int, file_path[start + 3:stop]))

events = pd.read_csv(file_path, index_col = [0, 1])

events.loc[:, ['eta_{}'.format(i) for i in range(1, 7)]] = np.abs(events[['eta_{}'.format(i) for i in range(1, 7)]])

keys = ['tt', 'tttt', 'ttbbbb', 'ttZjj', 'ttZZ', 'ttHjj', 'ttHZ', 'ttHH']
sub_events = events.loc[keys]
sub_events = sub_events.dropna(axis = 0)
del events
gc.collect()

#%%

class SimpleBinary(torch.nn.Module):
  def __init__(self, n_input):
    super(SimpleBinary, self).__init__()
    self.linear1 = torch.nn.Linear(n_input, 2000)
    self.linear2 = torch.nn.Linear(2000, 1)

    self.relu1 = torch.nn.ReLU()

  def forward(self, x):

    u1 = self.linear1(x)
    x1 = self.relu1(u1)
    u2 = self.linear2(x1)
    
    return torch.sigmoid(u2)

class SimpleMultiClass(torch.nn.Module):
  def __init__(self, n_input):
    super(SimpleMultiClass, self).__init__()
    self.linear1 = torch.nn.Linear(n_input, 100)
    self.linear2 = torch.nn.Linear(100, 100)
    self.linear3 = torch.nn.Linear(100, 100)
    self.linear4 = torch.nn.Linear(100, 3)

    self.relu1 = torch.nn.ReLU()
    self.relu2 = torch.nn.ReLU()
    self.relu3 = torch.nn.ReLU()

  def forward(self, x):

    u1 = self.linear1(x)
    x1 = self.relu1(u1)
    u2 = self.linear2(x1)
    x2 = self.relu2(u2)
    u3 = self.linear3(x2)
    x3 = self.relu3(u3)
    u4 = self.linear4(x3)

    return u4

def train_model(input, target, input_testing, target_testing, model, criterion, nbatch = 32, nepochs = 10):

    optimizer = torch.optim.Adam(model.parameters(), lr = 1E-5)

    # This is a standard training loop
    loss_hist = []
    test_loss_hist = []

    for epoch in range(nepochs): 
        print('\nEpoch {}\n'.format(epoch))

        # It is important to shuffle the dataset at each epoch so that the minibatches are not always the same
        shuffle = np.random.permutation(input.shape[0])
        input_shuffled = input[shuffle]
        target_shuffled = target[shuffle]

        nbatches = len(target_shuffled)//nbatch
        for i in range(nbatches):
            input_batch = input_shuffled[i*nbatch:(i+1)*nbatch]
            target_batch = target_shuffled[i*nbatch:(i+1)*nbatch]

            optimizer.zero_grad()

            # forward pass
            pred_batch = model(input_batch).squeeze(1)

            # compute loss
            loss = criterion(pred_batch, target_batch)
            
            # backward pass
            loss.backward()

            optimizer.step()

            optimizer.zero_grad()
      
            if i%(nbatches//10) == 0:
                print('[{}/{}]\tLoss = {}'.format(i*nbatch, len(input_shuffled), loss.item()))
            
            loss_hist.append(loss.item())
      
        test_pred = model(input_testing).squeeze(1)
        test_loss = criterion(test_pred, target_testing)
        print('Test Loss = {}'.format(test_loss.item()))

        test_loss_hist.append(test_loss.item())

    return loss_hist, test_loss_hist

# def train_model_macro(input, target, input_testing, target_testing, n_input = 10, nbatch = 32, nepochs = 10):

#     # This is a standard training loop
#     model_list = []

#     predictions = torch.Tensor(())
#     predictions = predictions.new_zeros((len(input), 4))
#     predictions_testing = torch.Tensor(())
#     predictions_testing = predictions_testing.new_zeros((len(input_testing), 4))

#     for i in range(3):
#         mask = np.isin(target.detach(), [i, 3])
#         sub_target = target[mask]
#         mask_testing = np.isin(target_testing.detach(), [i, 3])
#         sub_target_testing = target_testing[mask_testing]

#         sub_target = torch.where(sub_target == i, 1., 0.)
#         sub_target_testing = torch.where(sub_target_testing == i, 1., 0.)

#         binary_model = SimpleBinary(n_input)
#         binary_model.train()
#         train_model(input[mask], sub_target, input_testing[mask_testing], sub_target_testing, 
#                         binary_model, nbatch = nbatch, nepochs = nepochs)
#         binary_model.eval()
#         model_list.append(binary_model.state_dict())

#         binary_preds = binary_model(input).squeeze(1)
#         predictions[:, i] = binary_preds

#         binary_test = binary_model(input_testing).squeeze(1)
#         predictions_testing[:, i] = binary_test

#     return predictions, predictions_testing, model_list


def train_model_GD(input, target, input_testing, target_testing, model, criterion, lr = 1E-3, nepochs = 10):

    # This is a standard training loop
    loss_hist = []
    test_loss_hist = []

    model.train()
    for epoch in range(nepochs): 

        model.zero_grad()

        # It is important to shuffle the dataset at each epoch so that the minibatches are not always the same
        shuffle = np.random.permutation(input.shape[0])
        input_shuffled = input.detach()[shuffle]
        target_shuffled = target.detach()[shuffle]

        # forward pass
        pred = model(input_shuffled)

        # compute loss
        criterion = torch.nn.CrossEntropyLoss()
        loss = criterion(pred, target_shuffled)
        
        # backward pass
        loss.backward()

        for name, par in model.named_parameters():
            with torch.no_grad():
                model.get_parameter(name).data -= lr*par.grad
            model.get_parameter(name).grad = None

        loss_hist.append(loss.item())
        
        if epoch%100 == 0:
            print('\nEpoch {}\n'.format(epoch))
            print('Training Loss = {}'.format(loss.item()))
        
            model.eval()
            test_pred = model(input_testing)
            test_loss = criterion(test_pred, target_testing)
            print('Test Loss = {}'.format(test_loss.item()))

            test_loss_hist.append(test_loss.item())
            model.train()

    model.eval()
    test_pred = model(input_testing)
    test_loss = criterion(test_pred, target_testing)
    print('Test Loss = {}'.format(test_loss.item()))
    test_loss_hist.append(test_loss.item())

    return loss_hist, test_loss_hist


sub_events = sub_events.assign(Target = 0)
# sub_events.loc['ttZjj', 'Target'] = 1
sub_events.loc['ttHjj', 'Target'] = 1
# sub_events.loc['ttZZ', 'Target'] = 2
sub_events.loc['ttHZ', 'Target'] = 1
sub_events.loc['ttHH', 'Target'] = 2


keys = ['tt*', 'ttH*', 'ttHH']

model_list = []
shuffles = []
ntots = []

# selected_events = sub_events.copy()#[sub_events['Event_type'] == i]

n_tot = max([len(sub_events[sub_events['Target'] == 0]), 
             len(sub_events[sub_events['Target'] == 1]),
             len(sub_events[sub_events['Target'] == 2])])
sub_events = sub_events.groupby('Target', as_index = False).nth(list(range(n_tot)))

# n_tot = len(selected_events)
ntots.append(n_tot)

excluded = ['Event.CrossSection', 'nLeptons', 'Event_type', 'Bjet_group', 'Target', 
            'bTag_1', 'bTag_2', 'bTag_3', 'bTag_4']
included = ['x_HH', 'x_HZ', 'x_ZZ', 'm_12', 'm_34', 'm_HZ_1', 'm_HZ_2', 'm_ZZ_1', 'm_ZZ_2', 'xWt1']
included = [*['pT_{}'.format(i+1) for i in range(6)], *['eta_{}'.format(i+1) for i in range(6)]]
included = ['pT_12', 'pT_34', 'pT_56', 'dR_12', 'dR_34', 'dR_56', 'm_12', 'm_34', 'm_56']
included = ['m_1234', 'dR_1234', 'dR_1256', 'dR_3456', *['eta_{}'.format(i+1) for i in range(6)]]
included = ['Event.HT', 'HT_select', 'm_min', 'm_mean', 'm_max', 'dEta_min', 'dEta_mean', 'dEta_max']
included = ['pT_12', 'pT_34', 'pT_56', 'dR_12', 'dR_34', 'dR_56', 'dEta_12', 'dEta_34', 'dEta_56']
included = ['pT_12', 'pT_34', 'pT_56', *['pT_{}'.format(i+1) for i in range(6)]]
included = ['x_ZZ', 'pT_1', 'pT_2', 'pT_3', 'pT_5', 
            'm_12', 'm_34', 'pT_34', 'm_1234', 'dR_1234', 
            'dEta_mean', 'm_max', 'dEta_12', 'pT_12', 'pT_34', 'pT_56']
included = ['x_ZZ', 'x_HZ', 'm_ZZ_2',  *['pT_{}'.format(i+1) for i in range(6)], 
            'm_12', 'm_34', 'm_56', 'pT_56', 'm_1234', 'dR_1234', 'dR_3456', 'eta_1', 
            'Event.HT', 'HT_select', 'dEta_mean', 'dEta_min', 'm_max', 'dR_12', 'dR_34', 
            'dEta_12', 'pT_12', 'pT_34', 'pT_56']
included = ['x_HZ', 'm_ZZ_1', 'm_HZ_1', 'xWt1', 'm_34', 'pT_3', 'pT_5', 'pT_6', 
            'pT_56', 'dR_56', 'm_56', 'm_1234', 'dR_3456', 'dR_1256', 'eta_1', 
            'eta_2', 'Event.HT', 'HT_select', 'm_max', 'dEta_min', 'dR_34', 'pT_12', 'pT_34']
included = ['x_ZZ', 'x_HZ', 'm_HZ_1', 'm_ZZ_2', 'pT_1', 'pT_2', 'pT_4', 'pT_6', 
            'm_12', 'm_34', 'pT_56', 'm_1234', 'dR_1234', 'dR_3456', 
            'Event.HT', 'm_mean', 'm_max', 'dEta_mean', 'dR_12', 'dR_34', 'dEta_12', 'dEta_34', 'pT_12']
columns = sub_events.columns[~sub_events.columns.isin(excluded)]
X = sub_events[columns].to_numpy()
X_transformed = np.zeros_like(X)
for i in range(len(columns)):
    data_sorted = np.sort(X[:, i])
    p = np.linspace(0, 1, num = len(data_sorted), endpoint = False)
    f = lambda x: np.interp(x, data_sorted, p)
    X_transformed[:, i] = f(X[:, i])
X_scaled = (X - X.mean(axis = 0))/X.std(axis = 0)
T = sub_events['Target'].to_numpy()
X_tensor = torch.Tensor(X_transformed)
T_tensor = torch.Tensor(T).to(torch.long)

shuffle = np.random.permutation(X_tensor.shape[0])
X_shuffled = X_tensor[shuffle]
T_shuffled = T_tensor[shuffle]
shuffles.append(shuffle)

n_tot = len(shuffle)
X_training = X_shuffled[:n_tot//2]
T_training = T_shuffled[:n_tot//2]
X_testing = X_shuffled[n_tot//2:]
T_testing = T_shuffled[n_tot//2:]

#%%

n_epochs = 100
n_retrains = 3
performance = 1
test_data = sub_events.copy().iloc[shuffle[n_tot//2:]]
criterion = torch.nn.CrossEntropyLoss() 
for i in range(n_retrains):
    # model.load_state_dict(model_state_dict)
    print(len(T_training), len(T_testing))
    model = SimpleMultiClass(len(columns))

    model.train()
    loss_history = train_model(X_training, T_training, X_testing, T_testing,
                                model, criterion, nepochs = n_epochs, nbatch = 2000)
    model.eval()

    plt.figure()
    plt.plot(np.linspace(0, n_epochs, num = len(loss_history[0])), loss_history[0], c = cdict['dark blue'])
    plt.plot(np.linspace(0, n_epochs, num = len(loss_history[1])), loss_history[1], c = cdict['dark pink'], 
            ls = '', marker = '.')
    plt.show()

    model_output = model(X_testing).detach().numpy()
    predicted_classes = np.argmax(model_output, axis = 1)
    predicted_signal = predicted_classes == 2
    true_signal = T_testing.detach().numpy() == 2
    false_predictions = np.not_equal(predicted_classes, T_testing).to(bool).detach().numpy()
    false_positives = np.logical_and(predicted_signal, false_predictions)
    false_negatives = np.logical_and(true_signal, false_predictions)
    retrain_mask = false_positives #np.logical_or(false_negatives, false_positives)

    # cut1 = model_output[:, 0] <= 0.5
    # cut2 = model_output[:, 2] >= 0.25
    # cut = np.logical_and(cut1, cut2)
    # retrain_mask = np.logical_and(cut, false_positives)

    tot_retrain = np.sum(retrain_mask)
    tot_notretrain = np.sum(~retrain_mask)
    print(tot_retrain, len(retrain_mask), tot_retrain/len(retrain_mask))

    if tot_retrain/sum(predicted_signal) > performance: break

    performance = tot_retrain/sum(predicted_signal)
    model_list.append(model.state_dict())

    if i < n_retrains - 1:
        X_training = torch.Tensor(np.append(X_training, X_testing[retrain_mask][:tot_retrain//4], axis = 0))
        T_training = torch.Tensor(np.append(T_training, T_testing[retrain_mask][:tot_retrain//4], axis = 0)).to(torch.long)
        X_testing = torch.Tensor(np.append(X_testing[~retrain_mask][tot_notretrain//4:], 
                                            X_testing[retrain_mask][tot_retrain//4:], axis = 0))
        T_testing = torch.Tensor(np.append(T_testing[~retrain_mask][tot_notretrain//4:], 
                                            T_testing[retrain_mask][tot_retrain//4:], axis = 0)).to(torch.long)
        test_data = pd.concat([test_data[~retrain_mask].iloc[tot_notretrain//4:],
                            test_data[retrain_mask].iloc[tot_retrain//4:]], axis = 0)
    



#%%

def plotYieldHistos(ax, data, variable, bins, labels, colours, 
                    ratio_yscale_log = False, xlabel = None):
    n = len(labels)
    bottom = 0
    for i in range(n):
        mask = data.index.get_level_values(0) == labels[i]
        heights, edges = np.histogram(np.fmin(data.loc[mask, variable], max(bins)), 
                    bins = bins, density = False)
        scaling = len(test_data.loc[labels[i]])/len(sub_events.loc[labels[i]])
        heights = lumi*xs[labels[i]]/(n_events[labels[i]]*scaling) * heights
        # if i == n - 1: heights *= 1000

        ax.stairs(heights + bottom*(1 - i//(n - 1)), edges, 
                    baseline = bottom*(1 - i//(n - 1)), 
                    fill = bool(1 - i//(n - 1)), color = colours[i], 
                    label = labels[i], lw = 2)

        if i < n - 1: bottom += heights

    ax.set_xlim(min(bins), max(bins))
    divider = make_axes_locatable(ax)
    ax_ratio = divider.append_axes("bottom", 1.2, pad=0.1, sharex=ax)
    ax.xaxis.set_tick_params(labelbottom=False)
    centres = edges[:-1] + np.diff(edges)/2
    ax_ratio.errorbar(centres, (heights/sum(heights))/(bottom/sum(bottom)), xerr = np.diff(edges)/2,
                    ls = '', marker = 'o', color = colours[i], label = labels[i])
    ax_ratio.axhline(1, c = 'k')
    ax_ratio.set_xlabel(xlabel)
    ax_ratio.set_ylabel('ttHH / normed bkg')
    ax_ratio.set_xlim(min(bins), max(bins))
    if ratio_yscale_log == True: ax_ratio.set_yscale('log')
    else: ax_ratio.set_ylim(0, 2)

    # axs[j].set_xlabel(variables[j])
    ax.legend()
    ax.annotate(r'3000 fb$^{-1}$', xy = (1, 1.01), xycoords = 'axes fraction', 
                horizontalalignment = 'right', verticalalignment = 'bottom')
    ax.set_yscale('log')
    ax.set_ylabel('Event yields')

softmax = torch.nn.Softmax(dim = 1)
model_i = SimpleMultiClass(len(columns))
test_data_list = []
best_cuts = []
colours = [cdict['dark turquoise'], cdict['off white'], cdict['dark blue']]

model_i.load_state_dict(model_list[-1])

# X = test_data[columns].to_numpy()
# X_scaled = (X - X.mean(axis = 0))/X.std(axis = 0)
# T = test_data['Target'].to_numpy()
# X_tensor = torch.Tensor(X_scaled)
# T_tensor = torch.Tensor(T).to(torch.long)
# X_shuffled = X_tensor[shuffles[0]][ntots[0]//2:]
# T_shuffled = T_tensor[shuffles[0]][ntots[0]//2:]

multi_output = softmax(model_i(X_testing)).detach().numpy()
# test_data = test_data.iloc[shuffles[0]].iloc[n_tot//2:]

# for j in [0, 1, 2]:
#     plt.figure()
#     bottom = np.zeros(20)
#     for c in [0, 1, 2]:
#         true_signal = multi_output[test_data['Target'] == c][:, j]

#         signal = np.histogram(true_signal, density = True, bins = np.linspace(0, 1, num = 21))
#         plt.bar(np.linspace(0.025, 0.975, num = 20), signal[0], bottom = bottom,
#                 width = 0.05, alpha = 0.5, label = keys[c], color = colours[c])
#         # bottom += signal[0]

#     plt.xlabel('Prediction class {}'.format(j))
#     plt.legend()
#     plt.show()

test_data = test_data.assign(pred_0 = 0, pred_1 = 0, pred_2 = 0)
test_data.loc[:, ['pred_0', 'pred_1', 'pred_2']] = [list(multi_output[i]) for i in range(len(multi_output))]
test_data_list.append(test_data)

colours = [cdict['dark blue'], cdict['dark turquoise'], cdict['light turquoise'], 
           cdict['dark yellow'], cdict['lilac'], cdict['light green'], #cdict['dark green'],
           cdict['dark pink']]
labels = ['ttHZ', 'ttZZ', 'tttt', 'ttbbbb', 'ttHjj', 'ttZjj', 'ttHH']

for j in range(3):
    fig, ax = plt.subplots(figsize = (7, 7))
    plotYieldHistos(ax, test_data, 'pred_{}'.format(j), np.linspace(0, 1, num = 21), 
                    labels, colours, ratio_yscale_log = True)

# for a, b in [(0, 1), (0, 2), (1, 2)]:
#     plt.figure()
#     for c in [0, 1, 2]:
#         class_a = multi_output[test_data['Target'] == c][:, a]
#         class_b = multi_output[test_data['Target'] == c][:, b]

#         plt.scatter(class_a, class_b, label = keys[c], alpha = 0.5, s = 1, c = colours[c])

#     plt.xlabel('Prediction class {}'.format(a))
#     plt.ylabel('Prediction class {}'.format(b))
#     plt.legend()
#     plt.show()

predicted_classes = np.argmax(multi_output, axis = 1)

predicted_ttHH = test_data[predicted_classes == 2]
predicted_ttHX = test_data[predicted_classes == 1]
predicted_tt4b = test_data[predicted_classes == 0]

variables = ['m_12', 'm_34', 'm_1234']

bins = [np.linspace(0, 700, 21), np.linspace(0, 700, 21), np.linspace(100, 1200, 21)]
for j in range(3):
    fig, ax = plt.subplots(figsize = (7, 7))
    plotYieldHistos(ax, sub_events, variables[j], bins[j], labels, colours)
plt.show()


# for j in range(3):
#     fig, ax = plt.subplots(figsize = (7, 7))
#     plotYieldHistos(ax, predicted_ttHH, variables[j], np.linspace(0, 600, num = 21), labels, colours)
# plt.show()

# fig, ax = plt.subplots(figsize = (7, 7))
# plotYieldHistos(ax, predicted_ttHH, 'm_1234', np.linspace(100, 1000, num = 21), labels, colours)
# plt.show()

plt.figure()
ttHH = np.histogram(predicted_ttHH['Target'], density = True, bins = [-0.5, 0.5, 1.5, 2.5])
ttHX = np.histogram(predicted_ttHX['Target'], density = True, bins = [-0.5, 0.5, 1.5, 2.5])
tt4b = np.histogram(predicted_tt4b['Target'], density = True, bins = [-0.5, 0.5, 1.5, 2.5])
confusion_matrix = np.vstack((tt4b[0], ttHX[0], ttHH[0]))
plt.imshow(confusion_matrix, cmap = cmap_red, vmin = 0, vmax = 1, origin = 'lower')
plt.xticks([0, 1, 2], ['tt*', 'ttH*', 'ttHH'])
plt.yticks([0, 1, 2], ['tt*', 'ttH*', 'ttHH'])
plt.xlabel('True class')
plt.ylabel('Predicted class')
for i in range(3):
    for j in range(3):
        plt.text(j, i, np.round(confusion_matrix[i, j], 2),
                ha="center", va="center", color="w")
plt.title('$p$(True class | Predicted class)')
plt.show()

print('Fraction of events with predicted class ttHH that are ttHH events: ', ttHH[0][-1])

#%%

def significance(cut, test_data):

    signal_yields = 0
    background_yields = 0

    signal = test_data[cut]
    labels = np.asarray(signal.index.get_level_values(0).unique(), dtype = str)
    if 'ttHH' in labels:
        scaling = len(test_data.loc['ttHH'])/len(sub_events.loc['ttHH'])
        signal_yields += lumi*xs['ttHH']*len(signal.loc['ttHH'])/(n_events['ttHH']*scaling)
    if 'ttHZ' in labels:
        scaling = len(test_data.loc['ttHZ'])/len(sub_events.loc['ttHZ'])
        background_yields += lumi*xs['ttHZ']*len(signal.loc['ttHZ'])/(n_events['ttHZ']*scaling)
    if 'ttHjj' in labels:
        scaling = len(test_data.loc['ttHjj'])/len(sub_events.loc['ttHjj'])
        background_yields += lumi*xs['ttHjj']*len(signal.loc['ttHjj'])/(n_events['ttHjj']*scaling)
    if 'ttZZ' in labels:
        scaling = len(test_data.loc['ttZZ'])/len(sub_events.loc['ttZZ'])
        background_yields += lumi*xs['ttZZ']*len(signal.loc['ttZZ'])/(n_events['ttZZ']*scaling)
    if 'ttZjj' in labels:
        scaling = len(test_data.loc['ttZjj'])/len(sub_events.loc['ttZjj'])
        background_yields += lumi*xs['ttZjj']*len(signal.loc['ttZjj'])/(n_events['ttZjj']*scaling)
    if 'ttbbbb' in labels:
        scaling = len(test_data.loc['ttbbbb'])/len(sub_events.loc['ttbbbb'])
        background_yields += lumi*xs['ttbbbb']*len(signal.loc['ttbbbb'])/(n_events['ttbbbb']*scaling)
    if 'tttt' in labels:
        scaling = len(test_data.loc['tttt'])/len(sub_events.loc['tttt'])
        background_yields += lumi*xs['tttt']*len(signal.loc['tttt'])/(n_events['tttt']*scaling)
    # if 'tt' in labels:
    #     scaling = len(test_data.loc['tt'])/len(sub_events.loc['tt'])
    #     background_yields += lumi*xs['tt']*len(signal.loc['tt'])/(n_events['tt']*scaling)

    return signal_yields, background_yields


#%%

lumi = 3000

cuts_0 = np.linspace(0.12, 0.88, num = 20)
cuts_1 = np.linspace(0.41, 0.59, num = 10)
cuts_2 = np.linspace(0.01, 0.49, num = 25)
sweep_significance = np.zeros((len(cuts_0), len(cuts_1), len(cuts_2)))
n_signal = np.zeros_like(sweep_significance)
for j in range(len(cuts_0)):
    for k in range(len(cuts_1)):
        print(j, k)
        for l in range(len(cuts_2)):
            cut0 = test_data['pred_0'] <= cuts_0[j]
            cut1 = test_data['pred_1'] <= cuts_1[k]
            cut2 = test_data['pred_2'] >= cuts_2[l]
            # cut2 = test_data[['pred_1', 'pred_2']].sum(axis = 1) >= cuts[k]
            cut = cut0 & cut1 & cut2
            predicted_classes = np.argmax(multi_output, axis = 1)
            final_cut = cut # & (predicted_classes == 2)
            signal_yields, background_yields = significance(final_cut, test_data)
            if background_yields == 0: sweep_significance[j, k, l] = np.nan; continue
            sweep_significance[j, k, l] = signal_yields/background_yields**0.5
            n_signal[j, k, l] = len(test_data[cut0 & cut1 & cut2])

plt.figure()
plt.contourf(cuts_2, cuts_0, sweep_significance[:, -1, :], levels = 50, cmap = cmap_blue)
plt.colorbar()
plt.xlim(0, 0.5)
plt.ylim(0.1, 0.9)
plt.xlabel('Cut on class 2 predictions')
plt.ylabel('Cut on class 0 predictions')
plt.show()

max_value = np.nanmax(sweep_significance)
max_mask = sweep_significance >= max_value - 0.001
best_cut_ind_flat = np.nanargmax(np.where(max_mask, n_signal, 0))
best_cut_ind = np.unravel_index(best_cut_ind_flat, np.shape(n_signal))
best_cut = (cuts_0[best_cut_ind[0]], cuts_1[best_cut_ind[1]], cuts_2[best_cut_ind[2]])
print(best_cut, sweep_significance[best_cut_ind], n_signal[best_cut_ind])
best_cuts.append(best_cut)

# test_data = selected_events.iloc[shuffle].iloc[9*n_tot//10:n_tot]
best_cut_mask = (multi_output[:, 0] <= best_cut[0]) & (multi_output[:, 1] <= best_cut[1]) \
                & (multi_output[:, 2] >= best_cut[2])
#%%
predicted_classes = np.argmax(multi_output[best_cut_mask], axis = 1)

predicted_signal = test_data[best_cut_mask] #[predicted_classes == 2]
predicted_background = test_data[~best_cut_mask] #[predicted_classes != 2]

colours = [cdict['dark blue'], cdict['dark turquoise'], cdict['light turquoise'], 
           cdict['dark yellow'], cdict['lilac'], cdict['light green'], #cdict['dark green'],
           cdict['dark pink']]
labels = ['ttZZ', 'ttHZ', 'ttZjj', 'tttt', 'ttHjj', 'ttbbbb', 'ttHH']
variables = ['m_12', 'm_34', 'm_1234']

fig, axs = plt.subplots(nrows = 3, figsize = (7, 7*3))
bins = [np.linspace(0, 700, 21), np.linspace(0, 700, 21), np.linspace(100, 1200, 21)]
for j in range(3):
    plotYieldHistos(axs[j], predicted_signal, variables[j], bins[j], labels, colours)
plt.show()


plt.figure()
signal = np.histogram(predicted_signal['Target'], density = True, bins = [-0.5, 0.5, 1.5, 2.5])
background = np.histogram(predicted_background['Target'], density = True, bins = [-0.5, 0.5, 1.5, 2.5])
confusion_matrix = np.vstack((background[0], signal[0]))
plt.imshow(confusion_matrix, cmap = cmap_red, vmin = 0, vmax = 1, origin = 'lower')
plt.xticks([0, 1, 2], ['tt*', 'ttH*', 'ttHH'])
plt.yticks([0, 1], ['background', 'signal'])
plt.xlabel('True class')
plt.ylabel('Predicted class')
for i in range(2):
    for j in range(3):
        plt.text(j, i, np.round(confusion_matrix[i, j], 2),
                ha="center", va="center", color="w")
plt.title('$p$(True class | Predicted class)')
plt.show()

print('Fraction of events classified as signal that are ttHH events: ', signal[0][-1])
#%%
bounds = np.vstack((np.min(X_scaled, axis = 0), np.max(X_scaled, axis = 0))).T

problem = {
    'num_vars': len(columns),
    'names': columns.values,
    'bounds': list(bounds)
}

param_values = saltelli.sample(problem, 256)

Y = np.zeros((param_values.shape[0], 3))
for i, X in enumerate(param_values):
    Y[i] = model(torch.Tensor(X)).detach().numpy()

total_Sis = []
for i in range(3):
    Si = sobol.analyze(problem, Y[:, i])

    total_Si = pd.Series(Si['ST'], index = columns.values)
    total_Sis.append(total_Si)
    # first_Si = pd.Series(Si['S1'], index = columns.values)
    # second_Si = pd.DataFrame(np.where(np.isnan(Si['S2']), Si['S2'].T, Si['S2']), 
    #                         index = columns.values, columns = columns.values)

total_Si_all = pd.concat(total_Sis, axis = 1)
print('Input parameter ranking:')
print(total_Si_all.sort_values(2, ascending = False))

    # ranking = total_Si.sort_values(ascending = False).index
    # sorted_second_Si = second_Si.loc[ranking, ranking]
    # plt.figure()
    # plt.imshow(sorted_second_Si, cmap = cmap, origin = 'lower')
    # plt.colorbar()
    # plt.show()

# %%
# colours = ['#0D090C', cdict['dark blue'], cdict['dark turquoise'], cdict['dark yellow'], cdict['off white']]
# cmapName = 'DiHiggs'
# cmap = LinearSegmentedColormap.from_list(cmapName, colors = colours, N = 256)


# plt.figure()
# plt.hist2d(pred_list[0]['pT_34'], pred_list[0]['dR_34'], bins = 30, cmap = cmap)
# plt.colorbar()
# plt.show()

# plt.figure()
# plt.hist2d(pred_list[1]['pT_34'], pred_list[1]['dR_34'], bins = 30, cmap = cmap)
# plt.colorbar()
# plt.show()

# %% 



#%%

def significance(cut, data, btag, xs, lumi):

    signal_yields = 0
    background_yields = 0

    signal = data[cut]
    labels = np.asarray(signal.index.get_level_values(0).unique(), dtype = str)
    if 'ttHH' in labels:
        scaling = len(data.loc['ttHH'])/len(sub_events.loc['ttHH'])
        signal_yields += lumi*xs['ttHH']*btag['ttHH']*len(signal.loc['ttHH'])/(n_events['ttHH']*scaling)
    if 'ttHZ' in labels:
        scaling = len(data.loc['ttHZ'])/len(sub_events.loc['ttHZ'])
        background_yields += lumi*xs['ttHZ']*btag['ttHZ']*len(signal.loc['ttHZ'])/(n_events['ttHZ']*scaling)
    if 'ttHjj' in labels:
        scaling = len(data.loc['ttHjj'])/len(sub_events.loc['ttHjj'])
        background_yields += lumi*xs['ttHjj']*btag['ttHjj']*len(signal.loc['ttHjj'])/(n_events['ttHjj']*scaling)
    if 'ttZZ' in labels:
        scaling = len(data.loc['ttZZ'])/len(sub_events.loc['ttZZ'])
        background_yields += lumi*xs['ttZZ']*btag['ttZZ']*len(signal.loc['ttZZ'])/(n_events['ttZZ']*scaling)
    if 'ttZjj' in labels:
        scaling = len(data.loc['ttZjj'])/len(sub_events.loc['ttZjj'])
        background_yields += lumi*xs['ttZjj']*btag['ttZjj']*len(signal.loc['ttZjj'])/(n_events['ttZjj']*scaling)
    if 'ttbbbb' in labels:
        scaling = len(data.loc['ttbbbb'])/len(sub_events.loc['ttbbbb'])
        background_yields += lumi*xs['ttbbbb']*btag['ttbbbb']*len(signal.loc['ttbbbb'])/(n_events['ttbbbb']*scaling)
    if 'tttt' in labels:
        scaling = len(data.loc['tttt'])/len(sub_events.loc['tttt'])
        background_yields += lumi*xs['tttt']*btag['tttt']*len(signal.loc['tttt'])/(n_events['tttt']*scaling)
    if 'tt' in labels:
        scaling = len(data.loc['tt'])/len(sub_events.loc['tt'])
        background_yields += lumi*xs['tt']*btag['tt']*len(signal.loc['tt'])/(n_events['tt']*scaling)

    return signal_yields, background_yields

def limits(lumi, eff, bins):

    btag = {
        'ttHH': np.power((eff/0.70),6),
        'ttHjj': np.power((eff/0.70),4),
        'ttbbbb': np.power((eff/0.70),6),
        'ttHZ': np.power((eff/0.70),4)*(0.85)+np.power((eff/0.70),6)*(0.15),
        'tttt': np.power((eff/0.70),4),
        'tt': np.power((eff/0.70),2),
        'ttZjj':np.power((eff/0.70),4)*(0.2)+np.power((eff/0.70),2)*(0.8),
        'ttZZ':np.power((eff/0.70),6)*(0.2)+np.power((eff/0.70),2)*(0.8)
    }

    if lumi <= 500:
        xs = {  'ttHH': 0.78*0.34,
                'ttHjj': 329*0.58,
                'ttHZ': 1.2*0.58,
                'ttZjj': 428*0.7,
                'ttZZ': 1.5*0.7**2,
                'ttbbbb': 800,
                'tttt': 5.3,
                'tt': 834E3}

    else:
        xs = {  'ttHH': 0.78*1.11*1.07*0.34,
                'ttHjj': 329*1.13*1.07*0.58,
                'ttHZ': 1.2*1.11*0.7*0.58,
                'ttZjj': 428*1.20*0.7,
                'ttZZ': 1.5*1.20*0.7**2,
                'ttbbbb': 937,
                'tttt': 5.3*1.19*1.11,
                'tt': 834E3*1.11*1.06}

    cut0 = test_data['pred_0'] <= best_cut[0] # 0.36
    cut1 = test_data['pred_1'] <= best_cut[1] # 0.55
    cut2 = test_data['pred_2'] >= best_cut[2] # 0.33
    cut = cut0 & cut1 & cut2

    if bins == 1:
        signal_yields = []
        background_yields = []
        bkg_uncertainties = []
        signal_yield, background_yield = significance(cut, test_data, btag, xs, lumi)
        signal_yields.append(signal_yield)
        background_yields.append(background_yield)
        bkg_uncertainties.append((background_yield)**0.5)

    elif bins == 9:
        signal_yields = []
        background_yields = []
        bkg_uncertainties = []
        for i in [0, 1, 2]:
            for j in [4, 5, 6]:
                decay = test_data['Event_type'] == i
                bjets = test_data['nBjets'] == j
                if j == 6: bjets = test_data['nBjets'] >= j
                final_cut = cut & decay & bjets
                signal_yield, background_yield = significance(final_cut, test_data, btag, xs, lumi)
                signal_yields.append(signal_yield)
                background_yields.append(background_yield)
                bkg_uncertainties.append((background_yield)**0.5)

    print(signal_yields, background_yields)

    inference_model = pyhf.simplemodels.uncorrelated_background(
        signal=signal_yields, bkg=background_yields, bkg_uncertainty=bkg_uncertainties
    )
    data = [signal_yields[i] + background_yields[i] for i in range(len(signal_yields))] + inference_model.config.auxdata


    poi_vals = np.linspace(0, 100, 101)
    results = [list(map(float, pyhf.infer.hypotest(test_poi, data, inference_model,
                    par_bounds = [[0, max(poi_vals)]], test_stat="qtilde", return_expected_set=True)[1]))
        for test_poi in poi_vals
    ]

    results = np.array(results)

    limits = poi_vals[np.argmin(np.abs(results - 0.05), axis = 0)]
    print(limits)
    better_limits = []
    for i in range(len(limits)):
        new_poi_vals = np.linspace(limits[i] - 0.5, limits[i] + 0.5, num = 101)
        new_results = [list(map(float, pyhf.infer.hypotest(test_poi, data, inference_model,
                    par_bounds = [[0, max(new_poi_vals)]], test_stat="qtilde", return_expected_set=True)[1]))
            for test_poi in new_poi_vals
        ]
        new_results = np.array(new_results)
        new_limit = new_poi_vals[np.argmin(np.abs(new_results[:, i] - 0.05))]
        better_limits.append(new_limit)

    print(better_limits)
    return better_limits

settings = {1: [140, 0.70, 1],
            2: [500, 0.70, 1],
            3: [500, 0.85, 1],
            4: [3000, 0.70, 1],
            5: [3000, 0.70, 9],
            6: [3000, 0.85, 1],
            7: [3000, 0.85, 9]}

boxes = []
for key in settings.keys():
    boxes.append(limits(*settings[key]))

boxes = np.array(boxes)
#%%

fig, ax = plt.subplots(figsize = (5, 3))
for i in range(7):
    label = [None, None, None]
    if i == 0:
        label = [r'$\pm 2 \sigma$', r'$\pm 1 \sigma$', 'Expected limit']
    ax.barh(6.0 - i, boxes[i, 4] - boxes[i, 0], height = 1,
            left = boxes[i, 0], color = cdict['dark yellow'],
            align = 'center', label = label[0])
    ax.barh(6.0 - i, boxes[i, 3] - boxes[i, 1], height = 1, 
            left = boxes[i, 1], color = cdict['light turquoise'],
            align = 'center', label = label[1])
    ax.plot((boxes[i, 2], boxes[i, 2]), (6.5 - i, 5.5 - i), 
            'k--', lw = 1, label = label[2])
    # ax.broken_barh([(boxes[i, 0], boxes[i, 1] - boxes[i, 0]), 
    #                 (boxes[i, 3], boxes[i, 4] - boxes[i, 3])], 
    #                 (5.5 - i, 1), color = cdict['dark yellow'])
ax.set_yticks([0, 1, 2, 3, 4, 5, 6], 
              ['3000 fb$^{-1}$, 85% b-tag, 9 bins',
               '3000 fb$^{-1}$, 85% b-tag, 1 bin',
               '3000 fb$^{-1}$, 70% b-tag, 9 bins',
               '3000 fb$^{-1}$, 70% b-tag, 1 bin',
               '500 fb$^{-1}$, 85% b-tag, 1 bin',
               '500 fb$^{-1}$, 70% b-tag, 1 bin',
               '140 fb$^{-1}$, 70% b-tag, 1 bin'])
ax.set_xlim(0, 100)
ax.set_ylim(-0.5, 6.5)
ax.set_xlabel('$\mu$')
ax.legend()
fig.savefig('Limits.pdf', bbox_inches = 'tight')
plt.show()


#%%

lumi = 3000
eff = 0.85
btag = {
    'ttHH': np.power((eff/0.70),6),
    'ttHjj': np.power((eff/0.70),4),
    'ttbbbb': np.power((eff/0.70),6),
    'ttHZ': np.power((eff/0.70),4)*(0.85)+np.power((eff/0.70),6)*(0.15),
    'tttt': np.power((eff/0.70),4),
    'tt': np.power((eff/0.70),2),
    'ttZjj':np.power((eff/0.70),4)*(0.2)+np.power((eff/0.70),2)*(0.8),
    'ttZZ':np.power((eff/0.70),6)*(0.2)+np.power((eff/0.70),2)*(0.8)
}

xs = {  'ttHH': 0.78*1.11*1.07*0.34,
        'ttHjj': 329*1.13*1.07*0.58,
        'ttHZ': 1.2*1.11*0.7*0.58,
        'ttZjj': 428*1.20*0.7,
        'ttZZ': 1.5*1.20*0.7**2,
        'ttbbbb': 937,
        'tttt': 5.3*1.19*1.11,
        'tt': 834E3*1.11*1.06}

cut0 = test_data['pred_0'] <= best_cut[0] # 0.36
cut1 = test_data['pred_1'] <= best_cut[1] # 0.55
cut2 = test_data['pred_2'] >= best_cut[2] # 0.33
cut = cut0 & cut1 & cut2

signal_yields = []
background_yields = []
bkg_uncertainties = []
for i in [0, 1, 2]:
    for j in [4, 5, 6]:
        decay = test_data['Event_type'] == i
        bjets = test_data['nBjets'] == j
        if j == 6: bjets = test_data['nBjets'] >= j
        final_cut = cut & decay & bjets
        signal_yield, background_yield = significance(final_cut, test_data, btag, xs, lumi)
        signal_yields.append(signal_yield)
        background_yields.append(background_yield)
        bkg_uncertainties.append((background_yield)**0.5)

print(signal_yields, background_yields)

inference_model = pyhf.simplemodels.uncorrelated_background(
    signal=signal_yields, bkg=background_yields, bkg_uncertainty=bkg_uncertainties
)
data = [signal_yields[i] + background_yields[i] for i in range(len(signal_yields))] + inference_model.config.auxdata


poi_vals = np.linspace(0, 10, 101)
results = [list(map(float, pyhf.infer.hypotest(test_poi, data, inference_model,
                par_bounds = [[0, max(poi_vals)]], test_stat="qtilde", return_expected_set=True)[1]))
    for test_poi in poi_vals
]

results = np.array(results)

linestyle = [':', ':', '--', ':', ':']
colours = [cdict['dark yellow'], cdict['light turquoise'], cdict['dark blue']]

fig, ax = plt.subplots(figsize = (7, 6))

for i in range(results.shape[1]):
    label = None
    if i == 2: label = r'CL$_{s, exp}$'
    ax.plot(poi_vals, results[:, i], color = colours[2], ls = linestyle[i], label = label)

ax.fill_between(poi_vals, results[:, 0], results[:, -1], color = colours[0], label = r'$\pm 2 \sigma$ CL$_{s, exp}$')

ax.fill_between(poi_vals, results[:, 1], results[:, -2], color = colours[1], label = r'$\pm 1 \sigma$ CL$_{s, exp}$')

ax.axhline(0.05, color = cdict['dark pink'], label = r'$\alpha = 0.05$')

# ax.annotate(r'95% CL upper limit ($\pm 1 \sigma$): $6.5^{+2.5}_{-1.8}$',
#             xy = (0.3, 0.98), xycoords = 'axes fraction',
#             horizontalalignment = 'left', verticalalignment = 'top')

ax.annotate(r'3000 fb$^{-1}$, 14 TeV', xy = (1, 1.01), xycoords = 'axes fraction', 
                horizontalalignment = 'right', verticalalignment = 'bottom')

ax.set_xlim(0, max(poi_vals))
ax.set_ylim(0, 1)

ax.set_xlabel(r'$\mu$')
ax.set_ylabel(r'CL$_s$')

ax.legend()
plt.savefig('CLs HL 9bin btag85 zoomed.pdf')
plt.show()


# %%

xs_13 = {'ttHH': 0.78*0.34,
        'ttHZ': 1.2*0.58,
        'ttZZ': 1.5*0.7**2,
        'tttt': 5.3,
        'ttHjj': 329*0.58,
        'ttZjj': 428*0.7,
        'ttbbbb': 800,
        'tt': 834E3}

xs_14 = {'ttHH': 0.78*1.11*1.07,
        'ttHZ': 1.2*1.11*1.07,
        'ttZZ': 1.5*1.20,
        'tttt': 5.3*1.19,
        'ttbbbb': 150,
        'ttHjj': 329*1.13*1.07,
        'ttZjj': 428*1.20,
        'tt': 834E3*1.11*1.06}

colours = [cdict['dark pink'], cdict['dark blue'], cdict['dark turquoise'], cdict['light turquoise'], 
           cdict['dark yellow'], cdict['lilac'], cdict['light green'], cdict['dark green']]
labels = ['ttZZ', 'ttHZ', 'ttZjj', 'tttt', 'ttHjj', 'ttbbbb', 'ttHH']

fig, ax = plt.subplots(figsize = (6, 2), constrained_layout = True)
y_pos = np.arange(len(xs))
x_err = np.vstack((np.zeros(len(xs)), np.array(list(xs_14.values())) - np.array(list(xs_13.values()))))
hbars = ax.barh(y_pos, xs_14.values(), color = colours, alpha = 0.8, lw = 2, edgecolor = colours)
ax.bar_label(hbars, xs_14.keys(), padding = 5)
ax.set_xscale('log')
ax.set_yticks([])
ax.invert_yaxis()
ax.set_xlabel('cross-section (fb) (14 TeV)')
plt.savefig('Cross sections.pdf')
plt.show()


# %%

colours = [cdict['dark blue'], cdict['dark turquoise'], cdict['light turquoise'], 
           cdict['dark yellow'], cdict['lilac'], cdict['light green'], cdict['dark green']]
labels = ['ttHZ', 'ttZZ', 'tttt', 'ttbbbb', 'ttHjj', 'ttZjj', 'tt']

cut0 = test_data['pred_0'] <= best_cut[0] # 0.36
cut1 = test_data['pred_1'] <= best_cut[1] # 0.55
cut2 = test_data['pred_2'] >= best_cut[2] # 0.33
cut = cut0 & cut1 & cut2
bkg_components = []
signal = test_data[cut]
for label in labels:
    scaling = len(test_data.loc[label])/len(sub_events.loc[label])
    bkg_yield = lumi*xs[label]*len(signal.loc[label])/(n_events[label]*scaling)
    bkg_components.append(bkg_yield)

fig, ax = plt.subplots(figsize = (4, 4))
ax.pie(np.log10(bkg_components), labels = labels, colors = colours)
ax.axis('equal')
#plt.savefig('Bkg decomp.pdf')
plt.show()

# %%

colours = [cdict['dark blue'], cdict['dark turquoise'], cdict['light turquoise'], 
           cdict['dark yellow'], cdict['lilac'], cdict['light green'],# cdict['dark green'],
           cdict['dark pink']]
labels = ['ttHZ', 'ttZZ', 'tttt', 'ttHjj', 'ttZjj', 'ttbbbb', 'ttHH']
bins = np.linspace(0, 1, num = 21)

with plt.xkcd():
    n = len(labels)
    bottom = 0
    for i in range(n):
        mask = test_data.index.get_level_values(0) == labels[i]
        heights, edges = np.histogram(np.fmin(test_data.loc[mask, 'pred_2'], max(bins)), 
                    bins = bins, density = False)
        scaling = len(test_data.loc[labels[i]])/len(sub_events.loc[labels[i]])
        heights = lumi*xs[labels[i]]/(n_events[labels[i]]*scaling) * heights
        if i < n - 1: bottom += heights

    fig, ax = plt.subplots(figsize = (7, 7))
    ax.stairs(bottom, edges, color = cdict['dark blue'], label = 'Background', lw = 2)
    ax.stairs(heights, edges, color = cdict['dark pink'], label = 'Signal', lw = 2)

    ax.set_xlim(min(bins), max(bins))

    ax.set_xlabel('Prediction')
    ax.legend()
    ax.set_yscale('log')
    ax.set_yticklabels([])
    ax.set_ylabel('Event yields')
    ax.set_xlabel('Prediction')

    plt.show()



# %%
