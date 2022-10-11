# -*- coding: utf-8 -*-
# @Time    : 2022/3/27 2:34
# @Author  : SimonZHENG
# @Email   : zhengsiming2016@163.com
# @File    : midr_demo.py
# @Software: PyCharm
from model_nets import *

# arguments setting
parser = argparse.ArgumentParser()
parser.add_argument('--batchSz', type=int, default=512, help='mini batch size')
parser.add_argument('--latent_dim', type=int, default=5, help='the dimension of latent space')
parser.add_argument('--nEpochs', type=int, default=2000, help='the number of outer loop')
parser.add_argument('--cuda_device', type=int, default=0, help='choose cuda device')
parser.add_argument('--no-cuda', action='store_true', help='if TRUE, cuda will not be used')
parser.add_argument('--save', help='path to save results')
parser.add_argument('--seed', type=int, default=123, help='random seed')
parser.add_argument('--lr', type=float, default=1e-4)
args = parser.parse_args()

args.cuda = not args.no_cuda and torch.cuda.is_available()
print(args.latent_dim, args.cuda)
device = torch.device("cuda" if args.cuda else "cpu")
args.save = args.save or 'sim/toy_reg'
setup_seed(args.seed, args.cuda)

if not os.path.exists(args.save):
    os.makedirs(args.save, exist_ok=True)

kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}

mydata = np.loadtxt('./superconduct/train.csv', delimiter=',', skiprows=1)
X = mydata[:, :-1]
y = mydata[:, -1].reshape((-1, 1))
X = X.astype(np.float32)
y = y.astype(np.float32)
n_sample = X.shape[0]
validation_n = 3000 # Validation sample size
test_n = 4200 # Testing sample size
train_n = n_sample - (validation_n + test_n) # Training sample size

x_dim = X.shape[1]
y_dim = y.shape[1]
u_dim = args.latent_dim

# The number of the repeated weight initializations.
ini_num = 10

ind = np.arange(n_sample)
np.random.shuffle(ind)
print(ind[:5])
X_train_val, X_test = X[ind[:train_n + validation_n],], X[ind[train_n + validation_n:],]
y_train_val, y_test = y[ind[:train_n + validation_n],], y[ind[train_n + validation_n:],]
ind = np.arange(train_n + validation_n)
np.random.shuffle(ind)
X_train, X_validation = X_train_val[ind[0:train_n]], X_train_val[ind[train_n:]]
y_train, y_validation = y_train_val[ind[0:train_n]], y_train_val[ind[train_n:]]

pre_scaler = StandardScaler()
pre_scaler.fit(X_train)
X_train = pre_scaler.transform(X_train)
X_validation = pre_scaler.transform(X_validation)
X_test = pre_scaler.transform(X_test)

U_train = torch.rand(train_n, args.latent_dim)
U_validation = torch.rand(validation_n, args.latent_dim)
U_test = torch.rand(test_n, args.latent_dim)

train_dat = TensorDataset(torch.from_numpy(y_train), torch.from_numpy(X_train).float(), U_train)
trainLoader = DataLoader(train_dat, batch_size=args.batchSz, shuffle=True)

validation_dat = TensorDataset(torch.from_numpy(y_validation), torch.from_numpy(X_validation).float(),
                               U_validation)
validationLoader = DataLoader(validation_dat, batch_size=len(validation_dat), shuffle=False)

test_dat = TensorDataset(torch.from_numpy(y_test), torch.from_numpy(X_test).float(), U_test)
testLoader = DataLoader(test_dat, batch_size=args.batchSz, shuffle=False)
testLoader_cor = DataLoader(test_dat, batch_size=len(test_dat), shuffle=False)

best_dc = None
R_width_vec = [x_dim, 128, 128, 128, u_dim]
best_R_net = Representer(x_dim, u_dim, R_width_vec)
if args.cuda:
    best_R_net = best_R_net.cuda()
for k in range(ini_num):
    R_net = Representer(x_dim, u_dim, R_width_vec)
    D_width_vec = [u_dim + y_dim, 64, 64]
    D_net = MIDiscriminator(u_dim, y_dim, D_width_vec)
    Q_width_vec = [u_dim, 64, 64]
    Q_net = PushDiscriminator(u_dim, Q_width_vec)

    R_net.apply(weight_init)
    D_net.apply(weight_init)
    Q_net.apply(weight_init)

    if args.cuda:
        R_net = R_net.cuda()
        D_net = D_net.cuda()
        Q_net = Q_net.cuda()

    # default weight decay parameter
    wd = 1e-3
    # user-selected learning rate
    mylr = args.lr
    optimizer_R = optim.Adam(R_net.parameters(), lr=mylr, weight_decay=wd)
    optimizer_D = optim.Adam(D_net.parameters(), lr=mylr, weight_decay=wd)
    optimizer_Q = optim.Adam(Q_net.parameters(), lr=mylr, weight_decay=wd)

    patience = 400
    early_stopping = EarlyStopping(patience, verbose=True)
    for epoch in range(1, args.nEpochs + 1):
        print('Epoch:', epoch)
        train(trainLoader, R_net, D_net, Q_net, optimizer_R, optimizer_D, optimizer_Q, device)
        dCor_loss = test(R_net, validationLoader, device)
        early_stopping(args.save, -1 * dCor_loss, R_net, epoch)
        if early_stopping.early_stop:
            print("Early stopping at Epoch:", epoch)
            break
    print('Init', k, '. When Stopping, DC is', -1 * early_stopping.best_score, 'and Iteration is', early_stopping.epoch)
    # Save the best representer network
    if best_dc is None or -1 * early_stopping.best_score > best_dc:
        model_path = os.path.join(args.save, 'R.pt')
        R_net.load_state_dict(torch.load(model_path))
        best_dc = -1 * early_stopping.best_score
        torch.save(R_net.state_dict(), os.path.join(args.save, 'best_R.pt'))
    del R_net
    del D_net
    del Q_net
    torch.cuda.empty_cache()
best_model_path = os.path.join(args.save, 'best_R.pt')
best_R_net.load_state_dict(torch.load(best_model_path))
best_R_net.eval()

_, X_test_t, _ = next(iter(testLoader_cor))
X_test_t = X_test_t.to(device)
lant_t = best_R_net(X_test_t).cpu().detach().numpy()
saved_data = np.concatenate((lant_t, y_test), axis=1)
np.savetxt('MIRL.csv', saved_data, delimiter=',')

