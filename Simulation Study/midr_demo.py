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
parser.add_argument('--latent_dim', type=int, default=2, help='the dimension of latent space')
parser.add_argument('--nEpochs', type=int, default=1000, help='the number of outer loop')
parser.add_argument('--cuda_device', type=int, default=0, help='choose cuda device')
parser.add_argument('--no-cuda', action='store_true', help='if TRUE, cuda will not be used')
parser.add_argument('--save', help='path to save results')
parser.add_argument('--seed', type=int, default=1, help='random seed')
parser.add_argument('--model', type=int, default=2, help='1: Model A; 2: Model B; 3: Model C')
parser.add_argument('--scenario', type=int, default=2, help='1: scenario 1 ; 2: scenario 2; '
                                                            '3: scenario 3; 4: scenario 4')
parser.add_argument('--lr', type=float, default=1e-3)
args = parser.parse_args()
print(args.model, args.scenario)

args.cuda = not args.no_cuda and torch.cuda.is_available()
print(args.cuda)
device = torch.device("cuda" if args.cuda else "cpu")
args.save = args.save or 'sim/toy_reg'
setup_seed(args.seed, args.cuda)

if not os.path.exists(args.save):
    os.makedirs(args.save, exist_ok=True)

device = torch.device("cuda" if args.cuda else "cpu")
kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}
# Training sample size
train_n = 4000
# Validation sample size
validation_n = 1000
# Testing sample size
test_n = 1000
n_sample = train_n + validation_n + test_n

p = 10
# Scenario type
if args.scenario == 1:
    X = np.random.uniform(size=(n_sample, p), low=-2, high=2)
if args.scenario == 2:
    X = np.random.randn(n_sample, p)
elif args.scenario == 3:
    distributions = [
        {"type": np.random.normal, "kwargs": {"loc": 2, "scale": 1}},
        {"type": np.random.uniform, "kwargs": {"low": -2, "high": 2}},
        {"type": np.random.normal, "kwargs": {"loc": -2, "scale": 1}},
    ]
    coefficients = np.array([1.0, 2, 1])
    coefficients /= coefficients.sum()
    num_distr = len(distributions)
    data = np.zeros((n_sample, num_distr, p))
    for idx, distr in enumerate(distributions):
        data[:, idx] = distr["type"](size=(n_sample, p), **distr["kwargs"])
    random_idx = np.random.choice(np.arange(num_distr), size=(n_sample,), p=coefficients)
    X = data[np.arange(n_sample), random_idx]
elif args.scenario == 4:
    mean = np.zeros(p)
    cov = 0.5 * np.eye(p) + 0.5 * np.multiply(np.ones((p, 1)), np.ones((1, p)))
    X = np.random.multivariate_normal(mean, cov, n_sample)

# Model type
if args.model == 1:
    truth = 0.5 * X[:, 0] + X[:, 3]
elif args.model == 2:
    truth = (X[:, 0] ** 2 + X[:, 1] ** 2) ** 0.5 * np.log((X[:, 0] ** 2 + X[:, 1] ** 2) ** 0.5)
elif args.model == 3:
    truth = (X[:, 0] + X[:, 1]) ** 2 / (1 + np.exp(X[:, 1]))
elif args.model == 4:
    truth = np.sin(np.pi * (X[:, 0] + X[:, 1]) / 10.) + X[:, 1] ** 2

eps = np.random.randn(n_sample, 1)
sigma = 0.25
y = truth.reshape(n_sample, 1) + sigma * eps
X = X.astype(np.float32)
y = y.astype(np.float32)

# The splitting number of cross-validation
kf = KFold(n_splits=6)
tup1 = ()
tup2 = ()

x_dim = X.shape[1]
y_dim = y.shape[1]
u_dim = args.latent_dim

# The number of the repeated weight initializations.
ini_num = 10

for f, (train_val_index, test_index) in enumerate(kf.split(X)):
    X_train_val, X_test = X[train_val_index], X[test_index]
    y_train_val, y_test = y[train_val_index], y[test_index]
    ind = np.arange(train_n + validation_n)
    np.random.shuffle(ind)
    X_train, X_validation = X_train_val[ind[:train_n]], X_train_val[ind[train_n:]]
    y_train, y_validation = y_train_val[ind[:train_n]], y_train_val[ind[train_n:]]
    U_train = torch.rand(train_n, args.latent_dim)
    U_validation = torch.rand(validation_n, args.latent_dim)
    U_test = torch.rand(test_n, args.latent_dim)

    train_dat = TensorDataset(torch.from_numpy(y_train), torch.from_numpy(X_train).float(), U_train)
    trainLoader = DataLoader(train_dat, batch_size=args.batchSz, shuffle=True)

    train_val_dat = TensorDataset(torch.from_numpy(y_train_val), torch.from_numpy(X_train_val).float(),
                                  torch.cat((U_train, U_validation), 0))
    train_valLoader = DataLoader(train_val_dat, batch_size=args.batchSz, shuffle=True)

    validation_dat = TensorDataset(torch.from_numpy(y_validation), torch.from_numpy(X_validation).float(),
                                   U_validation)
    validationLoader = DataLoader(validation_dat, batch_size=len(validation_dat), shuffle=False)

    test_dat = TensorDataset(torch.from_numpy(y_test), torch.from_numpy(X_test).float(), U_test)
    testLoader = DataLoader(test_dat, batch_size=args.batchSz, shuffle=False)
    testLoader_cor = DataLoader(test_dat, batch_size=len(test_dat), shuffle=False)

    best_dc = None
    best_R_net = Representer(x_dim, u_dim)
    if args.cuda:
        best_R_net = best_R_net.cuda()
    for k in range(ini_num):
        R_net = Representer(x_dim, u_dim)
        D_net = MIDiscriminator(u_dim, y_dim)
        Q_net = PushDiscriminator(u_dim)

        R_net.apply(weight_init)
        D_net.apply(weight_init)
        Q_net.apply(weight_init)

        if args.cuda:
            R_net = R_net.cuda()
            D_net = D_net.cuda()
            Q_net = Q_net.cuda()

        # default weight decay parameter
        wd = 1e-4
        # user-selected learning rate
        mylr = args.lr
        optimizer_R = optim.Adam(R_net.parameters(), lr=mylr, weight_decay=wd)
        optimizer_D = optim.Adam(D_net.parameters(), lr=mylr, weight_decay=wd)
        optimizer_Q = optim.Adam(Q_net.parameters(), lr=mylr, weight_decay=wd)

        patience = 200
        early_stopping = EarlyStopping(patience, verbose=True)
        for epoch in range(1, args.nEpochs + 1):
            train(trainLoader, R_net, D_net, Q_net, optimizer_R, optimizer_D, optimizer_Q, device)
            dc_loss = test(R_net, validationLoader, device)
            early_stopping(args.save, -1 * dc_loss, R_net, epoch)
            if early_stopping.early_stop:
                print("Early stopping at Epoch:", early_stopping.epoch)  # stop training
                break
        print('fold', f, 'Init', k, '. When Stopping, DC is', -1 * early_stopping.best_score, 'and Iteration is',
              early_stopping.epoch)
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
    # The empirical DC calculation
    y_test_t, X_test_t, _ = next(iter(testLoader_cor))
    X_test_t, y_test_t = X_test_t.to(device), y_test_t.to(device)
    lant_t = best_R_net(X_test_t)
    tup1 += (cor(lant_t, y_test_t, X_test_t.shape[0], device).item(),)
    torch.cuda.empty_cache()
    X_train_tf, y_train_tf = npLoader(train_valLoader, best_R_net, device)
    X_test_tf, y_test_tf = npLoader(testLoader, best_R_net, device)

    # Linear validation protocol
    scaler = StandardScaler()
    scaler.fit(X_train_tf)
    X_train_tf = scaler.transform(X_train_tf)
    X_test_tf = scaler.transform(X_test_tf)
    reg = LinearRegression().fit(X_train_tf, y_train_tf)
    y_pred = reg.predict(X_test_tf)
    rms = sqrt(mean_squared_error(y_test_tf, y_pred))
    tup2 += (rms,)
print("done! Results are")
print('MSRL DC: {:.2f}({:.2f})\n'.format(np.mean(tup1), np.var(tup1) ** 0.5))
print('MSRL MSE: {:.2f}({:.2f})\n'.format(np.mean(tup2), np.var(tup2) ** 0.5))
data_path = args.save+'/SimData'
SavedData = np.concatenate((X, y), axis=1)
if not os.path.exists(data_path):
    os.makedirs(data_path, exist_ok=True)
np.savetxt('./'+data_path+'/Data_S('+str(args.scenario)+')_M('+str(args.model)+').csv', SavedData, delimiter=',')
ResultData = [tup1, tup2]
ResultData = np.array(ResultData)
np.savetxt('./'+data_path+'/Result_S('+str(args.scenario)+')_M('+str(args.model)+').csv', ResultData, delimiter=',')
