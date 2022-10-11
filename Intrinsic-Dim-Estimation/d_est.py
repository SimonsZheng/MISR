# -*- coding: utf-8 -*-
# @Time    : 2022/4/19 14:38
# @Author  : SimonZHENG
# @Email   : zhengsiming2016@163.com
# @File    : d_est.py
# @Software: PyCharm
from model_nets import *
from scipy.stats import norm

# arguments setting
parser = argparse.ArgumentParser()
parser.add_argument('--batchSz', type=int, default=512, help='mini batch size')
parser.add_argument('--nEpochs', type=int, default=2000, help='the number of outer loop')
parser.add_argument('--cuda_device', type=int, default=0, help='choose cuda device')
parser.add_argument('--no-cuda', action='store_true', help='if TRUE, cuda will not be used')
parser.add_argument('--save', help='path to save results')
parser.add_argument('--seed', type=int, default=42, help='random seed')
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--eta', type=float, default=0.1)
args = parser.parse_args()

args.cuda = not args.no_cuda and torch.cuda.is_available()
print(args.cuda)
device = torch.device("cuda" if args.cuda else "cpu")
args.save = args.save or 'sim/toy_reg'
setup_seed(args.seed, args.cuda)

if not os.path.exists(args.save):
    os.makedirs(args.save, exist_ok=True)

device = torch.device("cuda" if args.cuda else "cpu")
kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}
train_n = 4000  # Training sample size
validation_n = 2000  # Validation sample size
test_n = 2000  # Testing sample size
n_sample = train_n + validation_n + test_n
Repeat_num = 100
d_array = np.ndarray(shape=(args.nEpochs, 1), dtype=float)
timer = Timer()
for R in range(Repeat_num):
    # Data generating
    X = np.random.randn(n_sample, 10)
    eps1 = np.random.randn(n_sample, 1)
    y = norm.cdf(X[:, 0]).reshape((-1, 1)) + norm.cdf(X[:, 1]).reshape((-1, 1)) * eps1
    X = X.astype(np.float32)
    y = y.astype(np.float32)

    x_dim = X.shape[1]
    y_dim = y.shape[1]
    UD = x_dim
    LD = 1

    est_d = UD

    args.latent_dim = UD
    u_dim = args.latent_dim

    # The splitting number of cross-validation
    kf = KFold(n_splits=4)
    ini_num = 1
    tup1 = ()
    # Dichotomy-based cross-validation intrinsic dimension estimation
    for f, (train_val_index, test_index) in enumerate(kf.split(X)):
        best_test_mi = None
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

        validation_dat = TensorDataset(torch.from_numpy(y_validation), torch.from_numpy(X_validation).float(),
                                       U_validation)
        validationLoader = DataLoader(validation_dat, batch_size=len(validation_dat), shuffle=False)

        test_dat = TensorDataset(torch.from_numpy(y_test), torch.from_numpy(X_test).float(), U_test)
        testLoader = DataLoader(test_dat, batch_size=args.batchSz, shuffle=False)
        testLoader_cor = DataLoader(test_dat, batch_size=len(test_dat), shuffle=False)

        best_mi = None
        for k in range(ini_num):
            R_width_vec = [x_dim, 128, u_dim]
            R_net = Representer(x_dim, u_dim, R_width_vec)
            D_width_vec = [u_dim + y_dim, 64]
            D_net = MIDiscriminator(u_dim, y_dim, D_width_vec)
            Q_width_vec = [u_dim, 64]
            Q_net = PushDiscriminator(u_dim, Q_width_vec)

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
                mi_loss = test(R_net, D_net, validationLoader, device)
                early_stopping(mi_loss, epoch)
                if early_stopping.dec:
                    test_mi = -1 * test(R_net, D_net, testLoader_cor, device) + 1
                if early_stopping.early_stop:
                    break
            # Save the best mi estimate
            if best_mi is None or -1 * early_stopping.best_val_score + 1 > best_mi:
                best_mi = -1 * early_stopping.best_val_score + 1
                best_test_mi = test_mi
            del R_net
            del D_net
            del Q_net
            torch.cuda.empty_cache()
        tup1 += (best_test_mi,)
        torch.cuda.empty_cache()
    temp = np.mean(tup1)
    print('R:', R, 'd:', args.latent_dim, 'MIDR MI: {:.2f}({:.2f})\n'.format(temp, np.var(tup1) ** 0.5))
    while UD != LD:
        args.latent_dim = (UD + LD) // 2
        u_dim = args.latent_dim
        kf = KFold(n_splits=4)
        ini_num = 1
        tup1 = ()
        for f, (train_val_index, test_index) in enumerate(kf.split(X)):
            best_test_mi = None
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

            validation_dat = TensorDataset(torch.from_numpy(y_validation), torch.from_numpy(X_validation).float(),
                                           U_validation)
            validationLoader = DataLoader(validation_dat, batch_size=len(validation_dat), shuffle=False)

            test_dat = TensorDataset(torch.from_numpy(y_test), torch.from_numpy(X_test).float(), U_test)
            testLoader = DataLoader(test_dat, batch_size=args.batchSz, shuffle=False)
            testLoader_cor = DataLoader(test_dat, batch_size=len(test_dat), shuffle=False)

            best_mi = None
            for k in range(ini_num):
                R_width_vec = [x_dim, 128, u_dim]
                R_net = Representer(x_dim, u_dim, R_width_vec)
                D_width_vec = [u_dim + y_dim, 64]
                D_net = MIDiscriminator(u_dim, y_dim, D_width_vec)
                Q_width_vec = [u_dim, 64]
                Q_net = PushDiscriminator(u_dim, Q_width_vec)

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
                    mi_loss = test(R_net, D_net, validationLoader, device)
                    early_stopping(mi_loss, epoch)
                    if early_stopping.dec:
                        test_mi = -1 * test(R_net, D_net, testLoader_cor, device) + 1
                    if early_stopping.early_stop:
                        break
                # Save the best mi estimate
                if best_mi is None or -1 * early_stopping.best_val_score + 1 > best_mi:
                    best_mi = -1 * early_stopping.best_val_score + 1
                    best_test_mi = test_mi
                del R_net
                del D_net
                del Q_net
                torch.cuda.empty_cache()
            tup1 += (best_test_mi,)
            torch.cuda.empty_cache()
        com_val = np.mean(tup1)
        print('R:', R, 'd:', args.latent_dim, 'MIDR MI: {:.2f}({:.2f})\n'.format(com_val, np.var(tup1) ** 0.5))
        if np.abs(com_val-temp)/temp <= args.eta:
            UD = args.latent_dim
            est_d = args.latent_dim
            temp = com_val
        else:
            LD = args.latent_dim + 1
    d_array[R] = est_d
    print('Estimated d:', est_d)
np.savetxt('./Result.csv', d_array, delimiter=',')
print("done!")
f'{timer.stop():.2f} sec'
