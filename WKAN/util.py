from .Module import *
import torch
from .Train import *
from .DataPrep import *
import pandas as pd
from typing import List, Callable, Union, Any, TypeVar, Tuple
Tensor = TypeVar('torch.tensor')
import pickle

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def Fit_cord_DNN (data_train, location_data = None, hidden_dims = [30, 25, 15], num_epochs_max = 500, path = "", filename = "PreOrg_Mousesc", batch_size = 4, num_workers = 0, number_error_try = 15, initial_learning_rate = 0.001, seednum = 2021):
    #
    random.seed(seednum)
    torch.manual_seed(seednum)
    np.random.seed(seednum)
    g = torch.Generator()
    g.manual_seed(seednum)
    #
    if location_data is None:
        location_data = data_train.obs
    #
    traindata = (data_train.X.A if issparse(data_train.X) else data_train.X)
    tdatax = np.expand_dims(traindata, axis = 0)
    tdata_rs = np.swapaxes(tdatax, 1, 2)
    DataTra = wrap_gene_location(tdata_rs, location_data)
    t_loader= torch.utils.data.DataLoader(DataTra, batch_size=batch_size, num_workers = num_workers, shuffle = True, worker_init_fn=seed_worker, generator=g)
    # model = DNN( in_channels = DataTra[1][0].shape[0], hidden_dims = hidden_dims) # [100,50,25] )
    if len(hidden_dims) == 3:
        model = DNN3_cord(in_channels=DataTra[1][0].shape[0], hidden_dims=hidden_dims)
    if len(hidden_dims) == 4:
        model = DNN4_cord(in_channels=DataTra[1][0].shape[0], hidden_dims=hidden_dims)
    if len(hidden_dims) == 5:
        model = DNN5_cord(in_channels=DataTra[1][0].shape[0], hidden_dims=hidden_dims)

    model = model.float()
    model = model.to(device)
    #
    CoOrg = TrainerExe()
    CoOrg.train(model = model, train_loader = t_loader, num_epochs= num_epochs_max, RCcountMax = number_error_try, learning_rate = initial_learning_rate)
    #
    try:
        os.makedirs("{path}".format(path = path))
    except FileExistsError:
        print("Folder already exists")
    filename3 = "{path}/{filename}.obj".format(path = path, filename = filename)
    filehandler2 = open(filename3, 'wb')
    pickle.dump(model, filehandler2)
    return model

def Fit_cord_KAN ( data_train,model_config = "paper", location_data = None, hidden_dims = [30, 25, 15],
                  num_epochs_max = 500, path = "", filename = "PreOrg_Mousesc",
                  batch_size = 4, num_workers = 0, number_error_try = 15, initial_learning_rate = 0.001, seednum = 2021):
    #
    random.seed(seednum)
    torch.manual_seed(seednum)
    np.random.seed(seednum)
    g = torch.Generator()
    g.manual_seed(seednum)
    #
    if location_data is None:
        location_data = data_train.obs
    #
    traindata = (data_train.X.A if issparse(data_train.X) else data_train.X)
    tdatax = np.expand_dims(traindata, axis = 0)
    tdata_rs = np.swapaxes(tdatax, 1, 2)
    DataTra = wrap_gene_location(tdata_rs, location_data)
    t_loader= torch.utils.data.DataLoader(DataTra, batch_size=batch_size, num_workers = num_workers, shuffle = True, worker_init_fn=seed_worker, generator=g)

    if len(hidden_dims) == 3:
        if model_config == "original":
            model = KAN3_cord(in_channels=DataTra[1][0].shape[0], hidden_dims=hidden_dims)
        else:
            model = WKAN3_cord(in_channels=DataTra[1][0].shape[0], hidden_dims=hidden_dims)

    if len(hidden_dims) == 4:
        if model_config == "original":
            model = KAN4_cord(in_channels=DataTra[1][0].shape[0], hidden_dims=hidden_dims)
        else:
            model = WKAN4_cord(in_channels=DataTra[1][0].shape[0], hidden_dims=hidden_dims)

    if len(hidden_dims) == 5:
        if model_config == "original":
            model = KAN5_cord(in_channels=DataTra[1][0].shape[0], hidden_dims=hidden_dims)
        if model_config == "modified":
            model = Wixos5_cord(in_channels=DataTra[1][0].shape[0], hidden_dims=hidden_dims)
        else:
            model = WKAN5_cord(in_channels=DataTra[1][0].shape[0], hidden_dims=hidden_dims)


    # model = KAN( in_channels = DataTra[1][0].shape[0], hidden_dims = hidden_dims) # [100,50,25] )

    model = model.float()
    model = model.to(device)
    #
    CoOrg = TrainerExe()
    CoOrg.train(model = model, train_loader = t_loader, num_epochs= num_epochs_max, RCcountMax = number_error_try, learning_rate = initial_learning_rate)
    #
    try:
        os.makedirs("{path}".format(path = path))
    except FileExistsError:
        print("Folder already exists")
    filename3 = "{path}/{filename}.obj".format(path = path, filename = filename)
    filehandler2 = open(filename3, 'wb')
    pickle.dump(model, filehandler2)
    return model

def Fit_cord (data_train, location_data = None, hidden_dims = [30, 25, 15], num_epochs_max = 500, path = "", filename = "PreOrg_Mousesc", batch_size = 4, num_workers = 0, number_error_try = 15, initial_learning_rate = 0.001, seednum = 2021):
    #
    random.seed(seednum)
    torch.manual_seed(seednum)
    np.random.seed(seednum)
    g = torch.Generator()
    g.manual_seed(seednum)
    #
    if location_data is None:
        location_data = data_train.obs
    #
    traindata = (data_train.X.A if issparse(data_train.X) else data_train.X)
    tdatax = np.expand_dims(traindata, axis = 0)
    tdata_rs = np.swapaxes(tdatax, 1, 2)
    DataTra = wrap_gene_location(tdata_rs, location_data)
    t_loader= torch.utils.data.DataLoader(DataTra, batch_size=batch_size, num_workers = num_workers, shuffle = True, worker_init_fn=seed_worker, generator=g)
    # Create Deep Neural Network for Coordinate Regression
    if len(hidden_dims) == 5:#原来是5，DNN5
        # DNNmodel = DNN5( in_channels = DataTra[1][0].shape[0], hidden_dims = hidden_dims) # [100,50,25] )
        DNNmodel = KANDNN( in_channels = DataTra[1][0].shape[0], hidden_dims = hidden_dims) # [100,50,25] )
    else:  #ADD KAN layer ↓
        DNNmodel = KANDNN( in_channels = DataTra[1][0].shape[0], hidden_dims = hidden_dims) # [100,50,25] )
    DNNmodel = DNNmodel.float()
    #
    CoOrg = TrainerExe()
    CoOrg.train(model = DNNmodel, train_loader = t_loader, num_epochs= num_epochs_max, RCcountMax = number_error_try, learning_rate = initial_learning_rate)
    #
    try:
        os.makedirs("{path}".format(path = path))
    except FileExistsError:
        print("Folder already exists")
    filename3 = "{path}/{filename}.obj".format(path = path, filename = filename) #"../output/CeLEry/Mousesc/PreOrg_Mousesc.obj"
    filehandler2 = open(filename3, 'wb')
    pickle.dump(DNNmodel, filehandler2)
    return DNNmodel


def Fit_layer (data_train, layer_weights, layer_data = None, layerkey = "layer", hidden_dims = [10, 5, 2], num_epochs_max = 500, path = "", filename = "PreOrg_layersc", batch_size = 8, num_workers = 0, number_error_try = 15, initial_learning_rate = 0.001, seednum = 2021):
    #
    random.seed(seednum)
    torch.manual_seed(seednum)
    np.random.seed(seednum)
    g = torch.Generator()
    g.manual_seed(seednum)
    #
    if layer_data is None:
        layer_data = data_train.obs
    #
    layer_weights = torch.tensor(layer_weights.to_numpy())
    traindata = (data_train.X.A if issparse(data_train.X) else data_train.X)
    tdatax = np.expand_dims(traindata, axis = 0)
    tdata_rs = np.swapaxes(tdatax, 1, 2)
    DataTra = wrap_gene_layer(tdata_rs, layer_data, layerkey)
    t_loader= torch.utils.data.DataLoader(DataTra, batch_size = batch_size, num_workers = 0, shuffle = True, worker_init_fn=seed_worker, generator=g)
    # Create Deep Neural Network for Coordinate Regression
    DNNmodel = KAN_LIBD( in_channels = DataTra[1][0].shape[0], num_classes = layer_weights.shape[0], hidden_dims = hidden_dims, importance_weights = layer_weights) # [100,50,25] )
    # DNNmodel = DNNordinal( in_channels = DataTra[1][0].shape[0], num_classes = layer_weights.shape[0], hidden_dims = hidden_dims, importance_weights = layer_weights) # [100,50,25] )
    DNNmodel = DNNmodel.float()
    #
    CoOrg= TrainerExe()
    CoOrg.train(model = DNNmodel, train_loader = t_loader, num_epochs= num_epochs_max, RCcountMax = number_error_try, learning_rate = initial_learning_rate)
    #
    filename3 = "{path}/{filename}.obj".format(path = path, filename = filename)
    filehandler2 = open(filename3, 'wb')
    pickle.dump(DNNmodel, filehandler2)


def FitPredModel_Load (beta, dataSection1):
    # Original Version
    # data_gen_rs = np.load("../output/LIBDmultiple/DataGen/data_gen_T{batch}_{beta}_n{nrep}.npy".format(batch = batch, beta = beta, nrep = nrep))
    # Attach the original
    tdatax = np.expand_dims(dataSection1.X, axis = 0)
    tdata_rs = np.swapaxes(tdatax, 1, 2)
    # datacomp = np.concatenate((data_gen_rs, tdata_rs), axis=0)
    datacomp = tdata_rs
    #
    dataDNN = wrap_gene_layer(datacomp, dataSection1.obs, "Layer")
    return dataDNN


def FitPredModelNE_KAN(dataSection1, filename3, hidden_dims=[1, 800, 300, 100, 10], layer_weights=114514):
    random.seed(2020)
    torch.manual_seed(2020)
    np.random.seed(2020)
    g = torch.Generator()
    g.manual_seed(2021)
    tdatax = np.expand_dims(dataSection1.X, axis=0)
    tdata_rs = np.swapaxes(tdatax, 1, 2)
    DataTra = wrap_gene_layer(tdata_rs, dataSection1.obs, "Layer")
    t_loader = torch.utils.data.DataLoader(DataTra, batch_size=4, num_workers=0, shuffle=True,
                                           worker_init_fn=seed_worker, generator=g)
    layer_weights = layer_weights

    if len(hidden_dims) == 5:
        # if model_config == "original":
            model = KAN5_LIBD(in_channels=DataTra[1][0].shape[0], num_classes=7, hidden_dims=hidden_dims, importance_weights=layer_weights)
        # if model_config == "original":
        # else:
        #     model = WKAN5_LIBD(in_channels=DataTra[1][0].shape[0], num_classes=7, hidden_dims=hidden_dims, importance_weights=layer_weights)
    if len(hidden_dims) == 4:
        model = KAN4_LIBD(in_channels=DataTra[1][0].shape[0], num_classes=7, hidden_dims=hidden_dims, importance_weights=layer_weights)
    if len(hidden_dims) == 3:
        model = KAN3_LIBD(in_channels=DataTra[1][0].shape[0], num_classes=7, hidden_dims=hidden_dims, importance_weights=layer_weights)


    model = model.float()

    model = model.to(device)
    #
    CoOrg = TrainerExe()
    CoOrg.train(model=model, train_loader=t_loader, num_epochs=114514, RCcountMax=5, learning_rate=0.01)
    #

    # REMEMBER TO CHANGE DIR
    filename3 = filename3
    filehandler2 = open(filename3, 'wb')
    pickle.dump(model, filehandler2)
    return model



def FitPredModelNE_WKAN(dataSection1, filename3, hidden_dims=[1, 800, 300, 100, 10], layer_weights=114514):
    random.seed(2020)
    torch.manual_seed(2020)
    np.random.seed(2020)
    g = torch.Generator()
    g.manual_seed(2021)
    tdatax = np.expand_dims(dataSection1.X, axis=0)
    tdata_rs = np.swapaxes(tdatax, 1, 2)
    DataTra = wrap_gene_layer(tdata_rs, dataSection1.obs, "Layer")
    t_loader = torch.utils.data.DataLoader(DataTra, batch_size=4, num_workers=0, shuffle=True,
                                           worker_init_fn=seed_worker, generator=g)
    layer_weights = layer_weights

    if len(hidden_dims) == 5:
        model = WKAN5_LIBD(in_channels=DataTra[1][0].shape[0], num_classes=7, hidden_dims=hidden_dims, importance_weights=layer_weights)

    if len(hidden_dims) == 4:
        model = WKAN4_LIBD(in_channels=DataTra[1][0].shape[0], num_classes=7, hidden_dims=hidden_dims, importance_weights=layer_weights)
    if len(hidden_dims) == 3:
        model = WKAN3_LIBD(in_channels=DataTra[1][0].shape[0], num_classes=7, hidden_dims=hidden_dims, importance_weights=layer_weights)


    model = model.float()

    model = model.to(device)
    #
    CoOrg = TrainerExe()
    CoOrg.train(model=model, train_loader=t_loader, num_epochs=114514, RCcountMax=5, learning_rate=0.01)
    #

    # REMEMBER TO CHANGE DIR
    filename3 = filename3
    filehandler2 = open(filename3, 'wb')
    pickle.dump(model, filehandler2)
    return model




def FitPredModelNE_DNN(dataSection1, filename3, hidden_dims=[1, 800, 300, 100, 10], layer_weights = 114514):
    random.seed(2020)
    torch.manual_seed(2020)
    np.random.seed(2020)
    g = torch.Generator()
    g.manual_seed(2021)
    tdatax = np.expand_dims(dataSection1.X, axis=0)
    tdata_rs = np.swapaxes(tdatax, 1, 2)
    DataTra = wrap_gene_layer(tdata_rs, dataSection1.obs, "Layer")
    t_loader = torch.utils.data.DataLoader(DataTra, batch_size=4, num_workers=0, shuffle=True,
                                           worker_init_fn=seed_worker, generator=g)
    layer_weights = layer_weights

    if len(hidden_dims) == 5:
        model = DNN5_LIBD(in_channels=DataTra[1][0].shape[0], num_classes=7, hidden_dims=hidden_dims,
                         importance_weights=layer_weights)
    if len(hidden_dims) == 4:
        model = DNN4_LIBD(in_channels=DataTra[1][0].shape[0], num_classes=7, hidden_dims=hidden_dims,
                         importance_weights=layer_weights)
    if len(hidden_dims) == 3:
        model = DNN3_LIBD(in_channels=DataTra[1][0].shape[0], num_classes=7, hidden_dims=hidden_dims,
                          importance_weights=layer_weights)

    model = model.float()

    model = model.to(device)
    #
    CoOrg = TrainerExe()
    CoOrg.train(model=model, train_loader=t_loader, num_epochs=114514, RCcountMax=5, learning_rate=0.01)
    #

    # REMEMBER TO CHANGE DIR
    filename3 = filename3
    filehandler2 = open(filename3, 'wb')
    pickle.dump(model, filehandler2)
    return model



def OverallAccSummary(path):
    classresults = pd.read_csv(path, header=None)

# pandasのapply関数とlambdaについてcase_whenを与える
    classresults['Type'] = classresults.apply(lambda row:
                                              'Same' if row[0] == row[1] else
                                              'Neighbour' if abs(row[0] - row[1]) == 1 else
                                              'Other', axis=1)


    summaries = classresults['Type'].value_counts()


    exact_acc = summaries.get('Same', 0) / summaries.sum()
    Neighbor_acc = exact_acc + summaries.get('Neighbour', 0) / summaries.sum()


    print(exact_acc)
    print(Neighbor_acc)

    return [exact_acc, Neighbor_acc]




def report_prop_method_sc (folder, name, data_test, Val_loader, outname = ""):
    """
        Report the results of the proposed methods in comparison to the other method
        :folder: string: specified the folder that keep the proposed DNN method
        :name: string: specified the name of the DNN method, also will be used to name the output files
        :data_test: AnnData: the data of query data
        :Val_loader: Dataload: the validation data from dataloader
        :outname: string: specified the name of the output, default is the same as the name
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    filename2 = f"{folder}/{name}.obj"
    with open(filename2, 'rb') as filehandler:
        DNNmodel = pickle.load(filehandler)
    DNNmodel.to(device)
    # filename2 = "{folder}/{name}.obj".format(folder = folder, name = name)
    # filehandler = open(filename2, 'rb')
    # DNNmodel = pickle.load(filehandler)
    #
    coords_predict = np.zeros((data_test.obs.shape[0], 2))

    for i, img in enumerate(Val_loader):
        recon = DNNmodel(img)
        # coords_predict[i,:] = recon[0].detach().numpy()
        coords_predict[i, :] = recon[0].detach().cpu().numpy()
    np.savetxt("{folder}/{name}_predmatrix.csv".format(folder = folder, name = name), coords_predict, delimiter=",")
    return coords_predict
    # coords_predict = np.zeros((data_test.obs.shape[0],2))
    # #
    # for i, img in enumerate(Val_loader):
    #     img = img.to(device)
    #     recon = DNNmodel(img)
    #     coords_predict[i,:] = recon[0].detach().numpy()
    # np.savetxt("{folder}/{name}_predmatrix.csv".format(folder = folder, name = name), coords_predict, delimiter=",")
    # return coords_predict


def distCompute(data_merfish):
    celery_dist = []
    true_dist = []
    Qdata_loc = np.array(data_merfish.obs[['x_cord', 'y_cord']])
    celery_pred = np.array(data_merfish.obs[['x_celery', 'y_celery']])

    for i in tqdm(range(Qdata_loc.shape[0])):
        celery_i = celery_pred[i, :]
        celery_points = celery_pred[i+1:, :]
        celery_dist.extend(np.sqrt(np.sum((celery_points - celery_i)**2, axis=1)))


        true_i = Qdata_loc[i, :]
        true_points = Qdata_loc[i+1:, :]
        true_dist.extend(np.sqrt(np.sum((true_points - true_i)**2, axis=1)))
    return celery_dist, true_dist


def Predict_cord (data_test, path = "", filename = "PreOrg_Mousesc", location_data = None):
    testdata= (data_test.X.A if issparse(data_test.X) else data_test.X)
    if location_data is None:
        location_data = pd.DataFrame(np.ones((data_test.shape[0],2)), columns = ["psudo1", "psudo2"])
    ## Wrap up Validation data in to dataloader
    vdatax = np.expand_dims(testdata, axis = 0)
    vdata_rs = np.swapaxes(vdatax, 1, 2)
    DataVal = wrap_gene_location(vdata_rs, location_data)
    Val_loader= torch.utils.data.DataLoader(DataVal, batch_size=1, num_workers = 0)#num_workers = 1
    #
    cord = report_prop_method_sc(folder = path,
                        name = filename, data_test = data_test,
                        Val_loader = Val_loader)
    data_test.obs["x_cord_pred"] = cord[:,0]
    data_test.obs["y_cord_pred"] = cord[:,1]
    return cord





def pred_transform(pred_cord, data_train):
    data_train = data_train.copy()
    traindata = (data_train.X.A if issparse(data_train.X) else data_train.X)
    tdatax = np.expand_dims(traindata, axis = 0)
    tdata_rs = np.swapaxes(tdatax, 1, 2)
    test_cord = wrap_gene_location(tdata_rs, data_train.obs[['x_cord', 'y_cord']])

    pred_cord_transformx = pred_cord[:,0]*(test_cord.xmax-test_cord.xmin) + test_cord.xmin
    pred_cord_transformy = pred_cord[:,1]*(test_cord.ymax-test_cord.ymin) + test_cord.ymin
    pred_cord_transform = np.array([pred_cord_transformx, pred_cord_transformy]).T
    return pred_cord_transform

A = 6/11
C = 2436.36
B = -1
def pointTrans(celery_pred, left, xname, yname):
    x = celery_pred[:, 0]
    y = celery_pred[:, 1]
    x1 = x - 2*A*((A*x + B*y + C)/(A*A + B*B))
    y1 = y - 2*B*((A*x + B*y + C)/(A*A + B*B))
    left.obs[xname] = x1
    left.obs[yname] = y1
    # return x1, y1


def rotateMatrix(a):
    return np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])




def anim( xy, i,d11 = None):
    x0 = np.quantile(d11.obs['x_cord'], 0.5)
    y0 = 5000
    newxy=(xy-[x0,y0]) @ rotateMatrix(-2*i*np.pi/180) + [x0,y0]
    return newxy

def Coordinate_Distance(d11_left, d11_right,data_train, path="output/Winnow_MerfishDNN91I_500_250_100_50_10",
                              filename="fig6_2",
                              save_dir_left="output/Winnow_MerfishDNN91I_500_250_100_50_10/fig6_2_left_celery.npy",
                              save_dir_right="output/Winnow_MerfishDNN91I_500_250_100_50_10/fig6_2_right_celery.npy"):
    get_zscore(d11_left)  #
    get_zscore(d11_right)  #

    filename = filename
    path = path
    save_dir_left = save_dir_left
    save_dir_right = save_dir_right
    #
    pred_cord_left = Predict_cord(data_test=d11_left, path=path, filename=filename)
    pred_cord_transform_left = pred_transform(pred_cord_left, data_train = data_train)
    #
    pred_cord_right = Predict_cord(data_test=d11_right, path=path, filename=filename)
    pred_cord_transform_right = pred_transform(pred_cord_right, data_train = data_train)

    os.makedirs(path, exist_ok=True)  #
    # np.save(save_dir_left, pred_cord_transform_left)
    # np.save(save_dir_right, pred_cord_transform_right)  #

    #
    d11_left.obs['x_celery'] = pred_cord_transform_left[:, 0]
    #
    d11_left.obs['y_celery'] = pred_cord_transform_left[:, 1]
    #
    d11_right.obs['x_celery'] = pred_cord_transform_right[:, 0]
    #
    d11_right.obs['y_celery'] = pred_cord_transform_right[:, 1]

    celery_dist, true_dist = distCompute(d11_left)  #
    celery_dist_r, true_dist_r = distCompute(d11_right)  #

    celery_dist.extend(celery_dist_r)

    return celery_dist







def Coordinate_Distance_Result(d11, d11_left, d11_right,data_train,
                               path="output/Winnow_MerfishDNN91I_500_250_100_50_10",
                              filename="fig6_2",
                              save_dir_left="output/Winnow_MerfishDNN91I_500_250_100_50_10/fig6_2_left_celery.npy",
                              save_dir_right="output/Winnow_MerfishDNN91I_500_250_100_50_10/fig6_2_right_celery.npy"):
    get_zscore(d11_left)  #
    get_zscore(d11_right)  #

    filename = filename
    path = path
    save_dir_left = save_dir_left
    save_dir_right = save_dir_right
    #
    pred_cord_left = Predict_cord(data_test=d11_left, path=path, filename=filename)
    pred_cord_transform_left = pred_transform(pred_cord_left, data_train = data_train)
    #
    pred_cord_right = Predict_cord(data_test=d11_right, path=path, filename=filename)
    pred_cord_transform_right = pred_transform(pred_cord_right, data_train = data_train)

    os.makedirs(path, exist_ok=True)  #
    np.save(save_dir_left, pred_cord_transform_left)
    np.save(save_dir_right, pred_cord_transform_right)  #

    #
    d11_left.obs['x_celery'] = pred_cord_transform_left[:, 0]
    #
    d11_left.obs['y_celery'] = pred_cord_transform_left[:, 1]
    #
    d11_right.obs['x_celery'] = pred_cord_transform_right[:, 0]
    #
    d11_right.obs['y_celery'] = pred_cord_transform_right[:, 1]

    celery_dist, true_dist = distCompute(d11_left)  #
    celery_dist_r, true_dist_r = distCompute(d11_right)  #

    celery_dist.extend(celery_dist_r)
    true_dist.extend(true_dist_r)

    # return celery_dist

    print(scipy.stats.pearsonr(true_dist, celery_dist))

    #
    pointTrans(pred_cord_transform_left, d11_left, "x_celery", "y_celery")  #
    Qdata = concat([d11_left, d11_right])  #

    newxy = anim(np.array(Qdata.obs[['x_cord', 'y_cord']]), -30, d11 = d11)
    Qdata.obs['x_rotate'] = newxy[:, 0]
    Qdata.obs['y_rotate'] = newxy[:, 1]
    Qdata.obs['y_rotate'] = Qdata.obs['y_rotate'] + 500
    Qdata.obs['x_rotate'] = Qdata.obs['x_rotate'] + 800

    sq = lambda x, y: (x - y) ** 2
    outresult = np.sqrt(
        np.sum(sq(np.array(Qdata.obs[['x_rotate', 'y_rotate']]), np.array(Qdata.obs[['x_celery', 'y_celery']])),
               axis=1))
    print(np.median(outresult))
    print(np.mean(outresult))
    return outresult

def report_prop_method_LIBD_APP (folder, tissueID, name, dataSection2, traindata, Val_loader, coloruse,class_num, outname = ""):
    """
        Report the results of the proposed methods in comparison to the other method
        :folder: string: specified the folder that keep the proposed DNN method
        :name: string: specified the name of the DNN method, also will be used to name the output files
        :dataSection2: AnnData: the data of Section 2
        :traindata: AnnData: the data used in training data. This is only needed for compute SSIM
        :Val_loader: Dataload: the validation data from dataloader
        :outname: string: specified the name of the output, default is the same as the name
        :ImageSec2: Numpy: the image data that are refering to
    """
    if outname == "":
        outname = name
    filename2 = "{folder}/{name}.obj".format(folder = folder, name = name)
    filehandler = open(filename2, 'rb')
    DNNmodel = pickle.load(filehandler)
    DNNmodel.to(device)
    #
    coords_predict = np.zeros(dataSection2.obs.shape[0])
    payer_prob = np.zeros((dataSection2.obs.shape[0],class_num+2))
    for i, img in enumerate(Val_loader):
        #
        # img_on_device = [x.to(device) for x in img]
        recon = DNNmodel(img)
        # logitsvalue = np.squeeze(torch.sigmoid(recon[0]).detach().numpy(), axis = 0)
        logits_tensor = torch.sigmoid(recon[0])
        logitsvalue = logits_tensor.squeeze(0).detach().cpu().numpy()
        if (logitsvalue[class_num-2] == 1):
            coords_predict[i] = class_num
            payer_prob[i,(class_num + 1)] = 1
        else:
            logitsvalue_min = np.insert(logitsvalue, 0, 1, axis=0)
            logitsvalue_max = np.insert(logitsvalue_min, class_num, 0, axis=0)
            prb = np.diff(logitsvalue_max)
            # prbfull = np.insert(-prb[0], 0, 1 -logitsvalue[0,0], axis=0)
            prbfull = -prb.copy()
            coords_predict[i] = np.where(prbfull == prbfull.max())[0].max() + 1
            payer_prob[i,2:] = prbfull
    #
    dataSection2.obs["pred_layer"] = coords_predict.astype(int)
    payer_prob[:,0] = dataSection2.obs["pred_layer"]
    # payer_prob[:,0] = dataSection2.obs["Layer"]
    # payer_prob[:,1] = dataSection2.obs["pred_layer"]
    dataSection2.obs["pred_layer_str"] = coords_predict.astype(int).astype('str')
    return payer_prob
    # plot_layer(adata = dataSection2, folder = "{folder}{tissueID}".format(folder = folder, tissueID = tissueID), name = name, coloruse = coloruse)
    # plot_confusion_matrix ( referadata = dataSection2, filename = "{folder}{tissueID}/{name}conf_mat_fig".format(folder = folder, tissueID = tissueID, name = name))
    # np.savetxt("{folder}{tissueID}/{name}_probmat.csv".format(folder = folder, tissueID = tissueID, name = name), payer_prob, delimiter=',')


def normalize_expression_matrix(adata, target_sum=1e4):
    """
    对 AnnData 的表达矩阵 .X 进行归一化，使每个细胞的表达总量达到 target_sum。

    Parameters:
    - adata: AnnData对象，包含 .X 表达矩阵
    - target_sum: 每个细胞的总表达目标值 (默认 10,000)

    Returns:
    - 归一化后的 AnnData 对象
    """
    # 计算每个细胞的表达总和
    total_counts = np.array(adata.X.sum(axis=1)).flatten()

    # 避免除以零，设置最小值为 1
    total_counts[total_counts == 0] = 1

    # 执行归一化
    adata.X = (adata.X / total_counts[:, None]) * target_sum
    return adata


def EvaluateX (testdata,traindata,folder, name, class_num):
    ## Wrap up Validation data in to dataloader
    vdatax = np.expand_dims(testdata.X, axis = 0)
    vdata_rs = np.swapaxes(vdatax, 1, 2)
    DataVal = wrap_gene_layer(vdata_rs, testdata.obs)
    Val_loader= torch.utils.data.DataLoader(DataVal, batch_size=1, num_workers = 0)
    #
    report_prop_method_LIBD_APP(folder =  folder,
                        tissueID = 114514, coloruse = 1919810,
                       name = name,traindata = traindata,
                       dataSection2 = testdata,
                       Val_loader = Val_loader,
                            class_num = class_num)


def gene_selector(model_weight, top_n):
    top_n = top_n
    model_weight = model_weight

    values, index = model_weight.topk(top_n, dim=1)

    empty_set = set()

    for i in range(index.size(0)):
        new_set = set(index[i].tolist())  # 转换为集合（如果它不是的话）
        empty_set = empty_set.union(new_set)  # 使用 union 方法合并集合

    # assert len(list(empty_set)) == num_gene

    return list(empty_set)

def ADPreprocess(ADdata):
    ADdata.var_names_make_unique()
    sc.pp.filter_cells(ADdata, min_counts=500)
    sc.pp.filter_cells(ADdata, min_genes=100)
    sc.pp.normalize_per_cell(ADdata)
    sc.pp.log1p(ADdata)
    return ADdata