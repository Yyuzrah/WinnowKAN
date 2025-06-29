from .Module import *
from .util import *

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

def plot_layer(adata, folder, name, coloruse=None):
    """
    This function creates and saves two scatter plots of the input AnnData object. One plot displays the predicted
    layers and the other shows the reference layers.

    :param adata: AnnData object containing the data matrix and necessary metadata
    :param folder: Path to the folder where the plots will be saved
    :param name: Prefix for the output plot file names
    :param coloruse: (Optional) List of colors to be used for the plots, default is None

    :return: None, saves two scatter plots as PDF files in the specified folder
    """

    # Define the default color palette if none is provided
    if coloruse is None:
        colors_use = ['#46327e', '#365c8d', '#277f8e', '#1fa187', '#4ac16d', '#a0da39', '#fde725', '#ffbb78', '#2ca02c',
                      '#ff7f0e', '#1f77b4', '#800080', '#959595', '#ffff00', '#014d01', '#0000ff', '#ff0000', '#000000']
    else:
        colors_use = coloruse

    # Ensure all labels are strings
    adata.obs["pred_layer_str"] = adata.obs["pred_layer_str"].astype(str)
    adata.obs["Layer"] = adata.obs["Layer"].astype(str)

    # Extract unique labels from both the predicted layer and the reference layer
    unique_pred_labels = np.unique(adata.obs["pred_layer_str"])
    unique_ref_labels = np.unique(adata.obs["Layer"])

    # Combine unique labels and sort them
    unique_labels = sorted(set(unique_pred_labels) | set(unique_ref_labels))

    # Create a color map using the combined unique labels
    color_map = {label: colors_use[i % len(colors_use)] for i, label in enumerate(unique_labels)}

    # Assign colors to predicted layer and reference layer based on the combined color map
    adata.uns["pred_layer_str_colors"] = [color_map[label] for label in unique_pred_labels]
    adata.uns["Layer_colors"] = [color_map[label] for label in unique_ref_labels]

    # Create a copy of the input AnnData object to avoid modifying the original data
    cdata = adata.copy()

    # Scale the x2 and x3 columns in the AnnData object's observation data
    cdata.obs["x4"] = cdata.obs["x2"] * 50
    cdata.obs["x5"] = cdata.obs["x3"] * 50

    # Create and customize the predicted layer scatter plot
    fig = sc.pl.scatter(cdata, alpha=1, x="x5", y="x4", color="pred_layer_str", palette=[color_map[label] for label in unique_pred_labels], show=False, size=50)
    fig.set_aspect('equal', 'box')
    fig.figure.savefig("{path}/{name}_Layer_pred.pdf".format(path=folder, name=name), dpi=300)

    # Create and customize the reference layer scatter plot
    fig2 = sc.pl.scatter(cdata, alpha=1, x="x5", y="x4", color="Layer", palette=[color_map[label] for label in unique_ref_labels], show=False, size=50)
    fig2.set_aspect('equal', 'box')
    fig2.figure.savefig("{path}/{name}_Layer_ref.pdf".format(path=folder, name=name), dpi=300)

# Example call to the function
# plot_layer(adata, folder="output", name="layer_example")

from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns


def plot_confusion_matrix(referadata, filename, nlayer=7):
    """
    Plot the confusion matrix
    :referadata: the main adata that are working with
    :filename: Numpy [n x 2]: the predicted coordinates based on deep neural network
    :nlayer: Number of layers, default is 7
    """

    # Convert labels to string type for consistency
    referadata.obs["Layer"] = referadata.obs["Layer"].astype(str)
    referadata.obs["pred_layer"] = referadata.obs["pred_layer"].astype(str)

    # List of labels from 1 to nlayer as strings
    labellist = [str(i + 1) for i in range(nlayer)]

    # Compute confusion matrix
    conf_mat = confusion_matrix(referadata.obs["Layer"], referadata.obs["pred_layer"], labels=labellist)
    conf_mat_perc = conf_mat / conf_mat.sum(axis=1, keepdims=True)  # transform the matrix to be row percentage
    conf_mat_CR = classification_report(referadata.obs["Layer"], referadata.obs["pred_layer"], output_dict=True,
                                        labels=labellist)

    # Plot confusion matrix
    plt.figure(figsize=(10, 7))
    sns.heatmap(conf_mat_perc, annot=True, fmt=".2%", cmap='Blues', xticklabels=labellist, yticklabels=labellist)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.title('Confusion Matrix')
    plt.savefig(filename)
    plt.close()

    return conf_mat, conf_mat_perc, conf_mat_CR


def EvaluateOrg (testdata, tissueID, traindata, folder, class_num, coloruse = None, name = "layer_PreOrgv2_S3_DNN1000I", ):
    ## Wrap up Validation data in to dataloader
    vdatax = np.expand_dims(testdata.X, axis = 0)
    vdata_rs = np.swapaxes(vdatax, 1, 2)
    DataVal = wrap_gene_layer(vdata_rs, testdata.obs, "Layer")
    Val_loader= torch.utils.data.DataLoader(DataVal, batch_size=1, num_workers = 0)
    #
    report_prop_method_LIBD(folder = folder, tissueID = tissueID,
                       name = name,
                       dataSection2 = testdata, traindata = traindata,
                       Val_loader = Val_loader, coloruse = coloruse, class_num = class_num)


def report_prop_method_LIBD(folder, tissueID, name, dataSection2, traindata, Val_loader, coloruse, class_num, outname=""):
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
    filename2 = "{folder}/{name}.obj".format(folder=folder, name=name)
    filehandler = open(filename2, 'rb')
    DNNmodel = pickle.load(filehandler)
    #
    coords_predict = np.zeros(dataSection2.obs.shape[0])
    payer_prob = np.zeros((dataSection2.obs.shape[0], class_num + 2))
    for i, img in enumerate(Val_loader):
        #
        # img_on_device = [x.to(device) for x in img]
        recon = DNNmodel(img)
        # logitsvalue = np.squeeze(torch.sigmoid(recon[0]).detach().numpy(), axis = 0)
        logits_tensor = torch.sigmoid(recon[0])
        logitsvalue = logits_tensor.squeeze(0).detach().cpu().numpy()
        if (logitsvalue[class_num - 2] == 1):
            coords_predict[i] = class_num
            payer_prob[i, (class_num + 1)] = 1
        else:
            logitsvalue_min = np.insert(logitsvalue, 0, 1, axis=0)
            logitsvalue_max = np.insert(logitsvalue_min, class_num, 0, axis=0)
            prb = np.diff(logitsvalue_max)
            # prbfull = np.insert(-prb[0], 0, 1 -logitsvalue[0,0], axis=0)
            prbfull = -prb.copy()
            coords_predict[i] = np.where(prbfull == prbfull.max())[0].max() + 1
            payer_prob[i, 2:] = prbfull
    #
    dataSection2.obs["pred_layer"] = coords_predict.astype(int)
    payer_prob[:, 0] = dataSection2.obs["Layer"]
    payer_prob[:, 1] = dataSection2.obs["pred_layer"]
    dataSection2.obs["pred_layer_str"] = coords_predict.astype(int).astype('str')
    plot_layer(adata=dataSection2, folder="{folder}{tissueID}".format(folder=folder, tissueID=tissueID), name=name,
               coloruse=coloruse)
    plot_confusion_matrix(referadata=dataSection2,
                          filename="{folder}{tissueID}/{name}conf_mat_fig".format(folder=folder, tissueID=tissueID,
                                                                                  name=name))
    np.savetxt("{folder}{tissueID}/{name}_probmat.csv".format(folder=folder, tissueID=tissueID, name=name), payer_prob,
               delimiter=',')