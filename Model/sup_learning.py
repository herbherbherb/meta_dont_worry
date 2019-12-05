import model_test as mc
import torch
import pickle, argparse, os

def get_args():
    parser = argparse.ArgumentParser()
    # basic info
    parser.add_argument("--model_name",                 default="DT",     type=str)
    parser.add_argument("--load",                       default=False,     action="store_true",    help="whether to load existing model")
    parser.add_argument("--current_epoch",              default=0,         type=int)
    # hyperparameter
    parser.add_argument("--num_epochs",                 default=1,        type=int)
    parser.add_argument("--batch_size",                 default=10,         type=int)
    parser.add_argument("--input_size",                 default=4,        type=int)
    parser.add_argument("--num_train",                  default=500,        type=int)
    parser.add_argument("--hidden_layer",               default=16,        type=int)

    parser.add_argument("--init_lr",                    default=1e-3,      type=float, help=("Start learning rate"))
    parser.add_argument("--num_class",                  default=5,         type=int,   help="num_class")
    parser.add_argument("--num_component",              default=4,         type=int, help="pca num_component , 10 is good")
    parser.add_argument("--weight_decay",               default=0.0,       type=float, help="l2 weight decay loss, 0.002 is a good number")
    parser.add_argument("--momentum",                   default=0.9,       type=float, help="momentum value, 0.9 is good")   
    parser.add_argument("--period_save_model",          default=1,         type=int,   help="period to save the model")
    # parse the arguments
    args = parser.parse_args()
    return args

config = get_args()
print(config)

if __name__ == '__main__':
    md_DT  = mc.model_DT(config)

    md_SVM = mc.model_SVM(config)


    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    md_ANN = mc.model_ANN(config,device)
    model  = {'DT':md_DT,'SVM':md_SVM,'ANN':md_ANN,'MLP':md_ANN}
    model[config.model_name].main()


