#Useful libraries
import copy
import torch
from torchvision import datasets,transforms
from sampling import cifar_iid, cifar_noniid

def exp_details(args):
    print('\nExperimental details:')
    print(f'    Model     : {args.model}')
    print(f'    Optimizer : {args.optimizer}')
    print(f'    Learning  : {args.lr}')
    print(f'    Global Rounds   : {args.epochs}\n')

    print('    Federated parameters:')
    if args.iid:
        print('    IID')
    else:
        print('    Non-IID')
    print(f'    Fraction of users  : {args.frac}')
    print(f'    Local Batch size   : {args.local_bs}')
    print(f'    Local Epochs       : {args.local_ep}\n')
    
    print('Mixed Parameters:')
    print(f'Alpha_batch: {args.alpha_b}')
    print(f'Alpha_group: {args.alpha_g}')	
    return

def get_dataset(args):
    #[TODO] Add wrapper for multiple datasets
    data_dir = '../data/cifar/'
    
    #Normalize used with mean and stds of Cifar10
    apply_transform = transforms.Compose(
        [transforms.RandomResizedCrop(32),
         transforms.RandomHorizontalFlip(),
         transforms.ToTensor(),
         transforms.Normalize([0.49139968, 0.48215841, 0.44653091], [0.24703223, 0.24348513, 0.26158784])])

    
    train_dataset = datasets.CIFAR10(data_dir, train= True,download=True, 
                                   transform=apply_transform)

    test_dataset = datasets.CIFAR10(data_dir,train=False,download=True,
                                  transform=apply_transform)
    
    #sample training
    if args.iid:
        #sample IID user
        user_group = cifar_iid(train_dataset,args.num_users)
    else:
        #sample Non-IID user
        user_group = cifar_noniid(train_dataset,args.num_users)
   
    return train_dataset, test_dataset, user_group 


def average_weights(w):
    """
    Returns the average of the weights.
    """
    w_avg = copy.deepcopy(w[0])
    for key in w_avg.keys():
        for i in range(1, len(w)):
            w_avg[key] += w[i][key]
        w_avg[key] = torch.div(w_avg[key], len(w))
    return w_avg