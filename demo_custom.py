import argparse
from engine_custom import *
from models_custom import *
from custom_data import *
import os
import pandas as pd
from datasets import DatasetDict
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import numpy as np

parser = argparse.ArgumentParser(description='WILDCAT Training')
parser.add_argument('data', metavar='DIR',
                    help='path to dataset (e.g. data/')
parser.add_argument('--image-size', '-i', default=32, type=int,
                    metavar='N', help='image size (default: 224)')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=20, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--epoch_step', default=[30], type=int, nargs='+',
                    help='number of epochs to change learning rate')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=16, type=int,
                    metavar='N', help='mini-batch size (default: 256)')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--lrp', '--learning-rate-pretrained', default=0.1, type=float,
                    metavar='LR', help='learning rate for pre-trained layers')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--print-freq', '-p', default=0, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')




def split_dataframe_by_dirs(df, base_path):
    """
    Splits a DataFrame into train, val, and test subsets based on folder contents.
    """
    train_rows, test_rows, valid_rows = [], [], []
    cnt = 0
    for _, row in df.iterrows():
        img = row['img_name']
        if os.path.exists(os.path.join(base_path, 'test', img)):
            test_rows.append(row)
        elif os.path.exists(os.path.join(base_path, 'train', img)):
            train_rows.append(row)
        elif os.path.exists(os.path.join(base_path, 'val', img)):
            valid_rows.append(row)
        else:
            cnt += 1
    print(f"Files not found: {cnt}")
    train_df = pd.DataFrame(train_rows, columns=df.columns)
    test_df = pd.DataFrame(test_rows, columns=df.columns)
    valid_df = pd.DataFrame(valid_rows, columns=df.columns)

    return DatasetDict({
        'train': train_df,
        'test': test_df,
        'val': valid_df,
    })



# def iou_func(y_true, y_pred, threshold=0.5):
#     """
#     Computes the Intersection over Union (IoU) score for multi-label classification and returns the IoU loss.

#     Arguments:
#     y_true -- Ground truth labels (shape: [batch_size, num_classes])
#     y_pred -- Predicted labels (shape: [batch_size, num_classes]) as raw logits or probabilities
#     threshold -- Threshold to convert predicted probabilities to binary values (default: 0.5)

#     Returns:
#     iou_mean -- Mean IoU score for all classes
#     iou_loss -- IoU loss (1 - IoU)
#     """
#     # Apply threshold to predictions to convert to binary values
#     y_pred_binary = (y_pred > threshold).float()
    
#     # Calculate intersection and union for each label
#     intersection = (y_true * y_pred_binary).sum(dim=1).float()  # Element-wise multiplication
#     union = (y_true + y_pred_binary).clamp(0, 1).sum(dim=1).float()  # Element-wise addition

#     # Calculate IoU for each class
#     iou = (intersection + 1e-6) / (union + 1e-6)  # Add epsilon to avoid division by zero
#     iou_loss = 1 - iou.mean()  # Average IoU loss for the batch

#     return iou.mean(), iou_loss


# def iou_loss_func(y_true, y_pred, threshold=0.5):
#     """
#     Wrapper around iou_func to return only the loss value.
#     """
#     _, iou_loss = iou_func(y_true, y_pred, threshold)
#     return iou_loss
import torch
import torch.nn as nn

class IoULoss(nn.Module):
    def __init__(self, threshold=0.5):
        """
        IoU Loss for multi-label classification.

        Args:
            threshold (float): Threshold for converting predictions to binary.
        """
        super(IoULoss, self).__init__()
        self.threshold = threshold

    def forward(self, y_true, y_pred):
        """
        Forward pass to calculate IoU-based loss.

        Args:
            y_true (Tensor): Ground truth labels (shape: [batch_size, num_classes]).
            y_pred (Tensor): Predicted logits or probabilities (shape: [batch_size, num_classes]).

        Returns:
            Tensor: IoU loss value.
        """
        # Convert predictions to binary using threshold
        y_pred_binary = (y_pred > self.threshold).float()

        # Calculate intersection and union
        intersection = (y_true * y_pred_binary).sum(dim=1).float()
        union = (y_true + y_pred_binary).clamp(0, 1).sum(dim=1).float()

        # Compute IoU for each sample
        iou = (intersection + 1e-6) / (union + 1e-6)

        # Compute IoU loss
        iou_loss = 1 - iou.mean()

        return iou_loss



def iou_score(y_true, y_pred, threshold=0.5):
    """
    Computes the Intersection over Union (IoU) score for multi-label classification and returns the IoU loss.

    Arguments:
    y_true -- Ground truth labels (shape: [batch_size, num_classes])
    y_pred -- Predicted labels (shape: [batch_size, num_classes]) as raw logits or probabilities
    threshold -- Threshold to convert predicted probabilities to binary values (default: 0.5)

    Returns:
    iou_mean -- Mean IoU score for all classes
    iou_loss -- IoU loss (1 - IoU)
    """
    # Apply threshold to predictions to convert to binary values
    y_pred_binary = (y_pred > threshold).float()
    
    # Calculate intersection and union for each label
    intersection = (y_true * y_pred_binary).sum(dim=1).float()  # Element-wise multiplication
    union = (y_true + y_pred_binary).clamp(0, 1).sum(dim=1).float()  # Element-wise addition

    # Calculate IoU for each class
    iou = (intersection + 1e-6) / (union + 1e-6)  # Add epsilon to avoid division by zero
    # iou_loss = 1 - iou.mean()  # Average IoU loss for the batch

    return iou.mean()






def main_voc2007():
    global args, best_prec1, use_gpu
    args = parser.parse_args()

    use_gpu = torch.cuda.is_available()
    inp_name='ML-GCN/artifact_glove_word2vec.pkl'
    
    # define dataset
    # train_dataset = Voc2007Classification(args.data, 'trainval', inp_name=inp_name)
    # val_dataset = Voc2007Classification(args.data, 'test', inp_name=inp_name)

        # Load and concatenate CSVs
    df = pd.concat([
        pd.read_csv('ML-GCN/archive-4/final_stable_diffusion_31k.csv'),
        pd.read_csv('ML-GCN/archive-4/stable_diffusion_27k.csv'),
        pd.read_csv('ML-GCN/archive-4/artifact_presence_latent_diffusion.csv'),
        pd.read_csv('ML-GCN/archive-4/artifact_presence_giga_gan.csv'),
        pd.read_csv('ML-GCN/archive-4/artifact_presence_giga_gan_t2i_coco256.csv').dropna()
    ], axis=0)

    # Shuffle and reset index
    df = df.sample(frac=1).reset_index(drop=True)

    # Ensure labels are numeric
    df.iloc[:, 1:] = df.iloc[:, 1:].apply(pd.to_numeric, errors='coerce')
    df.fillna(0, inplace=True)  # Replace NaNs with 0



    # Split the DataFrame
    dataset_dict = split_dataframe_by_dirs(df, base_path='./ML-GCN/archive-4')
    train_dataset = MultiLabelDataset(dataset_dict, split="train", base_path='./ML-GCN/archive-4',inp_path=inp_name)
    val_dataset = MultiLabelDataset(dataset_dict, split="val", base_path='./ML-GCN/archive-4',inp_path=inp_name)

    
    num_classes = 70

    # load model
    model = gcn_resnet101(num_classes=num_classes, t=0.4, adj_file='binary_matrix_0.5.pkl')

    # define loss function (criterion)
    criterion = nn.MultiLabelSoftMarginLoss()  # Move to GPU if available
    # define optimizer
    optimizer = torch.optim.SGD(model.get_config_optim(args.lr, args.lrp),
                                lr=args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    state = {'batch_size': args.batch_size, 'image_size': args.image_size, 'max_epochs': args.epochs,
             'evaluate': args.evaluate, 'resume': args.resume, 'num_classes':num_classes}
    state['difficult_examples'] = True
    state['save_model_path'] = 'ML-GCN/checkpoint/custom_data/'
    state['workers'] = args.workers
    state['epoch_step'] = args.epoch_step
    state['lr'] = args.lr
    if args.evaluate:
        state['evaluate'] = True
        print('In the evaluation mode')
    engine = GCNMultiLabelMAPEngine(state)
    engine.learning(model, criterion, train_dataset, val_dataset, optimizer)



if __name__ == '__main__':
    main_voc2007()
