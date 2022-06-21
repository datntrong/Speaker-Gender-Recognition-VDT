import argparse
import copy
import sys
import warnings
from time import time

import torch
from sklearn.metrics import confusion_matrix, accuracy_score
from torch import nn

from Model.TDNNModel import XTDNN
from pre_data import build_dataloaders
from predict import singlemodel_class
from utils import print_eta

warnings.filterwarnings("ignore", message="numpy.dtype size changed")
warnings.filterwarnings("ignore", message="numpy.ufunc size changed")

parser = argparse.ArgumentParser(description='PyTorch Digital Mammography Training')

parser.add_argument('--lr', default=1e-3, type=float, help='learning rate')
parser.add_argument('--net_type', default='resnet', type=str, help='model')
parser.add_argument('--depth', default=50, choices=[11, 16, 19, 18, 34, 50, 152, 161, 169, 121, 201], type=int,
                    help='depth of model')
parser.add_argument('--weight_decay', default=5e-4, type=float, help='weight decay')
parser.add_argument('--finetune', '-f', action='store_true', help='Fine tune pretrained model')
parser.add_argument('--trainer', default='adam', type=str, help='optimizer')
parser.add_argument('--duration', default=2, type=float, help='time duration for each file in second')
parser.add_argument('--n_tests', default=3, type=int, help='number of tests in valid set')
parser.add_argument('--gender', '-g', action='store_true', help='classify gender')
parser.add_argument('--accent', '-a', action='store_true', help='accent classifier')
parser.add_argument('--random_state', '-r', default=2, type=int, help='random state in train_test_split')

parser.add_argument('--model_path', type=str, default=' ')
parser.add_argument('--gamma', default=0.5, type=float)
parser.add_argument('--batch_size', default=128, type=int)
parser.add_argument('--num_epochs', default=100, type=int,
                    help='Number of epochs in training')
parser.add_argument('--dropout_keep_prob', default=0.5, type=float)
parser.add_argument('--check_after', default=5,
                    type=int, help='check the network after check_after epoch')
parser.add_argument('--train_from', default=1,
                    choices=[0, 1],  # 0: from scratch, 1: from pretrained 1 (need model_path)
                    type=int,
                    help="training from beginning (1) or from the most recent ckpt (0)")

parser.add_argument('--frozen_until', '-fu', type=int, default=-1,
                    help="freeze until --frozen_util block")
parser.add_argument('--val_ratio', default=0.1, type=float,
                    help="number of training samples per class")

if __name__ == '__main__':
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    args, unknown = parser.parse_known_args()
    model = XTDNN()
    dset_loaders, train_info = build_dataloaders(args)
    train_fns, semi_fns, val_fns, train_lbs, semi_lbs, val_lbs, submit_lbs, submit_fns = train_info
    model = model.to(device)
    lr = args.learning_rate
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=5e-4)
    criterion = nn.CrossEntropyLoss()
    ################################
    N_train = len(train_lbs)
    N_valid = len(val_lbs)
    best_acc = 0

    checkpoint = torch.load('./model.pt')
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    best_acc = checkpoint['best_acc']
    model.eval()
    ########## Start training
    print('Start training ... ')
    t0 = time()
    for epoch in range(args.num_epochs):

        print('#################################################################')
        print('=> Training Epoch #%d, LR=%.10f' % (epoch + 1, lr))
        running_loss, running_corrects, tot = 0.0, 0.0, 0.0
        running_loss_src, running_corrects_src, tot_src = 0.0, 0.0, 0.0
        ########################
        model.train()
        torch.set_grad_enabled(True)
        ## Training
        for batch_idx, (inputs, labels, _) in enumerate(dset_loaders['train']):
            optimizer.zero_grad()
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            a, preds = torch.max(outputs.data, 1)
            running_loss += loss.item()
            running_corrects += preds.eq(labels.data).cpu().sum()
            tot += labels.size(0)
            sys.stdout.write('\r')
            try:
                batch_loss = loss.item()
            except NameError:
                batch_loss = 0

            top1acc = float(running_corrects) / tot
            sys.stdout.write('| Epoch [%2d/%2d] Iter [%3d/%3d]\tBatch loss %.4f\tTop1acc %.4f'
                             % (epoch + 1, args.num_epochs, batch_idx + 1,
                                (len(train_fns) // args.batch_size), batch_loss / args.batch_size,
                                top1acc))
            sys.stdout.flush()
            sys.stdout.write('\r')

        top1acc = float(running_corrects) / N_train
        epoch_loss = running_loss / N_train
        print('\n| Training loss %.8f\tTop1error %.4f' \
              % (epoch_loss, top1acc))

        print_eta(t0, epoch, args.num_epochs)
        ###################################
        ## Validation
        if (epoch + 1) % args.check_after == 0:
            # Validation
            ######################
            n_files = len(val_lbs)
            print('On test set')
            pred_output, pred_prob, _ = singlemodel_class(model, dset_loaders['test'], num_tests=2)
            print(confusion_matrix(semi_lbs, pred_output))
            acc1 = accuracy_score(semi_lbs, pred_output)
            acc2 = accuracy_score(semi_lbs, pred_prob)
            print('acc_output: {}, acc_prob: {}'.format(acc1, acc2))
            print('On validation')
            pred_output, pred_prob, _ = singlemodel_class(model, dset_loaders['val'], num_tests=args.n_tests)
            print(confusion_matrix(val_lbs, pred_output))
            acc1 = accuracy_score(val_lbs, pred_output)
            acc2 = accuracy_score(val_lbs, pred_prob)
            print('acc_output: {}, acc_prob: {}'.format(acc1, acc2))
            ########## end test on multiple windows ##############3
            running_loss, running_corrects, tot = 0.0, 0.0, 0.0
            torch.set_grad_enabled(False)
            model.eval()
            for batch_idx, (inputs, labels, _) in enumerate(dset_loaders['val']):
                inputs = inputs.to(device)
                labels = labels.to(device)
                outputs = model(inputs)

                _, preds = torch.max(outputs.data, 1)
                running_loss += loss.item()
                running_corrects += preds.eq(labels.data).cpu().sum()
                tot += labels.size(0)

            epoch_loss = running_loss / N_valid
            top1acc = float(running_corrects) / N_valid
            # top3error = 1 - float(runnning_topk_corrects)/N_valid
            print('| Validation loss %.8f\tTop1acc %.4f' \
                  % (epoch_loss, top1acc))

            ################## save model based on best acc
            if acc1 > best_acc:
                best_acc = acc1
                print('Saving model')
                best_model = copy.deepcopy(model)

                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': loss,
                    'best_acc': best_acc
                }, 'model.pt')

                print('=======================================================================')
                print('model saved')
