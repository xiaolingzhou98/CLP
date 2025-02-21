import os
import time
import argparse
import torch
import numpy as np
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision.transforms as transforms
from data_utils import *
from resnet import *
from CIFAR_process import *
from model import *
import time
from PIL import Image
from infilling.Baseline import *
from torch.utils.data import DataLoader, TensorDataset


parser = argparse.ArgumentParser(description='CLP')
parser.add_argument('--dataset', default='cifar100', type=str,
                    help='dataset (cifar10 or cifar100[default])')
parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--num_classes', type=int, default=100)
parser.add_argument('--imb_factor', type=float, default=1)
parser.add_argument('--test-batch-size', type=int, default=128, metavar='N',
                    help='input batch size for testing (default: 100)')
parser.add_argument('--epochs', type=int, default=200, metavar='N',
                    help='number of epochs to train')
parser.add_argument('--lr', '--learning-rate', default=1e-1, type=float,
                    help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
parser.add_argument('--nesterov', default=True, type=bool, help='nesterov momentum')
parser.add_argument('--weight-decay', '--wd', default=5e-4, type=float,
                    help='weight decay (default: 5e-4)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--num_meta', type=int, default=1000,
                    help='The number of meta data.')
parser.add_argument('--seed', type=int, default=100, metavar='S',
                    help='random seed (default: 42)')
parser.add_argument('--print-freq', '-p', default=100, type=int,
                    help='print frequency (default: 10)')
parser.add_argument('--lambda_', default=0.5, type=float)
parser.add_argument('--meta_lr', default=0.001, type=float)
parser.add_argument('--save_name', default='name', type=str)
parser.add_argument('--idx', default='0', type=str)
parser.add_argument('--meta_interval', type=int, default=1)
parser.add_argument('--meta_net_hidden_size', type=int, default=100)
parser.add_argument('--meta_net_num_layers', type=int, default=2)  
parser.add_argument('--meta_weight_decay', type=float, default=1e-4)
parser.add_argument('--beta', type=float, default=1)
parser.add_argument('--corruption_type', type=str, default=None)
parser.add_argument('--corruption_ratio', type=float, default=0.0)
parser.add_argument('--forground', type=str, default="mean")
parser.add_argument('--background', type=str, default="shuffle")

class CosineSimilarity(torch.nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self,x1,x2):
        x2 = x2.t()
        x = x1.mm(x2)
        x1_ = x1.norm(dim=1).unsqueeze(0).t()
        x2_ = x2.norm(dim=0).unsqueeze(0)
        x_frobenins = x1_.mm(x2_)
        dist = x.mul(1/x_frobenins)
        return dist


def weight_feature(features, logits_vector, loss_vector, labels, loss_tensor_last, margin_tensor_last, model, class_prop):

    labels_one_hot = F.one_hot(labels,num_classes=(args.dataset == 'cifar10' and 10 or 100)).float() 
    last_epoch_ave_loss = torch.mm(labels_one_hot,loss_tensor_last.unsqueeze(1))
    last_epoch_ave_margin = torch.mm(labels_one_hot,margin_tensor_last.unsqueeze(1))
    logits_labels = torch.sum(F.softmax(logits_vector,dim=1) * labels_one_hot,dim=1)
    logits_vector_grad = torch.norm(1- F.softmax(logits_vector,dim=1),dim=1)
    logits_others_max =(F.softmax(logits_vector,dim=1)[labels_one_hot!=1].reshape(F.softmax(logits_vector,dim=1).size(0),-1)).max(dim=1).values
    logits_margin =  logits_labels - logits_others_max
    relative_loss = loss_vector.unsqueeze(1) - last_epoch_ave_loss
    relative_margin = logits_margin.unsqueeze(1) - last_epoch_ave_margin
    all_class_weights = list(model.linear.named_leaves())[0][1]
    sample_weights = torch.norm(torch.mm(labels_one_hot, all_class_weights), dim=1)
    sample_cosin = torch.cosine_similarity(features, torch.mm(labels_one_hot,all_class_weights), dim=1)
    entropy =  torch.sum(F.softmax(logits_vector,dim=1)*F.log_softmax(logits_vector,dim=1),dim=1)

    feature = torch.cat([loss_vector.unsqueeze(1),
                         logits_margin.unsqueeze(1),
                         logits_vector_grad.unsqueeze(1),
                         sample_cosin, 
                         entropy.unsqueeze(1),
                         class_prop.unsqueeze(1),
                         last_epoch_ave_loss,
                         sample_weights.unsqueeze(1),
                         relative_loss,
                         relative_margin,
                        ],dim=1)
    return feature




args = parser.parse_args()
for arg in vars(args):
    print("{}={}".format(arg, getattr(args, arg)))

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
kwargs = {'num_workers': 1, 'pin_memory': False}
use_cuda = not args.no_cuda and torch.cuda.is_available()

torch.manual_seed(args.seed)
device = torch.device("cuda" if use_cuda else "cpu")

train_loader, validation_loader, test_loader, img_num_list, _ = build_dataloader(
        seed=args.seed,
        dataset=args.dataset,
        num_meta_total=args.num_meta,
        imbalanced_factor=args.imb_factor,
        corruption_type=args.corruption_type,
        corruption_ratio=args.corruption_ratio,
        batch_size=args.batch_size,
    )


best_prec1 = 0


# adjustments
label_freq_array = np.array(img_num_list) / np.sum(np.array(img_num_list))
adjustments = np.log(label_freq_array ** 1.0 + 1e-12)
adjustments = torch.FloatTensor(adjustments).cuda()


beta = 0.9999
effective_num = 1.0 - np.power(beta, img_num_list)
per_cls_weights = (1.0 - beta) / np.array(effective_num)
per_cls_weights = per_cls_weights / np.sum(per_cls_weights) * len(img_num_list)
per_cls_weights = torch.FloatTensor(per_cls_weights).cuda()
weights = torch.tensor(per_cls_weights).float()

def main():
    global args, best_prec1
    args = parser.parse_args()

    model = build_model()
    optimizer_a = torch.optim.SGD(model.params(), args.lr,
                                  momentum=args.momentum, nesterov=args.nesterov,
                                  weight_decay=args.weight_decay)
    cudnn.benchmark = True
    criterion = nn.CrossEntropyLoss().cuda()
    start_time = time.time()
    for epoch in range(args.epochs):
        adjust_learning_rate(optimizer_a, epoch + 1)

        ratio = args.lambda_

        train_CLP(train_loader, validation_loader, model, optimizer_a, epoch, criterion, ratio)
        prec1, preds, gt_labels = validate(test_loader, model)
        is_best = prec1 > best_prec1
        best_prec1 = max(prec1, best_prec1)

        
        save_checkpoint(args, {
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'best_acc1': best_prec1,
            'optimizer': optimizer_a.state_dict(),
        }, is_best)
        
        
    end_time = time.time()
    print(end_time-start_time)
    print('Best accuracy: ', best_prec1)


def train(train_loader, model, optimizer_a, epoch):

    losses = AverageMeter()
    top1 = AverageMeter()
    model.train()

    for i, (_, input, target) in enumerate(train_loader):
        input_var = to_var(input, requires_grad=False)
        target_var = to_var(target, requires_grad=False)

        _, y_f = model(input_var)
        del _
        cost_w = F.cross_entropy(y_f, target_var, reduce=False)
        l_f = torch.mean(cost_w)
        prec_train = accuracy(y_f.data, target_var.data, topk=(1,))[0]


        losses.update(l_f.item(), input.size(0))
        top1.update(prec_train.item(), input.size(0))

        optimizer_a.zero_grad()
        l_f.backward()
        optimizer_a.step()

        if i % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                epoch, i, len(train_loader),
                loss=losses,top1=top1))

class HAttMattingModel:
    def __init__(self, model_path):
        self.model = torch.load(model_path)
        self.model.eval()

    def predict_alpha(self, image):
        with torch.no_grad():
            image = image.unsqueeze(0) 
            alpha = self.model(image)
            return alpha.squeeze(0) 

def generate_foreground_background_masks(image_path, model):
    image = Image.open(image_path).convert('RGB')
    transform = transforms.Compose([
        transforms.Resize((512, 512)),  
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    image_tensor = transform(image).unsqueeze(0)  

    alpha = model.predict_alpha(image_tensor)

    alpha = alpha.squeeze(0).cpu().numpy()  
    foreground_mask = (alpha > 0.5).astype(np.float32)  
    background_mask = (alpha <= 0.5).astype(np.float32) 

    return foreground_mask, background_mask


def get_masks_for_dataloader(validation_loader, model, threshold=0.5):
    all_foreground_masks = []
    all_background_masks = []

    with torch.no_grad():
        for batch in validation_loader:
            images = batch
            
            for image in images:
                image = transforms.ToPILImage()(image)
                foreground_mask, background_mask = generate_foreground_background_masks(image, model, threshold)
                
                all_foreground_masks.append(foreground_mask)
                all_background_masks.append(background_mask)

    return all_foreground_masks, all_background_masks


def train_CLP(train_loader, validation_loader, model, optimizer_a, epoch, criterion, ratio):

    losses = AverageMeter()
    top1 = AverageMeter()
    model.train()
    criterion1 = nn.CrossEntropyLoss(reduction='none').cuda()
    loss_tensor_last = torch.ones([args.dataset == 'cifar10' and 10 or 100]).cuda()*5
    loss_total = torch.zeros([args.dataset == 'cifar10' and 10 or 100]).cuda()
    class_total = torch.zeros([args.dataset == 'cifar10' and 10 or 100]).cuda()
    margin_tensor_last = torch.ones([args.dataset == 'cifar10' and 10 or 100]).cuda()
    margin_total = torch.zeros([args.dataset == 'cifar10' and 10 or 100]).cuda()

    model_path = ''
    haattmatting_model = HAttMattingModel(model_path)
    all_foreground_masks_val, all_background_masks_val = get_masks_for_dataloader(validation_loader, haattmatting_model, threshold=0.5)
    all_foreground_masks, all_background_masks = get_masks_for_dataloader(train_loader, haattmatting_model, threshold=0.5)


    foreground_inpainter = {
        'mean': MeanInpainter(),
        'shuffle': ShuffleInpainter(),
        'tile': TileInpainter(),
        'random_color': RandomColorWithNoiseInpainter(),
        'factual_mixed_random_tile': FactualMixedRandomTileInpainter()
       }[args.foreground]

    background_inpainter = {
        'mean': MeanInpainter(),
        'shuffle': ShuffleInpainter(),
        'zero': lambda x, mask: x * mask 
        }[args.background]


    augmented_images = []
    augmented_foreground_masks = []
    augmented_background_masks = []

    with torch.no_grad():
        for images, in train_loader:
            for image, fg_mask, bg_mask in zip(images, all_foreground_masks, all_background_masks):
                foreground_filled = foreground_inpainter(image.unsqueeze(0), fg_mask.unsqueeze(0))
                foreground_filled = foreground_filled.squeeze(0)
                background_filled = background_inpainter(image.unsqueeze(0), bg_mask.unsqueeze(0))
                background_filled = background_filled.squeeze(0)

                augmented_image = foreground_filled * fg_mask + background_filled * (1 - fg_mask)
                augmented_images.append(augmented_image)
                augmented_foreground_masks.append(fg_mask)
                augmented_background_masks.append(bg_mask)


    augmented_images_val = []
    augmented_foreground_masks_val = []
    augmented_background_masks_val = []


    with torch.no_grad():
        for images, in validation_loader:
            for image, fg_mask, bg_mask in zip(images, all_foreground_masks_val, all_background_masks_val):
                foreground_filled_val = foreground_inpainter(image.unsqueeze(0), fg_mask.unsqueeze(0))
                foreground_filled_val = foreground_filled_val.squeeze(0)
                background_filled_val = background_inpainter(image.unsqueeze(0), bg_mask.unsqueeze(0))
                background_filled_val = background_filled_val.squeeze(0)

                augmented_image_val = foreground_filled_val * fg_mask + background_filled_val * (1 - fg_mask)
                augmented_images_val.append(augmented_image_val)
                augmented_foreground_masks_val.append(fg_mask)
                augmented_background_masks_val.append(bg_mask)


    augmented_meta = TensorDataset(
        torch.stack(augmented_images_val),
        torch.stack(augmented_foreground_masks_val),
        torch.stack(augmented_background_masks_val)
        )

    augmented_loader = DataLoader(
        dataset=augmented_meta,
        batch_size=validation_loader.batch_size,
        shuffle=True
        )

    for i, (_, input, target) in enumerate(train_loader):
        target = target.cuda()
        input_var = to_var(input, requires_grad=False)
        target_var = to_var(target, requires_grad=False)
        meta_net = MLP(out_size = args.num_classes, in_size = 10, hidden_size=args.meta_net_hidden_size, num_layers=args.meta_net_num_layers).cuda()
        meta_optimizer = torch.optim.Adam(meta_net.parameters(), lr=args.meta_lr, weight_decay=args.meta_weight_decay)

        meta_model = ResNet32(args.dataset == 'cifar10' and 10 or 100)
        meta_model.load_state_dict(model.state_dict())
        meta_model.cuda()

            

        feat_hat, y_f_hat = meta_model(input_var)
        

        pseudo_loss_vector = criterion1(y_f_hat, target_var.long())
        # current_prop = torch.softmax(y_f_hat, dim=1)

        pseudo_training_feature = weight_feature(feat_hat, y_f_hat, pseudo_loss_vector, target_var, loss_tensor_last, margin_tensor_last, meta_model, torch.tensor(label_freq_array))
        pseudo_lp = meta_net(pseudo_training_feature)
        # pseudo_lp = labels_one_hot * pseudo_lp

        gradients = input_var.grad.data

        with torch.no_grad():
            saliency_loss = 0.0
            for i in range(input_var.size(0)):
                grad_squared = (gradients[i] ** 2).sum()

                mask_weights = (1 - all_foreground_masks[i]).float()
                weighted_grad_squared = grad_squared * mask_weights

                saliency_loss += (weighted_grad_squared.sum() / mask_weights.sum())

        regularization_loss = ratio * saliency_loss / input_var.size(0)

        cls_loss_meta = criterion(y_f_hat + pseudo_lp, target_var) + regularization_loss
        meta_model.zero_grad()

        grads = torch.autograd.grad(cls_loss_meta, (meta_model.params()), create_graph=True)
        meta_lr = args.lr * ((0.01 ** int(epoch >= 160)) * (0.01 ** int(epoch >= 180)))
        meta_model.update_params(meta_lr, source_params=grads)

        _, input_val, target_val = next(iter(augmented_loader))
        input_val_var = to_var(input_val, requires_grad=False)
        target_val_var = to_var(target_val, requires_grad=False)

        _, y_val = meta_model(input_val_var)

        gradients_val = input_val_var.grad.data

        with torch.no_grad():
            saliency_loss_val = 0.0
            for i in range(input_var.size(0)):
                grad_squared_val = (gradients_val[i] ** 2).sum()

                mask_weights_val = (1 - all_foreground_masks_val[i]).float()
                weighted_grad_squared_val = grad_squared_val * mask_weights_val

                saliency_loss_val += (weighted_grad_squared_val.sum() / mask_weights_val.sum())

        regularization_loss_val = ratio * saliency_loss_val / input_val_var.size(0)

        cls_meta = F.cross_entropy(y_val, target_val_var) + regularization_loss_val
        meta_optimizer.zero_grad()
        cls_meta.backward()
            
        meta_optimizer.step()

        

        del grad_cv, grads

        features, predicts = model(input_var)
        loss_vector = criterion1(predicts, target_var.long())
        training_feature = weight_feature(features, predicts, loss_vector, target_var, loss_tensor_last, margin_tensor_last, meta_model, torch.tensor(label_freq_array))
        lp = meta_net(training_feature.detach())
        # lp = labels_one_hot * lp
        
        gradients = input_var.grad.data

        with torch.no_grad():
            saliency_loss = 0.0
            for i in range(input_var.size(0)):
                grad_squared = (gradients[i] ** 2).sum()

                mask_weights = (1 - all_foreground_masks[i]).float()
                weighted_grad_squared = grad_squared * mask_weights

                saliency_loss += (weighted_grad_squared.sum() / mask_weights.sum())

        regularization_loss = ratio * saliency_loss / input_var.size(0)
        
        cls_loss = criterion(predicts + lp, target_var) + regularization_loss

        prec_train = accuracy(predicts.data, target_var.data, topk=(1,))[0]


        losses.update(cls_loss.item(), input.size(0))
        top1.update(prec_train.item(), input.size(0))

        optimizer_a.zero_grad()
        cls_loss.backward()
        optimizer_a.step()

        if i % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                epoch, i, len(train_loader),
                loss=losses,top1=top1))
    loss_tensor_last = loss_total/(class_total + 1e-6)
    margin_tensor_last = margin_total/(class_total + 1e-6)

def validate(val_loader, model):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    criterion = nn.CrossEntropyLoss()

    model.eval()

    true_labels = []
    preds = []

    end = time.time()
    for i, (_, input, target) in enumerate(val_loader):
        target = target.cuda()
        input = input.cuda()
        input_var = torch.autograd.Variable(input)
        target_var = torch.autograd.Variable(target)

        with torch.no_grad():
            _, output = model(input_var)
        losses = criterion(output, target)
        output_numpy = output.data.cpu().numpy()
        preds_output = list(output_numpy.argmax(axis=1))

        true_labels += list(target_var.data.cpu().numpy())
        preds += preds_output


        prec1 = accuracy(output.data, target, topk=(1,))[0]
        top1.update(prec1.item(), input.size(0))

        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Test: [{0}/{1}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss:.4f}\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                i, len(val_loader), batch_time=batch_time, loss=losses,
                top1=top1))

    print(' * Prec@1 {top1.avg:.3f}'.format(top1=top1))

    return top1.avg, preds, true_labels

def build_model():
    model = ResNet32(args.dataset == 'cifar10' and 10 or 100)

    if torch.cuda.is_available():
        model.cuda()
        torch.backends.cudnn.benchmark = True


    return model

def to_var(x, requires_grad=True):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x, requires_grad=requires_grad)


class AverageMeter(object):

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def adjust_learning_rate(optimizer, epoch):
    lr = args.lr * ((0.01 ** int(epoch >= 160)) * (0.01 ** int(epoch >= 180)))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def accuracy(output, target, topk=(1,)):
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def save_checkpoint(args, state, is_best):
    path = 'checkpoint/ours/' + args.idx + args.dataset + str(args.imb_factor) + '/'
    if not os.path.exists(path):
        os.makedirs(path)
    filename = path + args.save_name + '_ckpt.pth.tar'
    if is_best:
        torch.save(state, filename)

        
if __name__ == '__main__':
    main()