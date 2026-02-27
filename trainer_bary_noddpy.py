import argparse, os, glob
import torch, pdb
import numpy as np
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from PIL import Image
import math, random, time
import torch.backends.cudnn as cudnn
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from model_bary import *
from util.universal_dataset import TrainDataset
from torchvision.utils import save_image
from utils import unfreeze, freeze
from scipy import io as scio
import torch.nn.functional as F
import random
import cv2

# Training settings
parser = argparse.ArgumentParser(description="PyTorch SRResNet")
parser.add_argument("--batchSize", type=int, default=4, help="training batch size")
parser.add_argument("--nEpochs", type=int, default=200, help="number of epochs to train for")
parser.add_argument("--lr", type=float, default=2e-4, help="Learning Rate. Default=1e-4")
parser.add_argument("--step", type=int, default=20,
                    help="Sets the learning rate to the initial LR decayed by momentum every n epochs, Default: n=500")
parser.add_argument("--cuda", default=True, help="Use cuda?")
parser.add_argument("--resume", default=None, type=str,
                    help="Path to resume model (default: none")
parser.add_argument("--start-epoch", default=1, type=int, help="Manual epoch number (useful on restarts)")
parser.add_argument("--threads", type=int, default=16, help="Number of threads for data loader to use, (default: 1)")
parser.add_argument("--pretrained", default="", type=str, help="Path to pretrained model (default: none)")
parser.add_argument("--gpus", default="0", type=str, help="gpu ids (default: 0)")
parser.add_argument("--pairnum", default=10000000, type=int, help="num of paired samples")
parser.add_argument('--num_sources', type=int, default=3, help='number of source domains.')

parser.add_argument('--de_type', nargs='+', default=['denoise_15', 'denoise_25', 'denoise_50', 'derain', 'dehaze'],
                    help='which type of degradations is training and testing for.')
parser.add_argument('--denoise_dir', type=str, default='data/Train/Denoise/',
                    help='where clean images of denoising saves.')
parser.add_argument('--derain_dir', type=str, default='data/Train/Derain/',
                    help='where training images of deraining saves.')
parser.add_argument('--dehaze_dir', type=str, default='data/Train/Dehaze/',
                    help='where training images of dehazing saves.')
parser.add_argument('--deblur_dir', type=str, default='data/Train/Deblur/',
                    help='where training images of dehazing saves.')
parser.add_argument('--lowlight_dir', type=str, default='data/Train/lowlight/',
                    help='where training images of deraining saves.')

# parser.add_argument("--degset", default="./datasets/Deraining/train/Rain13K/input/", type=str, help="degraded data")
# parser.add_argument("--tarset", default="./datasets/Deraining/train/Rain13K/target/", type=str, help="target data")
parser.add_argument("--degset", default="./data/test/derain/Rain100L/input/", type=str, help="degraded data")
parser.add_argument("--tarset", default="./data/test/derain/Rain100L/target/", type=str, help="target data")
parser.add_argument("--Sigma", default=10000, type=float)
parser.add_argument("--sigma", default=1, type=float)
parser.add_argument("--optimizer", default="RMSprop", type=str, help="optimizer type")
parser.add_argument("--backbone", default="RCNet", type=str, help="architecture name")
parser.add_argument("--type", default="Deraining", type=str, help="to distinguish the ckpt name ")
parser.add_argument('--patch_size', type=int, default=128, help='patchsize of input.')
# path
parser.add_argument('--data_file_dir', type=str, default='data_dir/', help='where clean images of denoising saves.')


def get_parameter_number(net):
    total_num = sum(p.numel() for p in net.parameters())
    trainable_num = sum(p.numel() for p in net.parameters() if p.requires_grad)
    return total_num, trainable_num


def main():
    global opt, BaryIR, Lambda, K

    opt = parser.parse_args()
    print(opt)

    K = opt.num_sources
    cuda = opt.cuda
    if cuda:
        print("=> use gpu id: '{}'".format(opt.gpus))
        os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpus
        if not torch.cuda.is_available():
            raise Exception("No GPU found or Wrong gpu id, please run without --cuda")

    opt.seed = random.randint(1, 10000)
    print("Random Seed: ", opt.seed)
    torch.manual_seed(opt.seed)
    if cuda:
        torch.cuda.manual_seed(opt.seed)

    cudnn.benchmark = True
    #
    # deg_path = opt.degset
    # tar_path = opt.tarset
    # data_list = [deg_path, tar_path]
    print("------Datasets loaded------")
    if opt.backbone == 'BaryNet':
        BaryIR = BaryNet(decoder=True)
    elif opt.backbone == 'MRCNet':
        BaryIR = MRCNet(decoder=True)
    else:
        BaryIR = PromptIR(decoder=True)

    print("*****Using " + opt.backbone + " as the backbone architecture******")
    channels_latent = 384
    Pots = Potentials(num_potentials=opt.num_sources, channels=channels_latent, size=opt.patch_size)
    print("------Network constructed------")
    if cuda:
        BaryIR = BaryIR.cuda()
        Pots = Pots.cuda()

    if opt.resume:
        if os.path.isfile(opt.resume):
            print("=> loading checkpoint '{}'".format(opt.resume))
            checkpoint = torch.load(opt.resume)
            opt.start_epoch = checkpoint["epoch"] + 1
            BaryIR.load_state_dict(checkpoint["BaryIR"].state_dict())
            # Pots.load_state_dict(checkpoint["Pots"].state_dict())
    else:
        print("=> no checkpoint found at '{}'".format(opt.resume))
    # optionally copy weights from a checkpoint

    if opt.pretrained:
        if os.path.isfile(opt.pretrained):
            print("=> loading model '{}'".format(opt.pretrained))
            weights = torch.load(opt.pretrained)
            BaryIR.load_state_dict(weights['model'].state_dict())
            Pots.load_state_dict(weights['discr'].state_dict())
        else:
            print("=> no model found at '{}'".format(opt.pretrained))

    print("------Using Optimizer: '{}' ------".format(opt.optimizer))

    if opt.optimizer == 'Adam':
        BaryIR_optimizer = torch.optim.Adam(BaryIR.parameters(), lr=opt.lr/2)
        Pots_optimizer = torch.optim.Adam(Pots.parameters(), lr=opt.lr)
    elif opt.optimizer == 'RMSprop':
        BaryIR_optimizer = torch.optim.RMSprop(BaryIR.parameters(), lr=opt.lr/2)
        Pots_optimizer = torch.optim.RMSprop(Pots.parameters(), lr=opt.lr )

    print("------Training------")
    MSE = []
    BaryLOSS = []
    PotLOSS = []
    train_set = TrainDataset(opt)
    domain_sample_counts = train_set.get_num_samples()
    print(domain_sample_counts)
    inverse_counts = [1 / count for count in domain_sample_counts]
    total_inverse = sum(inverse_counts)

    Lambda = [inv_count / total_inverse for inv_count in inverse_counts]

    # train_set = DegTarDataset(deg_path, tar_path, pairnum=opt.pairnum)
    training_data_loader = DataLoader(dataset=train_set, num_workers=opt.threads, \
                                      batch_size=opt.batchSize, shuffle=True)
    num = 0
    deg_list = glob.glob(opt.degset + "*")
    deg_list = sorted(deg_list)

    tar_list = sorted(glob.glob(opt.tarset + "*"))
    for epoch in range(opt.start_epoch, opt.nEpochs + 1):
        BaryIRloss = 0
        Ploss = 0
        a, b = train(training_data_loader, BaryIR_optimizer, Pots_optimizer, BaryIR, Pots, epoch)

        p = evaluate(BaryIR, deg_list, tar_list)
        with open("./checksample/all/validation_results.txt", "a") as f:
            f.write(
                f"Net {opt.backbone}  Patchsize {opt.patch_size} Epoch {epoch}, psnr {p:.4f}, Batchsize {opt.batchSize}\n")

        BaryIRloss += a
        Ploss += b
        num += 1
        BaryIRloss = BaryIRloss / num
        BaryLOSS.append(format(BaryIRloss))
        PotLOSS.append(format(Ploss))
        scio.savemat('BaryIRLOSS.mat', {'BaryLOSS': BaryLOSS})
        scio.savemat('PotLOSS.mat', {'PotLOSS': PotLOSS})
        save_checkpoint(BaryIR, Pots, epoch)


def evaluate(BaryIR, deg_list, tar_list):
    cuda = True  # opt.cuda
    pp = 0
    print('----------validating-----------')
    with torch.no_grad():
        for deg_name, tar_name in zip(deg_list, tar_list):
            name = tar_name.split('/')
            print(name)
            print("Processing ", deg_name)
            deg_img = Image.open(deg_name).convert('RGB')
            tar_img = Image.open(tar_name).convert('RGB')
            deg_img = np.array(deg_img)
            tar_img = np.array(tar_img)
            h, w = deg_img.shape[0], deg_img.shape[1]
            shape1 = deg_img.shape
            shape2 = tar_img.shape
            if (h % 4) or (w % 4) != 0:
                continue
            if shape1 != shape2:
                continue
            deg_img = np.transpose(deg_img, (2, 0, 1))
            deg_img = torch.from_numpy(deg_img).float() / 255
            deg_img = deg_img.unsqueeze(0)
            data_degraded = deg_img

            tar_img = np.transpose(tar_img, (2, 0, 1))
            tar_img = torch.from_numpy(tar_img).float() / 255
            tar_img = tar_img.unsqueeze(0)
            gt = tar_img
            if cuda:
                BaryIR = BaryIR.cuda()
                gt = gt.cuda()
                data_degraded = data_degraded.cuda()
            else:
                BaryIR = BaryIR.cpu()

            # start_time = time.time()

            im_output, _, _, _ = BaryIR(data_degraded)
            im_output = im_output.squeeze(0).cpu()
            tar_img = tar_img.squeeze(0).cpu()

            im_output = im_output.numpy()
            tar_img = tar_img.numpy()
            im_output = np.transpose(im_output, (1, 2, 0))
            tar_img = np.transpose(tar_img, (1, 2, 0))
            pp += psnr(im_output, tar_img, data_range=1)
        p = pp / len(deg_list)
        return p


def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10"""
    lr = opt.lr * (0.1 ** (epoch // opt.step))
    return lr


def train(training_data_loader, BaryIR_optimizer, Pots_optimizer, BaryIR, Pots, epoch):
    lr = adjust_learning_rate(Pots_optimizer, epoch - 1)
    mse = []
    BaryIRloss = []
    Pots_loss = []

    for param_group in BaryIR_optimizer.param_groups:
        param_group["lr"] = lr / 2
    for param_group in Pots_optimizer.param_groups:
        param_group["lr"] = lr / 2

    print("Epoch={}, lr={}".format(epoch, Pots_optimizer.param_groups[0]["lr"]))

    for iteration, batch in enumerate(training_data_loader):
        ([clean_name, de_id], degraded, target) = batch
        # degraded = batch[0]
        # target = batch[1]
        # noise = np.random.normal(size=degraded.shape) * opt.noise_sigma/255.0
        # noise=torch.from_numpy(noise).float()

        if opt.cuda:
            target = target.cuda()
            degraded = degraded.cuda()

        # BaryIR  optmization

        freeze(Pots);
        unfreeze(BaryIR);

        BaryIR.zero_grad()
        # out_restored, _ = BaryIR(degraded)
        out_restored, source_latent, bary_latent, res_bary = BaryIR(degraded)

        # out_disc = Pots(out_restored).squeeze()
        diff = out_restored - target
        l1_loss = torch.mean(abs(diff))
        bary_loss = 0
        mse_loss = 0
        ort_loss = 0
        contra_loss = 0
        potential_loss = 0

        # Restoration map
        for i in range(out_restored.shape[0]):
            source_id_i = de_id[i]
            source_latent_slice_i = source_latent[i, :]
            bary_latent_slice_i = bary_latent[i, :]
            res_bary_slice_i = res_bary[i, :]

            # mse loss
            mse_loss = torch.mean((abs(source_latent_slice_i-bary_latent_slice_i)) ** 2) ** 0.5

            # common-specific orthogonality

            zc = F.normalize(
                bary_latent_slice_i.reshape(bary_latent.shape[1] * bary_latent.shape[2] * bary_latent.shape[3]), dim=0)
            orth = 0
            for j in range(out_restored.shape[0]):
                res_bary_slice_j = res_bary[j, :]

                zs = F.normalize(res_bary_slice_j.reshape(res_bary.shape[1] * res_bary.shape[2] * res_bary.shape[3]),
                                 dim=0)
                inner_product = torch.sum(zc * zs)
                orth += inner_product ** 2
            ort_loss = orth

            # contrastive loss

            zi = F.normalize(res_bary_slice_i.reshape(res_bary.shape[1] * res_bary.shape[2] * res_bary.shape[3]), dim=0)
            pos = neg = 0
            for j in range(out_restored.shape[0]):
                source_id_j = de_id[j]
                res_bary_slice_j = res_bary[j, :]
                zj = F.normalize(res_bary_slice_j.reshape(res_bary.shape[1] * res_bary.shape[2] * res_bary.shape[3]),
                                 dim=0)
                if source_id_i == source_id_j:
                    pos = pos + torch.mean(torch.exp(torch.sum(zi * zj) / 0.07))
                else:
                    neg = neg + torch.mean(torch.exp(torch.sum(zi * zj) / 0.07))
            contra_loss = -torch.log(pos / (pos + neg))

            potential_loss = 0
            # potential loss
            if source_id_i < 3:
                potential_loss += Pots(bary_latent_slice_i, 0).squeeze()
            else:
                potential_loss += Pots(bary_latent_slice_i, source_id_i - 2).squeeze()
            # for j in range(K):
            #     potential_loss -= Lambda[j] * Pots(bary_latent_slice_i, j).squeeze()

            # Sum
            if source_id_i < 3:
                bary_loss += Lambda[0] * (mse_loss + 0.05 * (ort_loss + contra_loss) - potential_loss)
            else:
                # print(Lambda[source_id_i-2])
                bary_loss += Lambda[source_id_i - 2] * (mse_loss + 0.05 * (ort_loss + contra_loss) - potential_loss)

        BaryIR_train_loss = bary_loss / out_restored.shape[0] + opt.Sigma * l1_loss

        BaryIRloss.append(BaryIR_train_loss.data)
        BaryIR_train_loss.backward()
        BaryIR_optimizer.step()

        unfreeze(Pots);
        freeze(BaryIR);

        # Potential

        if iteration % 1 == 0:
            Pots.zero_grad()
            potential_train_loss = 0
            _, _, bary_latent, _ = BaryIR(degraded)
            for i in range(out_restored.shape[0]):
                source_id_i = de_id[i]
                bary_latent_slice_i = bary_latent[i, :]
                potential_loss = 0
                if source_id_i < 3:
                    potential_loss += Pots(bary_latent_slice_i, 0).squeeze()
                else:
                    potential_loss += Pots(bary_latent_slice_i, source_id_i - 2).squeeze()

                if source_id_i < 3:
                    potential_train_loss += Lambda[0] * potential_loss
                else:
                    # print(Lambda[source_id_i-2])
                    potential_train_loss += Lambda[source_id_i - 2] * potential_loss

            potential = potential_train_loss / out_restored.shape[0]
            Pots_loss.append(potential_train_loss.data)
            potential.backward()
            Pots_optimizer.step()

        Pots.zero_grad()

        potential_constraint = 0
        for j in range(K):
            potential_constraint += Lambda[j] * Pots(bary_latent_slice_i, j).squeeze()

        potential_constraint_loss = 10 * (potential_constraint ** 2)
        potential_constraint_loss.backward()
        Pots_optimizer.step()


        if iteration % 10 == 0:
            print("Epoch {}({}/{}):Loss_Pots: {:.5}, Loss_BaryIR: {:.5}, Loss_mse: {:.5}".format(epoch,
                                                                                                 iteration,
                                                                                                 len(training_data_loader),
                                                                                                 potential_train_loss.data,
                                                                                                 BaryIR_train_loss.data,
                                                                                                 mse_loss,
                                                                                                 ))
            save_image(out_restored.data, './checksample/' + opt.type + '/output.png')
            # bary_vis = torch.mean(bary_latent, dim=1).view(3,1,16,-1)
            # res_vis = torch.mean(res_bary, dim=1).view(3,1,16,-1)
            # save_image(200*abs(res_vis).data, './checksample/' + opt.type + '/res.png')
            # save_image(20*abs(bary_vis).data, './checksample/' + opt.type + '/bary.png')
            save_image(degraded.data, './checksample/' + opt.type + '/degraded.png')
            save_image(target.data, './checksample/' + opt.type + '/target.png')

    return torch.mean(torch.FloatTensor(BaryIRloss)), torch.mean(torch.FloatTensor(Pots_loss))


def save_checkpoint(BaryIR, Pots, epoch):
    model_out_path = "checkpoint/" + "model_" + str(opt.type) + opt.backbone + str(opt.patch_size) + "_" + "_" + str(
        opt.nEpochs) + "_" + str(
        opt.sigma) + ".pth"
    state = {"epoch": epoch, "BaryIR": BaryIR, "Pots": Pots}
    if not os.path.exists("checkpoint/"):
        os.makedirs("checkpoint/")

    torch.save(state, model_out_path)

    print("Checkpoint saved to {}".format(model_out_path))


def PSNR(pred, gt, shave_border=0):
    height, width = pred.shape[:2]
    pred = pred[shave_border:height - shave_border, shave_border:width - shave_border]
    gt = gt[shave_border:height - shave_border, shave_border:width - shave_border]
    imdff = pred - gt
    rmse = math.sqrt((imdff ** 2).mean())
    if rmse == 0:
        return 100
    return 20 * math.log10(1.0 / rmse)


if __name__ == "__main__":
    main()