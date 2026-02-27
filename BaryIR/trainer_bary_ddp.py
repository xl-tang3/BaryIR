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
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

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
parser.add_argument("--batchSize", type=int, default=4, help="training batch size per GPU")
parser.add_argument("--nEpochs", type=int, default=200, help="number of epochs to train for")
parser.add_argument("--lr", type=float, default=2e-4, help="Learning Rate. Default=1e-4")
parser.add_argument("--step", type=int, default=20,
                    help="Sets the learning rate to the initial LR decayed by momentum every n epochs, Default: n=500")
parser.add_argument("--cuda", default=True, help="Use cuda?")
parser.add_argument("--resume", default=None, type=str,
                    help="Path to resume model (default: none")
parser.add_argument("--start-epoch", default=1, type=int, help="Manual epoch number (useful on restarts)")
parser.add_argument("--threads", type=int, default=4, help="Number of threads for data loader to use")
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
parser.add_argument('--single_dir', type=str, default='data/Train/single/',
                    help='where training images of deraining saves.')

parser.add_argument("--degset", default="./data/test/derain/Rain100L/input/", type=str, help="degraded data")
parser.add_argument("--tarset", default="./data/test/derain/Rain100L/target/", type=str, help="target data")
parser.add_argument("--Sigma", default=10000, type=float)
parser.add_argument("--sigma", default=1, type=float)
parser.add_argument("--optimizer", default="RMSprop", type=str, help="optimizer type")
parser.add_argument("--backbone", default="RCNet", type=str, help="architecture name")
parser.add_argument("--type", default="Deraining", type=str, help="to distinguish the ckpt name ")
parser.add_argument('--patch_size', type=int, default=128, help='patchsize of input.')
parser.add_argument('--data_file_dir', type=str, default='data_dir/', help='where clean images of denoising saves.')
parser.add_argument("--local_rank", type=int, default=-1, help="Local rank for DDP")


def get_parameter_number(net):
    total_num = sum(p.numel() for p in net.parameters())
    trainable_num = sum(p.numel() for p in net.parameters() if p.requires_grad)
    return total_num, trainable_num

def is_main_process():
    return dist.get_rank() == 0

def main():
    global opt, BaryIR, Lambda, K

    opt = parser.parse_args()

    if 'LOCAL_RANK' in os.environ:
        local_rank = int(os.environ['LOCAL_RANK'])
    else:
        local_rank = 0

    if opt.cuda:
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend='nccl')
        device = torch.device('cuda', local_rank)
    else:
        device = torch.device('cpu')
    
    if is_main_process():
        print(opt)

    K = opt.num_sources

    opt.seed = random.randint(1, 10000)
    if is_main_process():
        print("Random Seed: ", opt.seed)
    
    seed_tensor = torch.tensor(opt.seed, device=device)
    dist.broadcast(seed_tensor, 0)
    opt.seed = seed_tensor.item()
    
    torch.manual_seed(opt.seed)
    if opt.cuda:
        torch.cuda.manual_seed(opt.seed)

    cudnn.benchmark = True

    if is_main_process():
        print("------Datasets loaded------")
    
    if opt.backbone == 'BaryNet':
        BaryIR = BaryNet(decoder=True)
    elif opt.backbone == 'MRCNet':
        BaryIR = MRCNet(decoder=True)
    else:
        BaryIR = PromptIR(decoder=True)

    if is_main_process():
        print("*****Using " + opt.backbone + " as the backbone architecture******")
        print("------Network constructed------")

    channels_latent = 384
    Pots = Potentials(num_potentials=opt.num_sources, channels=channels_latent, size=opt.patch_size)
    
    if opt.cuda:
        BaryIR = BaryIR.to(device)
        Pots = Pots.to(device)

    if opt.resume:
        if os.path.isfile(opt.resume):
            if is_main_process():
                print("=> loading checkpoint '{}'".format(opt.resume))
            checkpoint = torch.load(opt.resume, map_location=device)
            opt.start_epoch = checkpoint["epoch"] + 1
            
            state_dict = checkpoint["BaryIR"].state_dict()
            new_state_dict = {}
            for k, v in state_dict.items():
                name = k.replace("module.", "") if "module." in k else k
                new_state_dict[name] = v
            BaryIR.load_state_dict(new_state_dict)
    else:
        if is_main_process():
            print("=> no checkpoint found at '{}'".format(opt.resume))

    if opt.pretrained:
        if os.path.isfile(opt.pretrained):
            if is_main_process():
                print("=> loading model '{}'".format(opt.pretrained))
            weights = torch.load(opt.pretrained, map_location=device)
            state_dict_m = weights['model'].state_dict() if 'model' in weights else weights
            new_state_dict_m = {k.replace("module.", ""): v for k, v in state_dict_m.items()}
            BaryIR.load_state_dict(new_state_dict_m, strict=False)

            if 'discr' in weights:
                state_dict_d = weights['discr'].state_dict()
                new_state_dict_d = {k.replace("module.", ""): v for k, v in state_dict_d.items()}
                Pots.load_state_dict(new_state_dict_d, strict=False)
        else:
            if is_main_process():
                print("=> no model found at '{}'".format(opt.pretrained))

    # Wrapping
    # Note: find_unused_parameters=True is set to handle cases where not all DDP params 
    # are used in every forward pass (e.g. BaryIR update doesn't update Pots)
    BaryIR = DDP(BaryIR, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True)
    Pots = DDP(Pots, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True)

    if is_main_process():
        print("------Using Optimizer: '{}' ------".format(opt.optimizer))

    if opt.optimizer == 'Adam':
        BaryIR_optimizer = torch.optim.Adam(BaryIR.parameters(), lr=opt.lr/2)
        Pots_optimizer = torch.optim.Adam(Pots.parameters(), lr=opt.lr)
    elif opt.optimizer == 'RMSprop':
        BaryIR_optimizer = torch.optim.RMSprop(BaryIR.parameters(), lr=opt.lr/2)
        Pots_optimizer = torch.optim.RMSprop(Pots.parameters(), lr=opt.lr )

    if is_main_process():
        print("------Training------")
    
    MSE = []
    BaryLOSS = []
    PotLOSS = []
    
    train_set = TrainDataset(opt)
    
    domain_sample_counts = train_set.get_num_samples()
    if is_main_process():
        print(domain_sample_counts)
    inverse_counts = [1 / count for count in domain_sample_counts]
    total_inverse = sum(inverse_counts)
    Lambda = [inv_count / total_inverse for inv_count in inverse_counts]

    train_sampler = DistributedSampler(train_set, shuffle=True)
    
    training_data_loader = DataLoader(dataset=train_set, 
                                      num_workers=opt.threads, 
                                      batch_size=opt.batchSize, 
                                      sampler=train_sampler,
                                      pin_memory=True)

    num = 0
    deg_list = glob.glob(opt.degset + "*")
    deg_list = sorted(deg_list)
    tar_list = sorted(glob.glob(opt.tarset + "*"))

    for epoch in range(opt.start_epoch, opt.nEpochs + 1):
        train_sampler.set_epoch(epoch)
        
        BaryIRloss = 0
        Ploss = 0
        a, b = train(training_data_loader, BaryIR_optimizer, Pots_optimizer, BaryIR, Pots, epoch, local_rank)

        if is_main_process():
            p = evaluate(BaryIR, deg_list, tar_list, device)
            
            if not os.path.exists("./checksample/all/"):
                os.makedirs("./checksample/all/")
                
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
        
        dist.barrier()


def evaluate(BaryIR, deg_list, tar_list, device):
    pp = 0
    print('----------validating-----------')
    BaryIR.eval()
    
    with torch.no_grad():
        for deg_name, tar_name in zip(deg_list, tar_list):
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
            
            gt = gt.to(device)
            data_degraded = data_degraded.to(device)

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
    lr = opt.lr * (0.1 ** (epoch // opt.step))
    return lr

def train(training_data_loader, BaryIR_optimizer, Pots_optimizer, BaryIR, Pots, epoch, local_rank):
    lr = adjust_learning_rate(Pots_optimizer, epoch - 1)
    
    BaryIR.train()
    Pots.train()

    for param_group in BaryIR_optimizer.param_groups:
        param_group["lr"] = lr / 2
    for param_group in Pots_optimizer.param_groups:
        param_group["lr"] = lr / 2

    if is_main_process():
        print("Epoch={}, lr={}".format(epoch, Pots_optimizer.param_groups[0]["lr"]))

    epoch_bary_loss = 0.0
    epoch_pot_loss = 0.0
    count = 0

    for iteration, batch in enumerate(training_data_loader):
        ([clean_name, de_id], degraded, target) = batch

        if opt.cuda:
            target = target.cuda(local_rank, non_blocking=True)
            degraded = degraded.cuda(local_rank, non_blocking=True)

        # ---------------------------
        # 1. BaryIR optimization
        # ---------------------------
        freeze(Pots)
        unfreeze(BaryIR)

        BaryIR_optimizer.zero_grad()
        out_restored, source_latent, bary_latent, res_bary = BaryIR(degraded)

        diff = out_restored - target
        l1_loss = torch.mean(abs(diff))
        bary_loss = 0
        mse_loss = 0
        ort_loss = 0
        contra_loss = 0
        
        # We need to call Pots during BaryIR loss calc. 
        # Since Pots is frozen, we don't need DDP sync here, but DDP wrapper still exists.
        # We use no_sync just to be safe and avoid "multiple forward" confusion, 
        # though strictly speaking gradients aren't flowing into Pots parameters anyway.
        with Pots.no_sync():
            for i in range(out_restored.shape[0]):
                source_id_i = de_id[i]
                source_latent_slice_i = source_latent[i, :]
                bary_latent_slice_i = bary_latent[i, :]
                res_bary_slice_i = res_bary[i, :]

                mse_loss = torch.mean((abs(source_latent_slice_i-bary_latent_slice_i)) ** 2) ** 0.5

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
                contra_loss = -torch.log((pos + 1e-6) / (pos + neg + 1e-6))

                potential_loss_val = 0
                if source_id_i < 3:
                    potential_loss_val += Pots(bary_latent_slice_i, 0).squeeze()
                else:
                    potential_loss_val += Pots(bary_latent_slice_i, source_id_i - 2).squeeze()

                if source_id_i < 3:
                    bary_loss += Lambda[0] * (mse_loss + 0.05 * (ort_loss + contra_loss) - potential_loss_val)
                else:
                    bary_loss += Lambda[source_id_i - 2] * (mse_loss + 0.05 * (ort_loss + contra_loss) - potential_loss_val)

        BaryIR_train_loss = bary_loss / out_restored.shape[0] + opt.Sigma * l1_loss

        epoch_bary_loss += BaryIR_train_loss.item()
        
        BaryIR_train_loss.backward()
        BaryIR_optimizer.step()

        # ---------------------------
        # 2. Potential (Pots) Optimization
        # ---------------------------
        unfreeze(Pots)
        freeze(BaryIR)
        
        Pots_optimizer.zero_grad()
        potential_train_loss_acc = 0.0
        
        # Because we have multiple calls to Pots (inside loops) followed by backward,
        # we MUST use no_sync() to prevent DDP from thinking we are crazy.
        # We will manually sync gradients after the block.
        with Pots.no_sync():
            
            # --- Part A: Minimize Potential ---
            if iteration % 1 == 0:
                with torch.no_grad():
                    _, _, bary_latent, _ = BaryIR(degraded)
                
                batch_size = out_restored.shape[0]
                for i in range(batch_size):
                    source_id_i = de_id[i]
                    bary_latent_slice_i = bary_latent[i, :]
                    
                    potential_loss = 0
                    if source_id_i < 3:
                        potential_loss = Pots(bary_latent_slice_i, 0).squeeze()
                    else:
                        potential_loss = Pots(bary_latent_slice_i, source_id_i - 2).squeeze()

                    if source_id_i < 3:
                        weighted_loss = Lambda[0] * potential_loss
                    else:
                        weighted_loss = Lambda[source_id_i - 2] * potential_loss
                    
                    potential_train_loss_acc += weighted_loss.item()
                    
                    # Backward per sample (accumulates in .grad locally)
                    loss_i = weighted_loss / batch_size
                    loss_i.backward()

            # --- Part B: Constraint (Sum Squared) ---
            # Using bary_latent_slice_i from previous loop (last element) - matching original logic
            potential_constraint = 0
            for j in range(K):
                potential_constraint += Lambda[j] * Pots(bary_latent_slice_i, j).squeeze()

            potential_constraint_loss = 10 * (potential_constraint ** 2)
            potential_constraint_loss.backward()
        
        # --- Manual Gradient Synchronization ---
        # Since we used no_sync(), we must manually average gradients across GPUs
        for param in Pots.parameters():
            if param.grad is not None:
                dist.all_reduce(param.grad)
                param.grad /= dist.get_world_size()

        Pots_optimizer.step()
        
        epoch_pot_loss += potential_train_loss_acc
        count += 1

        if iteration % 10 == 0 and is_main_process():
            print("Epoch {}({}/{}):Loss_Pots: {:.5}, Loss_BaryIR: {:.5}, Loss_mse: {:.5}".format(epoch,
                                                                                                 iteration,
                                                                                                 len(training_data_loader),
                                                                                                 potential_train_loss_acc / (out_restored.shape[0] if out_restored.shape[0]>0 else 1),
                                                                                                 BaryIR_train_loss.item(),
                                                                                                 mse_loss,
                                                                                                 ))
            if not os.path.exists('./checksample/' + opt.type):
                os.makedirs('./checksample/' + opt.type)
                
            save_image(out_restored.data, './checksample/' + opt.type + '/output.png')
            save_image(degraded.data, './checksample/' + opt.type + '/degraded.png')
            save_image(target.data, './checksample/' + opt.type + '/target.png')

    return epoch_bary_loss / count, epoch_pot_loss / count


def save_checkpoint(BaryIR, Pots, epoch):
    model_out_path = "checkpoint/" + "model_" + str(opt.type) + opt.backbone + str(opt.patch_size) + "_" + "_" + str(
        opt.nEpochs) + "_" + str(
        opt.sigma) + ".pt"
    
    state = {
        "epoch": epoch, 
        "BaryIR": BaryIR.module if hasattr(BaryIR, "module") else BaryIR, 
        "Pots": Pots.module if hasattr(Pots, "module") else Pots
    }
    if not os.path.exists("checkpoint/"):
        os.makedirs("checkpoint/")

    torch.save(state, model_out_path)
    print("Checkpoint saved to {}".format(model_out_path))


if __name__ == "__main__":
    main()