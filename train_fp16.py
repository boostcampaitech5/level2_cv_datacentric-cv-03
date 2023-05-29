import os
import os.path as osp
import time
import math
from datetime import timedelta
from argparse import ArgumentParser

import torch
from torch import cuda
from torch.utils.data import DataLoader
from torch.optim import lr_scheduler
from tqdm import tqdm

from east_dataset import EASTDataset
from dataset import SceneTextDataset
from model import EAST

import numpy as np
import random
import wandb
from accelerate import Accelerator

def parse_args():
    parser = ArgumentParser()

    # Conventional args
    parser.add_argument(
        "--data_dir",
        type=str,
        default=os.environ.get("SM_CHANNEL_TRAIN", "../data/medical"),
    )
    parser.add_argument(
        "--model_dir",
        type=str,
        default=os.environ.get("SM_MODEL_DIR", "trained_models"),
    )

    parser.add_argument("--device", default="cuda" if cuda.is_available() else "cpu")
    parser.add_argument("--num_workers", type=int, default=8)

    parser.add_argument("--image_size", type=int, default=2048)
    parser.add_argument("--input_size", type=int, default=1024)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--max_epoch", type=int, default=150)
    parser.add_argument(
        "--ignore_tags",
        type=list,
        default=["masked", "excluded-region", "maintable", "stamp"],
    )

    parser.add_argument("--exp_name", type=str, default="[tag]ExpName_V1")
    parser.add_argument("--seed", type=int, default=3)
    parser.add_argument("--n_save", type=int, default=3)
    parser.add_argument("--split_num", type=int, default=0)
    parser.add_argument("--patience", type=int, default=10)

    args = parser.parse_args()

    if args.input_size % 32 != 0:
        raise ValueError("`input_size` must be a multiple of 32")

    return args


def set_seed(random_seed):
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)  # if use multi-GPU
    # CUDA randomness
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    np.random.seed(random_seed)
    random.seed(random_seed)
    os.environ["PYTHONHASHSEED"] = str(random_seed)


class BestScore:
    """Track performance to store parameters for the best performing models"""

    def __init__(self, n, reverse=False):
        # n: How many are you going to save
        self.n = n
        self.reverse = reverse
        self.change = False
        self.reset()

    def reset(self):
        # [1st(best), 2nd, ...]
        self.metric = dict()

    def update(self, epoch, val, state_dict):
        self.metric[epoch] = {"score": val, "state_dict": state_dict}
        self.metric = sorted(
            self.metric.items(), key=lambda x: x[1]["score"], reverse=self.reverse
        )[: self.n]
        self.metric = dict(self.metric)


class AverageMeter:
    """Computes and stores the average and current value"""

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


def main(
    data_dir,
    model_dir,
    device,
    image_size,
    input_size,
    num_workers,
    batch_size,
    learning_rate,
    max_epoch,
    ignore_tags,
    exp_name,
    seed,
    n_save,
    split_num,
    patience,
):
    set_seed(seed)

    # wandb 초기 설정
    wandb.init(
        name=exp_name,
        project="ocr",
        entity="ganisokay",
        config=args,
    )

    model_dir = osp.join(model_dir, exp_name)
    if not osp.exists(model_dir):
        os.makedirs(model_dir)

    # ======== train dataset loader ========
    train_dataset = SceneTextDataset(
        data_dir,
        split="train",
        num=split_num,
        image_size=image_size,
        crop_size=input_size,
        ignore_tags=ignore_tags,
    )
    train_dataset = EASTDataset(train_dataset)
    train_num_batches = math.ceil(len(train_dataset) / batch_size)
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )

    # ======== val dataset loader ========
    val_dataset = SceneTextDataset(
        data_dir,
        split="val",
        num=split_num,
        image_size=image_size,
        crop_size=input_size,
        ignore_tags=ignore_tags,
    )
    val_dataset = EASTDataset(val_dataset)
    val_num_batches = math.ceil(len(val_dataset) / batch_size)
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )

    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    accelerator = Accelerator(gradient_accumulation_steps=2)
    device = accelerator.device
    model = EAST()
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = lr_scheduler.MultiStepLR(
        optimizer, milestones=[max_epoch // 2], gamma=0.1
    )

    # early stopping
    patience = patience
    counter = 0
    best_val_loss = np.inf
    
    # save best checkpoint(최대 n_save개)
    best_loss = BestScore(n=n_save)
    train_epoch_loss = AverageMeter()
    val_epoch_loss = AverageMeter()

    model, optimizer, train_loader, scheduler = accelerator.prepare(model, optimizer, train_loader, scheduler)
    
    for epoch in range(max_epoch):
        # ======== train ========
        model.train()
        epoch_start = time.time()
        train_epoch_loss.reset()
        with tqdm(total=train_num_batches, disable=not accelerator.is_local_main_process) as pbar:
            for img, gt_score_map, gt_geo_map, roi_mask in train_loader:
                pbar.set_description("[Epoch {}]".format(epoch + 1))

                with accelerator.accumulate(model):
                    optimizer.zero_grad()
                    loss, extra_info = model.train_step(
                    img, gt_score_map, gt_geo_map, roi_mask
                    )
                    
                    accelerator.backward(loss)
                    optimizer.step()

                train_epoch_loss.update(loss.item())

                pbar.update(1)
                train_dict = {
                    "Train Cls loss": extra_info["cls_loss"],
                    "Train Angle loss": extra_info["angle_loss"],
                    "Train IoU loss": extra_info["iou_loss"],
                }
                pbar.set_postfix(train_dict)
                wandb.log(train_dict)

        scheduler.step()

        print(
            "> Train : Mean loss: {:.4f} | Elapsed time: {}".format(
                train_epoch_loss.avg,
                timedelta(seconds=time.time() - epoch_start),
            )
        )

        # ======== val ========
        with torch.no_grad():
            model.eval()
            epoch_start = time.time()
            with tqdm(total=val_num_batches) as pbar:
                for img, gt_score_map, gt_geo_map, roi_mask in val_loader:
                    pbar.set_description('Evaluate..')
                    loss, extra_info = model.train_step(
                        img, gt_score_map, gt_geo_map, roi_mask
                    )
                    val_epoch_loss.update(loss.item())
                    
                    pbar.update(1)
                    val_dict = {
                        "Val Cls loss": extra_info["cls_loss"],
                        "Val Angle loss": extra_info["angle_loss"],
                        "Val IoU loss": extra_info["iou_loss"],
                    }
                    pbar.set_postfix(val_dict)
                    wandb.log(val_dict)

        # val loss 기준으로 best loss 저장
        if val_epoch_loss.avg < best_val_loss:
            ckpt_fpath = osp.join(model_dir, "best.pth")
            torch.save(model.state_dict(), ckpt_fpath)
            print("New best model for val loss : {:.4f}".format(val_epoch_loss.avg))
            best_val_loss = val_epoch_loss.avg
            counter = 0
        else:
            counter += 1
            print("Not Val Update.. Counter : {}".format(counter))

        print(
            "> Val : Mean loss: {:.4f} | Best Val loss: {:.4f} | Elapsed time: {}".format(
                val_epoch_loss.avg, best_val_loss,
                timedelta(seconds=time.time() - epoch_start),
            )
        )
        if counter > patience:
            print("Early Stopping!")
            break
                

        best_loss.update(epoch, val_epoch_loss.avg, model.state_dict())
        
        folder_epoch = set(os.listdir(model_dir))
        best_epoch = set(map(lambda x: str(x) + ".pth", list(best_loss.metric.keys())))

        remove_epoch = list(folder_epoch - best_epoch - set(["latest.pth"]) - set(["best.pth"]))
        add_epoch = list(best_epoch - folder_epoch)

        if remove_epoch:
            os.remove(osp.join(model_dir, remove_epoch[0]))
        if add_epoch:
            ckpt_fpath = osp.join(model_dir, add_epoch[0])
            torch.save(
                best_loss.metric[int(add_epoch[0][:-4])]["state_dict"], ckpt_fpath
            )

        ckpt_fpath = osp.join(model_dir, "latest.pth")
        #-- 나중에 finetuning 할 때 문제 있으면 사용
        # unwrapped_model = accelerator.unwrap_model(model)
        # accelerator.save(unwrapped_model.state_dict(), ckpt_fpath)
        torch.save(model.state_dict(), ckpt_fpath)

        wandb.log(
            {
                "Train Loss": train_epoch_loss.avg,
                # "Train F1-Score": train_f1,
                "Val Loss": val_epoch_loss.avg,
                # "Val F1-Score": val_f1,
            }
        )
    print(timedelta(seconds=time.time() - start_time))

if __name__ == "__main__":
    torch.cuda.empty_cache()
    args = parse_args()
    main(**args.__dict__)
