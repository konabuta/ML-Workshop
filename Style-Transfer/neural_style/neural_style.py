import argparse
import os
import sys
import time
import re

import numpy as np
import torch
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms
import torch.utils.data.distributed
import torch.onnx
import horovod.torch as hvd

import utils
from transformer_net import TransformerNet
from vgg import Vgg16

from azureml.core.run import Run
# get the Azure ML run object
run = Run.get_submitted_run()


def check_paths(args):
    try:
        if not os.path.exists(args.save_model_dir):
            os.makedirs(args.save_model_dir)
        if args.checkpoint_model_dir is not None and not (os.path.exists(args.checkpoint_model_dir)):
            os.makedirs(args.checkpoint_model_dir)
    except OSError as e:
        print(e)
        sys.exit(1)


def train(args):
    device = torch.device("cuda" if args.cuda else "cpu")

    # log content and style weight parameters
    if hvd.rank() == 0:
        run.log('content_weight', np.float(args.content_weight))
        run.log('style_weight', np.float(args.style_weight))

    transform = transforms.Compose([
        transforms.Resize(args.image_size),
        transforms.CenterCrop(args.image_size),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.mul(255))
    ])
    kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}
    train_dataset = datasets.ImageFolder(args.dataset, transform)
    


    # Horovod: partition dataset among workers using DistributedSampler
    train_sampler = torch.utils.data.distributed.DistributedSampler(
        train_dataset, num_replicas=hvd.size(), rank=hvd.rank())

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, 
        sampler=train_sampler, **kwargs)

    transformer = TransformerNet().to(device)

    # Horovod: broadcast parameters from rank 0 to all other processes
    hvd.broadcast_parameters(transformer.state_dict(), root_rank=0)
    # Horovod: scale learning rate by the number of GPUs
    optimizer = Adam(transformer.parameters(), args.lr * hvd.size())
    # Horovod: wrap optimizer with DistributedOptimizer
    optimizer = hvd.DistributedOptimizer(optimizer, 
        named_parameters=transformer.named_parameters())
    mse_loss = torch.nn.MSELoss()

    vgg = Vgg16(requires_grad=False).to(device)
    style_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.mul(255))
    ])
    style = utils.load_image(args.style_image, size=args.style_size)
    style = style_transform(style)
    style = style.repeat(args.batch_size, 1, 1, 1).to(device)

    features_style = vgg(utils.normalize_batch(style))
    gram_style = [utils.gram_matrix(y) for y in features_style]

    print("starting training...")
    for e in range(args.epochs):
        print("epoch {}...".format(e))
        transformer.train()
        agg_content_loss = 0.
        agg_style_loss = 0.
        count = 0
        for batch_id, (x, _) in enumerate(train_loader):
            n_batch = len(x)
            count += n_batch
            optimizer.zero_grad()

            x = x.to(device)
            y = transformer(x)

            y = utils.normalize_batch(y)
            x = utils.normalize_batch(x)

            features_y = vgg(y)
            features_x = vgg(x)

            content_loss = args.content_weight * mse_loss(features_y.relu2_2, features_x.relu2_2)

            style_loss = 0.
            for ft_y, gm_s in zip(features_y, gram_style):
                gm_y = utils.gram_matrix(ft_y)
                style_loss += mse_loss(gm_y, gm_s[:n_batch, :, :])
            style_loss *= args.style_weight

            total_loss = content_loss + style_loss
            total_loss.backward()
            optimizer.step()

            agg_content_loss += content_loss.item()
            agg_style_loss += style_loss.item()

            if (batch_id + 1) % args.log_interval == 0:
                avg_content_loss = agg_content_loss / (batch_id + 1)
                avg_style_loss = agg_style_loss / (batch_id + 1)
                avg_total_loss = (agg_content_loss + agg_style_loss) / (batch_id + 1)
                mesg = "{}\tEpoch {}:\t[{}/{}]\tcontent: {:.6f}\tstyle: {:.6f}\ttotal: {:.6f}".format(
                    time.ctime(), e + 1, count, len(train_sampler),
                                  avg_content_loss,
                                  avg_style_loss,
                                  avg_total_loss
                )
                print(mesg)

                # log the losses the run history
                run.log('avg_content_loss', np.float(avg_content_loss))
                run.log('avg_style_loss', np.float(avg_style_loss))
                run.log('avg_total_loss', np.float(avg_total_loss))


            if hvd.rank() == 0 and args.checkpoint_model_dir is not None and (batch_id + 1) % args.checkpoint_interval == 0:
                transformer.eval().cpu()
                ckpt_model_filename = "ckpt_epoch_" + str(e) + "_batch_id_" + str(batch_id + 1) + ".pth"
                ckpt_model_path = os.path.join(args.checkpoint_model_dir, ckpt_model_filename)
                torch.save(transformer.state_dict(), ckpt_model_path)
                transformer.to(device).train()

    # save model
    if hvd.rank() == 0:
        transformer.eval().cpu()
        if args.export_to_onnx:
            # export model to ONNX format
            dummy_input = torch.randn(1,3, 1024, 1024, device='cpu')
            save_model_path = os.path.join(args.save_model_dir, '{}.onnx'.format(args.model_name))
            torch.onnx.export(transformer, dummy_input, save_model_path)
        else:
            save_model_path = os.path.join(args.save_model_dir, '{}.pth'.format(args.model_name))
            torch.save(transformer.state_dict(), save_model_path)

        print("\nDone, trained model saved at", save_model_path)

def main():
    parser = argparse.ArgumentParser(description="parser for fast-neural-style")

    parser.add_argument("--epochs", type=int, default=2,
                                  help="number of training epochs, default is 2")
    parser.add_argument("--batch-size", type=int, default=4,
                                  help="batch size for training, default is 4")
    parser.add_argument("--dataset", type=str, required=True,
                                  help="path to training dataset, the path should point to a folder "
                                       "containing another folder with all the training images")
    parser.add_argument("--style-image", type=str, default="images/style-images/mosaic.jpg",
                                  help="path to style-image")
    parser.add_argument("--save-model-dir", type=str, required=True,
                                  help="path to folder where trained model will be saved.")
    parser.add_argument("--checkpoint-model-dir", type=str, default=None,
                                  help="path to folder where checkpoints of trained models will be saved")
    parser.add_argument("--image-size", type=int, default=256,
                                  help="size of training images, default is 256 X 256")
    parser.add_argument("--style-size", type=int, default=None,
                                  help="size of style-image, default is the original size of style image")
    parser.add_argument("--cuda", type=int, required=True,
                                  help="set it to 1 for running on GPU, 0 for CPU")
    parser.add_argument("--seed", type=int, default=42,
                                  help="random seed for training")
    parser.add_argument("--content-weight", type=float, default=1e5,
                                  help="weight for content-loss, default is 1e5")
    parser.add_argument("--style-weight", type=float, default=1e10,
                                  help="weight for style-loss, default is 1e10")
    parser.add_argument("--lr", type=float, default=1e-3,
                                  help="learning rate, default is 1e-3")
    parser.add_argument("--log-interval", type=int, default=500,
                                  help="number of images after which the training loss is logged, default is 500")
    parser.add_argument("--checkpoint-interval", type=int, default=2000,
                                  help="number of batches after which a checkpoint of the trained model will be created")
    parser.add_argument("--export-to-onnx", type=bool, default=False, help='export model to ONNX format')

    parser.add_argument("--model-name", type=str, default="model.pth",
                                  help="name for saved model")

    args = parser.parse_args()

    if args.cuda and not torch.cuda.is_available():
        print("ERROR: cuda is not available, try running on CPU")
        sys.exit(1)
        
    ## debug
    print("### train dataset ###",args.dataset)
    import os
    ls_file_name = os.listdir(args.dataset)
    print(ls_file_name)

    # Horovod: initialize
    hvd.init()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Horovod: pin GPU to local rank (one GPU per process)
    if args.cuda:
        torch.cuda.set_device(hvd.local_rank())
        torch.cuda.manual_seed(args.seed)

    if hvd.rank() == 0:
        check_paths(args)
    
    train(args)


if __name__ == "__main__":
    main()
