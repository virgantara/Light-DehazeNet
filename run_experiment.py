import torch
import torch.nn as nn
import torchvision
import torch.backends.cudnn as cudnn
import torch.optim
import os
import sys
import argparse
import time
import image_data_loader
# import lightdehazeNet
import lightdehazeNetLite as ldnet
import numpy as np
from torchvision import transforms
import wandb

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


def train(args):
	wandb.init(project="LightDehazeNet", config=args, name=args['exp_name'])
	ld_net = ldnet.LightDehazeNetLite().cuda()
	ld_net.apply(weights_init)
	wandb.watch(ld_net, log="all", log_freq=10)

	training_data = image_data_loader.hazy_data_loader(args["train_original"],
											 args["train_hazy"])		
	validation_data = image_data_loader.hazy_data_loader(args["train_original"],
											 args["train_hazy"], mode="val")		
	training_data_loader = torch.utils.data.DataLoader(training_data, batch_size=args["batch_size"], shuffle=True, num_workers=4, pin_memory=True)
	validation_data_loader = torch.utils.data.DataLoader(validation_data, batch_size=args["test_batch_size"], shuffle=True, num_workers=4, pin_memory=True)

	criterion = nn.MSELoss().cuda()
	optimizer = torch.optim.Adam(ld_net.parameters(), lr=float(args["learning_rate"]), weight_decay=0.0001)
	
	ld_net.train()

	num_of_epochs = int(args["epochs"])
	for epoch in range(num_of_epochs):
		for iteration, (hazefree_image, hazy_image) in enumerate(training_data_loader):

			hazefree_image = hazefree_image.cuda()
			hazy_image = hazy_image.cuda()

			dehaze_image = ld_net(hazy_image)

			loss = criterion(dehaze_image, hazefree_image)

			optimizer.zero_grad()
			loss.backward()
			torch.nn.utils.clip_grad_norm_(ld_net.parameters(),0.1)
			optimizer.step()

			if ((iteration+1) % 10) == 0:
				wandb.log({"train/loss": loss.item(), "epoch": epoch, "iteration": iteration + 1})
				print("Loss at iteration", iteration+1, ":", loss.item())
			if ((iteration+1) % 200) == 0:
				
				torch.save(ld_net.state_dict(), "trained_weights/" + "Epoch" + str(epoch) + '.pth') 		

		# Validation Stage
		for iter_val, (hazefree_image, hazy_image) in enumerate(validation_data_loader):



			hazefree_image = hazefree_image.cuda()
			hazy_image = hazy_image.cuda()

			dehaze_image = ld_net(hazy_image)

			torchvision.utils.save_image(torch.cat((hazy_image, dehaze_image, hazefree_image),0), "training_data_captures/" +str(iter_val+1)+".jpg")

			if iter_val == 0:
				wandb.log({
				    "val/example": [
				        wandb.Image(torch.cat((hazy_image, dehaze_image, hazefree_image), 0),
				                    caption="Hazy | Dehazed | GT")
				    ],
				    "epoch": epoch
				})
		torch.save(ld_net.state_dict(), "trained_weights/" + "trained_LDNet.pth") 

if __name__ == "__main__":

	ap = argparse.ArgumentParser()
	ap.add_argument("-th", "--train_hazy", required=True, help="path to hazy training images")
	ap.add_argument("-to", "--train_original", required=True, help="path to original training images")
	ap.add_argument("-e", "--epochs", required=True, help="number of epochs for training")
	ap.add_argument("-bs","--batch_size", type=int, default=64, help="batch_size")
	ap.add_argument("-tbs","--test_batch_size", type=int, default=32, help="batch_size")
	ap.add_argument("-en","--exp_name", type=str, default="exp_", help="batch_size")
	ap.add_argument("-lr", "--learning_rate", required=True, help="learning rate for training")
	
	args = vars(ap.parse_args())

	train(args)








	
