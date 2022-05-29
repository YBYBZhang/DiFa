"""
This file defines the core research contribution
"""
from argparse import Namespace
from tokenize import Name
import matplotlib
matplotlib.use('Agg')
import math

import torch
from torch import nn
from model.encoders import psp_encoders
from model.sg2_model import Generator


def get_keys(d, name):
	if 'state_dict' in d:
		d = d['state_dict']
	d_filt = {k[len(name) + 1:]: v for k, v in d.items() if k[:len(name)] == name}
	return d_filt


class pSp(nn.Module):

	def __init__(self, checkpoint_path, device, output_size=1024, has_decoder=False):
		super(pSp, self).__init__()
		self.opts = self.set_opts(checkpoint_path)
		self.has_decoder = has_decoder
		self.device = device
		# compute number of style inputs based on the output resolution
		self.opts.n_styles = int(math.log(output_size, 2)) * 2 - 2
		# Define architecture
		self.encoder = self.set_encoder()
		if self.has_decoder:
			self.decoder = Generator(output_size, 512, 8)
		# Load weights if needed
		self.load_weights()
		self.psp_encoder = 'e4e' if self.opts.encoder_type == 'Encoder4Editing' else 'psp'

	def set_encoder(self):
		if self.opts.encoder_type == 'GradualStyleEncoder':
			encoder = psp_encoders.GradualStyleEncoder(50, 'ir_se', self.opts)
		elif self.opts.encoder_type == 'BackboneEncoderUsingLastLayerIntoW':
			encoder = psp_encoders.BackboneEncoderUsingLastLayerIntoW(50, 'ir_se', self.opts)
		elif self.opts.encoder_type == 'BackboneEncoderUsingLastLayerIntoWPlus':
			encoder = psp_encoders.BackboneEncoderUsingLastLayerIntoWPlus(50, 'ir_se', self.opts)
		elif self.opts.encoder_type == 'Encoder4Editing':
			encoder = psp_encoders.Encoder4Editing(50, 'ir_se', self.opts)
		else:
			raise Exception('{} is not a valid encoders'.format(self.opts.encoder_type))
		return encoder

	def load_weights(self):
		if self.opts.checkpoint_path is not None:
			print('Loading pSp from checkpoint: {}'.format(self.opts.checkpoint_path))
			ckpt = torch.load(self.opts.checkpoint_path, map_location='cpu')
			self.encoder.load_state_dict(get_keys(ckpt, 'encoder'), strict=True)
			if self.has_decoder:
				self.decoder.load_state_dict(get_keys(ckpt, 'decoder'), strict=True)
			self.__load_latent_avg(ckpt)
		else:
			raise RuntimeError(f"There isn't psp encoder in {self.opts.checkpoint_path}")

	def forward(self, x, randomize_noise=True):
		codes = self.encoder(x)
		# normalize with respect to the center of an average face
		# if self.opts.start_from_latent_avg:
		# 	if self.opts.learn_in_w:
		# 		codes = codes + self.latent_avg.repeat(codes.shape[0], 1)
		# 	else:
		codes = codes + self.latent_avg.repeat(codes.shape[0], 1, 1)

		if self.has_decoder:
			images, result_latent = self.decoder([codes],
		                                     input_is_latent=True,
		                                     randomize_noise=randomize_noise,
		                                     return_latents=True)
		else:
			result_latent = codes

		# if resize:
		# 	images = self.face_pool(images)

		if self.has_decoder:
			return result_latent, images
		else:
			return result_latent, None

	def set_opts(self, opts_path):
		opts = torch.load(opts_path, map_location='cpu')['opts']
		opts['checkpoint_path'] = opts_path
		opts = Namespace(**opts)
		return opts

	def __load_latent_avg(self, ckpt, repeat=None):
		if 'latent_avg' in ckpt:
			self.latent_avg = ckpt['latent_avg'].to(self.device)
			if repeat is not None:
				self.latent_avg = self.latent_avg.repeat(repeat, 1)
		else:
			self.latent_avg = None
