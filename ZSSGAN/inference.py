import os
import math
import numpy as np
import cv2
import torch
import torchvision.transforms as transforms
import imageio
from PIL import Image
from typer import edit

from utils.file_utils import save_images
from options.train_options import TrainOptions
from model.sg2_model import Generator
from model.psp import pSp
from model.datasetgan import pixel_classifier
from utils.data_utils import *

SAVE_SRC = True
SAVE_DST = True
device = 'cuda'
# set seed after all networks have been initialized. Avoids change of outputs due to model changes.
torch.manual_seed(2)
np.random.seed(2)

args = TrainOptions().parse()
    
dataset_size = {
    'ffhq': 1024,
    'cat': 512,
    'dog': 512,
    'church': 256,
    'horse': 256,
    'car': 512,
}
palette_dict = {
    'ffhq': face_palette,
    'car': car_32_palette,
}

class_num = {
    'ffhq': 34,
    'car': 32,
}

class_dim = {
    'ffhq': 5888,
}
args.size = dataset_size[args.dataset]

# Make output directory
args.output_dir = 'Image_edit/'
os.makedirs(args.output_dir, exist_ok=True)

# Make preprocess
resize = (192, 256) if args.dataset == 'car' else (256, 256)
preprocess = transforms.Compose([transforms.Resize(resize),
                            transforms.ToTensor(),
                            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])


def get_avg_image(net):
    avg_image = net(net.latent_avg.unsqueeze(0),
                    input_code=True,
                    randomize_noise=False,
                    return_latents=False,
                    average_code=True)[0]
    avg_image = avg_image.to('cuda').float().detach()
    return avg_image


def get_transfer_image(img_path, align_img=None, edit_code=None):
    psp_encoder = pSp(args.psp_path, device, args.size, has_decoder=True)
    psp_encoder.to(device)
    psp_encoder.requires_grad_(False)

    net = Generator(args.size, style_dim=512, n_mlp=8,)
    net.to(device)

    checkpoint = torch.load(args.frozen_gen_ckpt, map_location=device)
    net.load_state_dict(checkpoint['g_ema'], strict=True)

    if align_img is not None:
        img = align_img
    else:
        img = Image.open(img_path).convert("RGB")
    img = preprocess(img).unsqueeze(0).to(device).float()

    with torch.no_grad():
        net.eval()
        psp_encoder.eval()
        code, invert_img = psp_encoder(img)
        # import pdb
        # pdb.set_trace()
        if edit_code is not None:
            code = code + edit_code
        trans_img, _ = net([code],
                        input_is_latent=True,
                        randomize_noise=True,
                        return_latents=True)
    prefix = os.path.basename(img_path).split(".")[0]
    save_images(invert_img, args.output_dir, f'{prefix}_invert', 1, 1)
    save_images(trans_img, args.output_dir, f'{prefix}_transfer', 1, 1)


def get_inversion_img(args, img_path):
    psp_encoder = pSp(args.psp_path, device, args.size, has_decoder=True)
    psp_encoder.to(device)
    psp_encoder.requires_grad_(False)
    img = Image.open(img_path).convert("RGB")
    img = preprocess(img).unsqueeze(0).to(device).float()
    with torch.no_grad():
        code, invert_img = psp_encoder(img)
    save_images(invert_img, args.output_dir, 'invert', 1, 1)


def get_mask(img_path):
    psp_encoder = pSp(args.psp_path, device, args.size, has_decoder=True)
    psp_encoder.to(device)

    classifier = pixel_classifier(model_path="../weights/model_1.pth", \
        numpy_class=class_num[args.dataset], dim=class_dim[args.dataset])
    classifier.to(device)

    img = Image.open(img_path).convert("RGB")
    img = preprocess(img).unsqueeze(0).to(device).float()
    num_feat = int(math.log(class_dim[args.dataset], 2)) * 2 - 2
    with torch.no_grad():
        code, _ = psp_encoder(img)
        invert_img, features = psp_encoder.decoder.g_synthesis(code, randomize_noise=True, num_feat=num_feat)
        mask = classifier(features)
    
    vis_mask = colorize_mask(mask[0], palette_dict[args.dataset])
    invert_img = invert_img[0].add_(1).mul(0.5*255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()
    vis_mask = cv2.resize(vis_mask, invert_img.shape[0:2], cv2.INTER_NEAREST)
    orig_img = cv2.resize(cv2.imread(img_path), invert_img.shape[0:2], cv2.INTER_CUBIC)[:, :, [2,1,0]]
    prefix = os.path.basename(img_path).split(".")[0]
    vis_mask = vis_mask * 0.5 + invert_img * 0.5
    orig_img = vis_mask * 0.7 + orig_img * 0.3
    vis_mask = np.uint8(vis_mask)
    orig_img = np.uint8(orig_img)
    imageio.imsave(os.path.join(args.output_dir, f'{prefix}_invert.jpg'), vis_mask)
    imageio.imsave(os.path.join(args.output_dir, f'{prefix}_orig.jpg'), orig_img)
    # imageio.imsave(os.path.join(args.output_dir, f'{prefix}_invert.jpg'), invert_img)

def run_alignment(image_path):
    import dlib
    from utils.align_faces_parallel import align_face
    if not os.path.exists("shape_predictor_68_face_landmarks.dat"):
        print('Downloading files for aligning face image ...')
        os.system('wget http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2')
        os.system('bzip2 -dk shape_predictor_68_face_landmarks.dat.bz2')
        print('Done.')
    predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
    aligned_image = align_face(filepath=image_path, predictor=predictor) 
    print("Aligned image has shape: {}".format(aligned_image.size))
    return aligned_image 

if __name__ == "__main__":
    image_path = args.real_image_path
    input_image = run_alignment(image_path)
    get_transfer_image(image_path, input_image)

