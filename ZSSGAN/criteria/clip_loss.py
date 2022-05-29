import pickle
import torch
import torchvision.transforms as transforms
import torch.nn.functional as F

import numpy as np
import os
import tqdm

import clip
from PIL import Image


from ZSSGAN.utils.text_templates import imagenet_templates, part_templates, imagenet_templates_small


class DirectionLoss(torch.nn.Module):

    def __init__(self, loss_type='mse'):
        super(DirectionLoss, self).__init__()

        self.loss_type = loss_type

        self.loss_func = {
            'mse':    torch.nn.MSELoss,
            'cosine': torch.nn.CosineSimilarity,
            'mae':    torch.nn.L1Loss
        }[loss_type]()

    def forward(self, x, y):
        if self.loss_type == "cosine":
            return 1. - self.loss_func(x, y)
        
        return self.loss_func(x, y)

class CLIPLoss(torch.nn.Module):
    def __init__(self, device, lambda_direction=1., lambda_patch=0., lambda_global=0., \
        lambda_manifold=0., lambda_texture=0., patch_loss_type='mae', \
            direction_loss_type='cosine', clip_model='ViT-B/32', args=None):
        super(CLIPLoss, self).__init__()

        self.device = device
        self.args = args
        self.model_name = clip_model
        self.model, clip_preprocess = clip.load(clip_model, device=self.device)
        self.visual_model = self.model.visual

        # self.model.requires_grad_(False)
        self.clip_preprocess = clip_preprocess
        
        # self.preprocess = transforms.Compose([transforms.Normalize(mean=[-1.0, -1.0, -1.0], std=[2.0, 2.0, 2.0])] + # Un-normalize from [-1.0, 1.0] (GAN output) to [0, 1].
        #                                       clip_preprocess.transforms[:2] +                                      # to match CLIP input scale assumptions
        #                                       clip_preprocess.transforms[4:])                                       # + skip convert PIL to tensor
        self.preprocess = transforms.Compose([
            transforms.Normalize(mean=[-1.0, -1.0, -1.0], std=[2.0, 2.0, 2.0]),
            transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
        ])
        self.target_direction      = None
        self.target_image          = None
        self.target_text           = None
        self.patch_text_directions = None
        self.target_beta = 0.005

        self.patch_loss     = DirectionLoss(patch_loss_type)
        self.direction_loss = DirectionLoss(direction_loss_type)
        self.patch_direction_loss = torch.nn.CosineSimilarity(dim=2)

        self.lambda_global    = lambda_global
        self.lambda_patch     = lambda_patch
        self.lambda_direction = lambda_direction
        self.lambda_manifold  = lambda_manifold
        self.lambda_texture   = lambda_texture
        self.lambda_partial   = self.args.lambda_partial
        self.alpha = args.alpha

        self.src_text_features = None
        self.target_text_features = None
        self.angle_loss = torch.nn.L1Loss()
        self.id_loss = DirectionLoss('cosine')

        # self.model_cnn, preprocess_cnn = clip.load("RN50", device=self.device)
        # self.preprocess_cnn = transforms.Compose([transforms.Normalize(mean=[-1.0, -1.0, -1.0], std=[2.0, 2.0, 2.0])] + # Un-normalize from [-1.0, 1.0] (GAN output) to [0, 1].
        #                                 preprocess_cnn.transforms[:2] +                                                 # to match CLIP input scale assumptions
        #                                 preprocess_cnn.transforms[4:])                                                  # + skip convert PIL to tensor

        self.texture_loss = torch.nn.MSELoss()
        self.condition = None
        self.clip_mean = self.get_clip_mean()

        self.target_keys = []
        self.target_tokens = []
        if self.lambda_partial > 0:
            self.hook_handlers = []
            self.feat_keys = []
            self.feat_tokens = []
            self.gen_attn_weights = []
            self._register_hooks(layer_ids=[args.clip_layer], facet='key')

    
    def _get_hook(self, facet):
        if facet in ['token']:
            def _hook(model, input, output):
                input = model.ln_1(input[0])
                attnmap = model.attn(input, input, input, need_weights=True, attn_mask=model.attn_mask)[1]
                self.feat_tokens.append(output[1:].permute(1, 0, 2))
                self.gen_attn_weights.append(attnmap)
            return _hook
        elif facet == 'feat':
            def _outer_hook(model, input, output):
                output = output[1:].permute(1, 0, 2)  # LxBxD -> BxLxD
                # TODO: Remember to add VisualTransformer ln_post, i.e. LayerNorm
                output = F.layer_norm(output, self.visual_model.ln_post.normalized_shape, \
                    self.visual_model.ln_post.weight.type(output.dtype), \
                    self.visual_model.ln_post.bias.type(output.dtype), \
                        self.visual_model.ln_post.eps)
                output = output @ self.visual_model.proj
                self.feat_tokens.append(output)
            return _outer_hook

        if facet == 'query':
            facet_idx = 0
        elif facet == 'key':
            facet_idx = 1
        elif facet == 'value':
            facet_idx = 2
        else:
            raise TypeError(f"{facet} is not a supported facet.")

        def _inner_hook(module, input, output):
            input = input[0]
            N, B, C = input.shape
            weight = module.in_proj_weight.detach()
            bias = module.in_proj_bias.detach()
            qkv = F.linear(input, weight, bias)[1:]  # remove cls key
            qkv = qkv.reshape(-1, B, 3, C).permute(2, 1, 0, 3)  # BxNxC
            self.feat_keys.append(qkv[facet_idx])
        return _inner_hook
    
    def _register_hooks(self, layer_ids=[11], facet='key'):
        for block_idx, block in enumerate(self.visual_model.transformer.resblocks):
            if block_idx in layer_ids:
                self.hook_handlers.append(block.register_forward_hook(self._get_hook('token')))
                assert facet in ['key', 'query', 'value']
                self.hook_handlers.append(block.attn.register_forward_hook(self._get_hook(facet)))
    
    def _unregister_hooks(self) -> None:
        """
        unregisters the hooks. should be called after feature extraction.
        """
        for handle in self.hook_handlers:
            handle.remove()
        self.hook_handlers = []

    def get_clip_mean(self):
        orig_sample_path = os.path.join('../weights/clip_mean/', f"{self.args.dataset}_{self.model_name[-2::]}_samples.pkl")
        if not os.path.exists(orig_sample_path):
            X = np.zeros((1, 512))
        else:
            print("Load from ", orig_sample_path)
            with open(orig_sample_path, 'rb') as f:
                X = pickle.load(f)
                X = np.array(X)
        self.samples = X
        mean =  torch.from_numpy(np.mean(X, axis=0)).float().to(self.device)
        # return mean / mean.norm(dim=-1, keepdim=True)
        return mean

    def tokenize(self, strings: list):
        return clip.tokenize(strings).to(self.device)

    def encode_text(self, tokens: list) -> torch.Tensor:
        return self.model.encode_text(tokens)

    def encode_images(self, images: torch.Tensor) -> torch.Tensor:
        images = self.preprocess(images).to(self.device)
        return self.model.encode_image(images)
        
    def encode_images_with_cnn(self, images: torch.Tensor) -> torch.Tensor:
        images = self.preprocess_cnn(images).to(self.device)
        return self.model_cnn.encode_image(images)
    
    def distance_with_templates(self, img: torch.Tensor, class_str: str, templates=imagenet_templates) -> torch.Tensor:

        text_features  = self.get_text_features(class_str, templates)
        image_features = self.get_image_features(img)

        similarity = image_features @ text_features.T

        return 1. - similarity
    
    def get_text_features(self, class_str: str, templates=imagenet_templates, norm: bool = True) -> torch.Tensor:
        template_text = self.compose_text_with_templates(class_str, templates)

        tokens = clip.tokenize(template_text).to(self.device)

        text_features = self.encode_text(tokens).detach()

        if norm:
            text_features /= text_features.norm(dim=-1, keepdim=True)

        return text_features

    def get_image_features(self, img: torch.Tensor, norm: bool = True) -> torch.Tensor:
        if self.lambda_partial > 0:
            self.feat_keys = []
            self.feat_tokens = []
            self.gen_attn_weights = []

        image_features = self.encode_images(img)
        
        if norm:
            image_features /= image_features.clone().norm(dim=-1, keepdim=True)

        return image_features

    def get_similar_img(self, tgt_vec):
        tgt = tgt_vec[0].cpu().numpy()
        sim = np.dot(self.samples, tgt)
        orders = np.argsort(sim)[::-1]
        print("Orders: {}, Similarities: {}".format(orders[0:20], sim[orders[0:20]]))
        src = self.samples[orders[0:1]]
        src = src * sim[orders[0:1], None]
        src = torch.from_numpy(src).to(tgt_vec.device, dtype=tgt_vec.dtype).mean(axis=0, keepdim=True)
        # src /= src.norm(dim=-1, keepdim=True)
        return src

    def get_prompt_features(self, word_num=50):
        '''
        Get text features of source class based on the samples of Domain A
        '''
        prompt_path = f"../weights/source_prompt/ViT_{self.model_name[-2::]}_{self.args.dataset}_{word_num}.pkl"
        if os.path.exists(prompt_path):
            print("Load the prompt of source class from {} !".format(prompt_path))
            with open(prompt_path, 'rb') as f:
                prompt_words = pickle.load(f)
        else:
            os.makedirs("../weights/source_prompt", exist_ok=True)

            words = set(clip.simple_tokenizer.SimpleTokenizer().encoder.keys())
            # Remove sub-words and keep intact words
            words = {w.replace("</w>", "") for w in words}
            word_list = [w for w in words if len(w) == len(w.encode())]
            word_temps = ["a photo of {}".format(w) for w in word_list]

            clip_mean = self.clip_mean / self.clip_mean.norm(dim=-1, keepdim=True)
            clip_mean = clip_mean.to(torch.float16)

            batch = 100
            sim_all = []
            for i in tqdm.tqdm(range(len(word_list) // batch), desc='Prompt'):
                temps = word_temps[i*batch:(i+1)*batch]
                tokens = clip.tokenize(temps).to(self.device)
                vecs = self.model.encode_text(tokens)
                vecs /= vecs.norm(dim=-1, keepdim=True)
                sim = vecs @ clip_mean.t()
                sim_all.extend(sim.flatten().tolist())

            sim_all = np.array(sim_all)
            sort_ids = np.argsort(sim_all)[::-1].tolist()
            prompt_words = np.array(word_list)[sort_ids[0:word_num]]

            with open(prompt_path, 'wb') as f:
                pickle.dump(prompt_words, f)
                print("Save the prompt of source class to {} !".format(prompt_path))
        print("Prompt words of source class:")
        print(prompt_words)
        source_features = []
        for w in prompt_words:
            vecs = self.get_text_features(w)
            source_features.append(vecs)
        prompt_vec = torch.stack(source_features, dim=0).mean(dim=0)
        prompt_vec /= prompt_vec.norm(dim=-1, keepdim=True)
        return prompt_vec

    def get_online_prompt_features(self, word_num=50):
        prompt_path = os.path.join(self.args.output_dir, f"ViT_{self.model_name[-2::]}_source_clip_mean.pkl")
        words = set(clip.simple_tokenizer.SimpleTokenizer().encoder.keys())
        # Remove sub-words and keep intact words
        words = {w.replace("</w>", "") for w in words}
        word_list = [w for w in words if len(w) == len(w.encode())]
        word_temps = ["a photo of {}".format(w) for w in word_list]

        clip_mean = self.args.online_clip_mean[self.model_name]
        clip_mean /= clip_mean.norm(dim=-1, keepdim=True)
        clip_mean = clip_mean.to(torch.float16)

        batch = 100
        sim_all = []
        for i in tqdm.tqdm(range(len(word_list) // batch), desc='Prompt'):
            temps = word_temps[i*batch:(i+1)*batch]
            tokens = clip.tokenize(temps).to(self.device)
            vecs = self.model.encode_text(tokens)
            vecs /= vecs.norm(dim=-1, keepdim=True)
            sim = vecs @ clip_mean.t()
            sim_all.extend(sim.flatten().tolist())

        sim_all = np.array(sim_all)
        sort_ids = np.argsort(sim_all)[::-1].tolist()
        prompt_words = np.array(word_list)[sort_ids[0:word_num]]
        with open(prompt_path, 'wb') as f:
                pickle.dump(prompt_words, f)
                print("Save the prompt of source class to {} !".format(prompt_path))
        print("Prompt words of source class:")
        print(prompt_words)
        source_features = []
        for w in prompt_words:
            vecs = self.get_text_features(w)
            source_features.append(vecs)
        prompt_vec = torch.stack(source_features, dim=0).mean(dim=0)
        prompt_vec /= prompt_vec.norm(dim=-1, keepdim=True)
        return prompt_vec

    def compute_text_direction(self, source_class: str, target_class: str) -> torch.Tensor:
        # TODO: Remove it
        target_features = self.get_text_features(target_class)
        self.target_text = target_features
        
        if self.args.source_type == 'source':
            source_features = self.get_text_features(source_class)
            print(source_features.mean(0) @ target_features.mean(0).t())
            text_direction = (target_features - source_features).mean(axis=0, keepdim=True)
            # text_direction = target_features.mean(axis=0, keepdim=True)
            text_direction /= text_direction.norm(dim=-1, keepdim=True)
        elif self.args.source_type == 'project':
            source_features = self.clip_mean.to(target_features.dtype).view(1, -1)
            # target_features = target_features.mean(axis=0, keepdim=True)
            target_features /= target_features.norm(dim=-1, keepdim=True)
            source_features /= source_features.norm(dim=-1, keepdim=True)
            sim = target_features @ source_features.t()
            text_direction = (target_features - sim * source_features).mean(axis=0, keepdim=True)
            text_direction /= text_direction.norm(dim=-1, keepdim=True)
        elif self.args.source_type == 'zero':
            target_features = target_features.mean(axis=0, keepdim=True)
            text_direction = target_features / target_features.norm(dim=-1, keepdim=True)
        elif self.args.source_type == 'mean':
            source_features = self.clip_mean
            text_direction = (target_features - source_features).mean(axis=0, keepdim=True)
            # text_direction = target_features.mean(axis=0, keepdim=True)
            text_direction /= text_direction.norm(dim=-1, keepdim=True)
        elif self.args.source_type == 'prompt':
            source_features = self.get_prompt_features()
            text_direction = (target_features - source_features).mean(axis=0, keepdim=True)
            # text_direction = target_features.mean(axis=0, keepdim=True)
            text_direction /= text_direction.norm(dim=-1, keepdim=True)
        elif self.args.source_type == 'online-prompt':
            source_features = self.get_online_prompt_features()
            text_direction = (target_features - source_features).mean(axis=0, keepdim=True)
            # text_direction = target_features.mean(axis=0, keepdim=True)
            text_direction /= text_direction.norm(dim=-1, keepdim=True)
        elif self.args.source_type == 'chosen':
            source_features = self.get_prompt_features().mean(axis=0)
            target_features = target_features.mean(axis=0)
            sort_ids = source_features.abs().argsort()
            select_ids = sort_ids[-int(0.3 * len(sort_ids)):]
            target_features[select_ids] = 0
            text_direction = target_features / target_features.norm()
            text_direction = text_direction.view(1, -1)
        elif self.args.source_type == 'prompt-project':
            source_features = self.get_prompt_features()
            sim_m = target_features @ source_features.t()
            sim = torch.diag(sim_m).view(-1, 1)
            print(sim)
            print(source_features.mean(0) @ target_features.mean(0).t())
            text_direction = (target_features - sim * source_features).mean(axis=0, keepdim=True)
            text_direction /= text_direction.norm(dim=-1, keepdim=True)
        else:
            raise RuntimeError("No type named {}".format(self.args.source_type))
        return text_direction

    def get_raw_img_features(self, imgs: str):
        pre_i = self.clip_preprocess(Image.open(imgs)).unsqueeze(0).to(self.device)
        encoding = self.model.encode_image(pre_i)
        encoding /= encoding.norm(dim=-1, keepdim=True)
        return encoding

    def compute_img2img_direction(self, source_images: torch.Tensor, target_images: list) -> torch.Tensor:
        key_list = []
        token_list = []
        with torch.no_grad():
            target_encodings = []
            for target_img in target_images:
                preprocessed = self.clip_preprocess(Image.open(target_img)).unsqueeze(0).to(self.device)
                
                encoding = self.model.encode_image(preprocessed)
                encoding /= encoding.norm(dim=-1, keepdim=True)

                target_encodings.append(encoding)
                if self.lambda_partial > 0:
                    key_list.append(self.feat_keys[0])
                    token_list.append(self.feat_tokens[0])
            
            target_encoding = torch.cat(target_encodings, axis=0)
            target_encoding = target_encoding.mean(dim=0, keepdim=True)
            target_encoding /= target_encoding.norm(dim=-1, keepdim=True)

            if self.lambda_partial > 0:
                self.target_keys = torch.cat(key_list, dim=0).mean(dim=0, keepdim=True)
                # self.target_keys = target_key / target_key.norm(dim=-1, keepdim=True)
                # self.target_keys = self.target_keys.repeat(self.args.batch, 1, 1)
                self.target_tokens = torch.cat(token_list, dim=0).mean(dim=0, keepdim=True)
                # self.target_tokens = target_token / target_token.norm(dim=-1, keepdim=True)
                # remove_back = self.gen_attn_weights[0][:, 0, 1:].unsqueeze(-1)
                # self.target_tokens = self.target_tokens * remove_back
                # self.target_tokens = self.target_tokens.repeat(self.args.batch, 1, 1)

            self.target_image = target_encoding

            # Option 1: Global mean
            src_encoding = self.clip_mean

            # Option 2: mini-batch mean
            if self.args.model_type == 'nada':
                src_encoding = self.get_image_features(source_images)
                src_encoding = src_encoding.mean(dim=0, keepdim=True)
            
            # Given src image
            # src_path = "../results/tmp/12_invert_000001.jpg"
            # src_preprocessed = self.clip_preprocess(Image.open(src_path)).unsqueeze(0).to(self.device)    
            # src_encoding = self.model.encode_image(src_preprocessed)
            # src_encoding /= src_encoding.norm(dim=-1, keepdim=True)

            direction = target_encoding - src_encoding
            direction /= direction.norm(dim=-1, keepdim=True)

        return direction

    def compute_corresponding_img2img_direction(self, source_images: list, target_images: list) -> torch.Tensor:
        with torch.no_grad():
            target_encodings = []
            for target_img in target_images:
                preprocessed = self.clip_preprocess(Image.open(target_img)).unsqueeze(0).to(self.device)
                
                encoding = self.model.encode_image(preprocessed)
                encoding /= encoding.norm(dim=-1, keepdim=True)

                target_encodings.append(encoding)
            
            target_encoding = torch.cat(target_encodings, axis=0)
            target_encoding = target_encoding.mean(dim=0, keepdim=True)

            source_encodings = []
            for source_img in source_images:
                preprocessed = self.clip_preprocess(Image.open(source_img)).unsqueeze(0).to(self.device)
                
                encoding = self.model.encode_image(preprocessed)
                encoding /= encoding.norm(dim=-1, keepdim=True)

                source_encodings.append(encoding)
            
            source_encoding = torch.cat(source_encodings, axis=0)
            source_encoding = source_encoding.mean(dim=0, keepdim=True)
            direction = target_encoding - source_encoding
            direction /= direction.norm(dim=-1, keepdim=True)
        return direction

    def set_text_features(self, source_class: str, target_class: str) -> None:
        source_features = self.get_text_features(source_class).mean(axis=0, keepdim=True)
        self.src_text_features = source_features / source_features.norm(dim=-1, keepdim=True)

        target_features = self.get_text_features(target_class).mean(axis=0, keepdim=True)
        self.target_text_features = target_features / target_features.norm(dim=-1, keepdim=True)

    def clip_angle_loss(self, src_img: torch.Tensor, source_class: str, target_img: torch.Tensor, target_class: str) -> torch.Tensor:
        if self.src_text_features is None:
            self.set_text_features(source_class, target_class)

        cos_text_angle = self.target_text_features @ self.src_text_features.T
        text_angle = torch.acos(cos_text_angle)

        src_img_features = self.get_image_features(src_img).unsqueeze(2)
        target_img_features = self.get_image_features(target_img).unsqueeze(1)

        cos_img_angle = torch.clamp(target_img_features @ src_img_features, min=-1.0, max=1.0)
        img_angle = torch.acos(cos_img_angle)

        text_angle = text_angle.unsqueeze(0).repeat(img_angle.size()[0], 1, 1)
        cos_text_angle = cos_text_angle.unsqueeze(0).repeat(img_angle.size()[0], 1, 1)

        return self.angle_loss(cos_img_angle, cos_text_angle)

    def compose_text_with_templates(self, text: str, templates=imagenet_templates) -> list:
        return [template.format(text) for template in templates]
   
    def update_text_direction(self, target_encoding: torch.Tensor, source_class: str) -> torch.Tensor:
        self.target_text = self.target_beta * target_encoding.detach().mean(dim=0, keepdim=True) + \
            self.target_text * (1 - self.target_beta)
        
        text_direction = (self.target_text- self.clip_mean).mean(axis=0, keepdim=True)
        text_direction /= text_direction.norm(dim=-1, keepdim=True)
        return text_direction

    def remd_loss(self, tgt_tokens, style_tokens):
        '''
        REMD Loss referring to style transfer
        '''
        tgt_tokens /= tgt_tokens.clone().norm(dim=-1, keepdim=True)
        style_tokens /= style_tokens.clone().norm(dim=-1, keepdim=True)

        attn_weights = torch.bmm(tgt_tokens, style_tokens.permute(0, 2, 1))

        cost_matrix = 1 - attn_weights
        B, N, M = cost_matrix.shape
        row_values, row_indices = cost_matrix.min(dim=2)
        col_values, col_indices = cost_matrix.min(dim=1)

        row_sum = row_values.mean(dim=1)
        col_sum = col_values.mean(dim=1)

        overall = torch.stack([row_sum, col_sum], dim=1)
        return overall.max(dim=1)[0].mean()
    
    def content_loss(self, src_tokens, tgt_tokens):
        B, N, D = tgt_tokens.shape
        src_tokens /= src_tokens.clone().norm(dim=-1, keepdim=True)
        tgt_tokens /= tgt_tokens.clone().norm(dim=-1, keepdim=True)

        tgt_mat = 1 - torch.bmm(tgt_tokens, tgt_tokens.permute(0, 2, 1))
        src_mat = 1 - torch.bmm(src_tokens, src_tokens.permute(0, 2, 1))

        tgt_mat /= tgt_mat.clone().sum(dim=1, keepdim=True)
        src_mat /= src_mat.clone().sum(dim=1, keepdim=True)
        loss = F.l1_loss(tgt_mat, src_mat)
        return loss

    def moment_matching_loss(self, img_tokens, target_tokens):
        B, N, D = img_tokens.shape
        img_mean = img_tokens.mean(dim=1)
        target_mean = target_tokens.mean(dim=1)

        img_cov = img_tokens-img_mean

    def clip_partial_loss(self, src_tokens, tgt_tokens):
        B, N, _ = tgt_tokens.shape

        style_tokens = self.target_tokens.repeat(B, 1, 1)

        return self.remd_loss(tgt_tokens, style_tokens) + self.args.lambda_content * self.content_loss(src_tokens, tgt_tokens)

    def clip_directional_loss(self, src_img: torch.Tensor, source_class: str, target_img: torch.Tensor, target_class: str) -> torch.Tensor:
        src_encoding    = self.get_image_features(src_img)

        if self.lambda_partial > 0 and len(self.target_keys) > 0:
            src_tokens = self.feat_tokens[0]
            src_tokens /= src_tokens.clone().norm(dim=-1, keepdim=True)

        target_encoding = self.get_image_features(target_img)

        if self.lambda_partial > 0 and len(self.target_keys) > 0:
            # img_keys = self.feat_keys[0]  # BxNxD
            # img_keys /= img_keys.clone().norm(dim=-1, keepdim=True)
            tgt_tokens = self.feat_tokens[0]
            tgt_tokens /= tgt_tokens.clone().norm(dim=-1, keepdim=True)

        if self.target_direction is None:
            self.target_direction = self.compute_text_direction(source_class, target_class)

        edit_direction = (target_encoding - src_encoding)
        edit_direction /= edit_direction.clone().norm(dim=-1, keepdim=True)
        
        loss = self.direction_loss(edit_direction, self.target_direction).mean()
        # loss = 0
        if self.lambda_partial > 0 and len(self.target_keys) > 0:
            loss += self.lambda_partial * self.clip_partial_loss(src_tokens, tgt_tokens)
        return loss

    def global_clip_loss(self, img: torch.Tensor, text) -> torch.Tensor:
        if self.target_image is not None:
            target_encoding = self.get_image_features(img)
            return self.direction_loss(self.target_image, target_encoding).mean()

        if not isinstance(text, list):
            text = [text]
            
        tokens = clip.tokenize(text).to(self.device)
        image  = self.preprocess(img)

        logits_per_image, _ = self.model(image, tokens)

        return (1. - logits_per_image / 100).mean()
        
    def random_patch_centers(self, img_shape, num_patches, size):
        batch_size, channels, height, width = img_shape

        half_size = size // 2
        patch_centers = np.concatenate([np.random.randint(half_size, width - half_size,  size=(batch_size * num_patches, 1)),
                                        np.random.randint(half_size, height - half_size, size=(batch_size * num_patches, 1))], axis=1)

        return patch_centers

    def generate_patches(self, img: torch.Tensor, patch_centers, size):
        batch_size  = img.shape[0]
        num_patches = len(patch_centers) // batch_size
        half_size   = size // 2

        patches = []

        for batch_idx in range(batch_size):
            for patch_idx in range(num_patches):

                center_x = patch_centers[batch_idx * num_patches + patch_idx][0]
                center_y = patch_centers[batch_idx * num_patches + patch_idx][1]

                patch = img[batch_idx:batch_idx+1, :, center_y - half_size:center_y + half_size, center_x - half_size:center_x + half_size]

                patches.append(patch)

        patches = torch.cat(patches, axis=0)

        return patches

    def patch_scores(self, img: torch.Tensor, class_str: str, patch_centers, patch_size: int) -> torch.Tensor:

        parts = self.compose_text_with_templates(class_str, part_templates)    
        tokens = clip.tokenize(parts).to(self.device)
        text_features = self.encode_text(tokens).detach()

        patches        = self.generate_patches(img, patch_centers, patch_size)
        image_features = self.get_image_features(patches)

        similarity = image_features @ text_features.T

        return similarity

    def clip_patch_similarity(self, src_img: torch.Tensor, source_class: str, target_img: torch.Tensor, target_class: str) -> torch.Tensor:
        patch_size = 196 #TODO remove magic number

        patch_centers = self.random_patch_centers(src_img.shape, 4, patch_size) #TODO remove magic number
   
        src_scores    = self.patch_scores(src_img, source_class, patch_centers, patch_size)
        target_scores = self.patch_scores(target_img, target_class, patch_centers, patch_size)

        return self.patch_loss(src_scores, target_scores)

    def patch_directional_loss(self, src_img: torch.Tensor, source_class: str, target_img: torch.Tensor, target_class: str) -> torch.Tensor:

        if self.patch_text_directions is None:
            src_part_classes = self.compose_text_with_templates(source_class, part_templates)
            target_part_classes = self.compose_text_with_templates(target_class, part_templates)

            parts_classes = list(zip(src_part_classes, target_part_classes))

            self.patch_text_directions = torch.cat([self.compute_text_direction(pair[0], pair[1]) for pair in parts_classes], dim=0)

        patch_size = 510 # TODO remove magic numbers

        patch_centers = self.random_patch_centers(src_img.shape, 1, patch_size)

        patches = self.generate_patches(src_img, patch_centers, patch_size)
        src_features = self.get_image_features(patches)

        patches = self.generate_patches(target_img, patch_centers, patch_size)
        target_features = self.get_image_features(patches)

        edit_direction = (target_features - src_features)
        edit_direction /= edit_direction.clone().norm(dim=-1, keepdim=True)

        cosine_dists = 1. - self.patch_direction_loss(edit_direction.unsqueeze(1), self.patch_text_directions.unsqueeze(0))

        patch_class_scores = cosine_dists * (edit_direction @ self.patch_text_directions.T).softmax(dim=-1)

        return patch_class_scores.mean()

    def cnn_feature_loss(self, src_img: torch.Tensor, target_img: torch.Tensor) -> torch.Tensor:
        src_features = self.encode_images_with_cnn(src_img)
        target_features = self.encode_images_with_cnn(target_img)

        return self.texture_loss(src_features, target_features)

    def forward(self, src_img: torch.Tensor, source_class: str, target_img: torch.Tensor, target_class: str, texture_image: torch.Tensor = None):
        clip_loss = 0.0
        if self.args.dataset == 'car':
            target_img = target_img[:, :, 64:448, :].contiguous()
            src_img = src_img[:, :, 64:448, :].contiguous()

        if self.lambda_global:
            clip_loss += self.lambda_global * self.global_clip_loss(target_img, [f"a {target_class}"])

        if self.lambda_patch:
            clip_loss += self.lambda_patch * self.patch_directional_loss(src_img, source_class, target_img, target_class)

        if self.lambda_direction:
            clip_loss += self.lambda_direction * self.clip_directional_loss(src_img, source_class, target_img, target_class)

        if self.lambda_manifold:
            clip_loss += self.lambda_manifold * self.clip_angle_loss(src_img, source_class, target_img, target_class)

        # if self.lambda_texture and (texture_image is not None):
        if self.lambda_texture:
            # clip_loss += self.lambda_texture * self.cnn_feature_loss(texture_image, target_img)
            clip_loss += self.lambda_texture * self.cnn_feature_loss(src_img, target_img)

        return clip_loss
