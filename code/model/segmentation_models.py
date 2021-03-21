from typing import Optional, Union, List
import torch.nn as nn
import torch
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo
import segmentation_models_pytorch as smp
from segmentation_models_pytorch.base import modules as md
from segmentation_models_pytorch.base import SegmentationHead, ClassificationHead, SegmentationModel
from segmentation_models_pytorch.base.modules import Activation
from segmentation_models_pytorch.encoders.resnet import resnet_encoders
from segmentation_models_pytorch.encoders.dpn import dpn_encoders
from segmentation_models_pytorch.encoders.vgg import vgg_encoders
from segmentation_models_pytorch.encoders.senet import senet_encoders
from segmentation_models_pytorch.encoders.densenet import densenet_encoders
from segmentation_models_pytorch.encoders.inceptionresnetv2 import inceptionresnetv2_encoders
from segmentation_models_pytorch.encoders.inceptionv4 import inceptionv4_encoders
from segmentation_models_pytorch.encoders.efficientnet import efficient_net_encoders
from segmentation_models_pytorch.encoders.mobilenet import mobilenet_encoders
from segmentation_models_pytorch.encoders.xception import xception_encoders
from segmentation_models_pytorch.encoders.timm_efficientnet import timm_efficientnet_encoders
from segmentation_models_pytorch.encoders.timm_resnest import timm_resnest_encoders
from segmentation_models_pytorch.encoders.timm_res2net import timm_res2net_encoders
from segmentation_models_pytorch.encoders.timm_regnet import timm_regnet_encoders
from segmentation_models_pytorch.encoders.timm_sknet import timm_sknet_encoders


class SmpNet(nn.Module):
    def __init__(self, encoder_name, encoder_weights="imagenet", in_channels=3, n_class=10,
                 decoder_attention_type=None):
        super().__init__()
        self.model = UnetPlusPlus(  # UnetPlusPlus
            encoder_name=encoder_name,  # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
            encoder_weights=encoder_weights,  # use `imagenet` pretrained weights for encoder initialization
            in_channels=in_channels,  # model input channels (1 for grayscale images, 3 for RGB, etc.)
            classes=n_class,  # model output channels (number of classes in your dataset)
            decoder_attention_type=decoder_attention_type,
            # aux_params={'classes': n_class}
        )

    def forward(self, x):
        x = self.model(x)
        return x


class SmpDeepLab3p(nn.Module):
    def __init__(self,
                 encoder_name="resnet34",
                 encoder_depth=5,
                 encoder_weights="imagenet",
                 encoder_output_stride=16,
                 decoder_channels=256,
                 decoder_atrous_rates=(12, 24, 36),
                 in_channels=3,
                 classes=10,
                 activation=None,
                 upsampling=4,
                 aux_params=None, ):
        super().__init__()
        self.model = smp.DeepLabV3Plus(
            encoder_name=encoder_name,
            encoder_depth=encoder_depth,
            encoder_weights=encoder_weights,
            encoder_output_stride=encoder_output_stride,
            decoder_channels=decoder_channels,
            decoder_atrous_rates=decoder_atrous_rates,
            in_channels=in_channels,
            classes=classes,
            activation=activation,
            upsampling=upsampling,
            aux_params=aux_params,
        )

    def forward(self, x):
        return self.model(x)


encoders = {}
encoders.update(resnet_encoders)
encoders.update(dpn_encoders)
encoders.update(vgg_encoders)
encoders.update(senet_encoders)
encoders.update(densenet_encoders)
encoders.update(inceptionresnetv2_encoders)
encoders.update(inceptionv4_encoders)
encoders.update(efficient_net_encoders)
encoders.update(mobilenet_encoders)
encoders.update(xception_encoders)
encoders.update(timm_efficientnet_encoders)
encoders.update(timm_resnest_encoders)
encoders.update(timm_res2net_encoders)
encoders.update(timm_regnet_encoders)
encoders.update(timm_sknet_encoders)


class UnetPlusPlus(SegmentationModel):
    """Unet++ is a fully convolution neural network for image semantic segmentation. Consist of *encoder*
    and *decoder* parts connected with *skip connections*. Encoder extract features of different spatial
    resolution (skip connections) which are used by decoder to define accurate segmentation mask. Decoder of
    Unet++ is more complex than in usual Unet.

    Args:
        encoder_name: Name of the classification model that will be used as an encoder (a.k.a backbone)
            to extract features of different spatial resolution
        encoder_depth: A number of stages used in encoder in range [3, 5]. Each stage generate features
            two times smaller in spatial dimentions than previous one (e.g. for depth 0 we will have features
            with shapes [(N, C, H, W),], for depth 1 - [(N, C, H, W), (N, C, H // 2, W // 2)] and so on).
            Default is 5
        encoder_weights: One of **None** (random initialization), **"imagenet"** (pre-training on ImageNet) and
            other pretrained weights (see table with available weights for each encoder_name)
        decoder_channels: List of integers which specify **in_channels** parameter for convolutions used in decoder.
            Lenght of the list should be the same as **encoder_depth**
        decoder_use_batchnorm: If **True**, BatchNorm2d layer between Conv2D and Activation layers
            is used. If **"inplace"** InplaceABN will be used, allows to decrease memory consumption.
            Avaliable options are **True, False, "inplace"**
        decoder_attention_type: Attention module used in decoder of the model. Avaliable options are **None** and **scse**.
            SCSE paper - https://arxiv.org/abs/1808.08127
        in_channels: A number of input channels for the model, default is 3 (RGB images)
        classes: A number of classes for output mask (or you can think as a number of channels of output mask)
        activation: An activation function to apply after the final convolution layer.
            Avaliable options are **"sigmoid"**, **"softmax"**, **"logsoftmax"**, **"identity"**, **callable** and **None**.
            Default is **None**
        aux_params: Dictionary with parameters of the auxiliary output (classification head). Auxiliary output is build
            on top of encoder if **aux_params** is not **None** (default). Supported params:
                - classes (int): A number of classes
                - pooling (str): One of "max", "avg". Default is "avg"
                - dropout (float): Dropout factor in [0, 1)
                - activation (str): An activation function to apply "sigmoid"/"softmax" (could be **None** to return logits)

    Returns:
        ``torch.nn.Module``: **Unet++**

    Reference:
        https://arxiv.org/abs/1807.10165

    """

    def __init__(
            self,
            encoder_name: str = "resnet34",
            encoder_depth: int = 5,
            encoder_weights: Optional[str] = "imagenet",
            decoder_use_batchnorm: bool = True,
            decoder_channels: List[int] = (256, 128, 64, 32, 16),
            decoder_attention_type: Optional[str] = None,
            in_channels: int = 3,
            classes: int = 1,
            activation: Optional[Union[str, callable]] = None,
            aux_params: Optional[dict] = None,
    ):
        super().__init__()

        self.encoder = get_encoder(
            encoder_name,
            in_channels=in_channels,
            depth=encoder_depth,
            weights=encoder_weights,
        )

        self.decoder = UnetPlusPlusDecoder(
            encoder_channels=self.encoder.out_channels,
            decoder_channels=decoder_channels,
            n_blocks=encoder_depth,
            use_batchnorm=decoder_use_batchnorm,
            center=True if encoder_name.startswith("vgg") else False,
            attention_type=decoder_attention_type,
        )

        # self.segmentation_head = SegmentationHead_deepsuper(
        #     in_channels=decoder_channels[-1],
        #     out_channels=classes,
        #     activation=activation,
        #     kernel_size=3,
        # )

        self.segmentation_head = SegmentationHead(
            in_channels=decoder_channels[-1],
            out_channels=classes,
            activation=activation,
            kernel_size=3,
        )

        if aux_params is not None:
            self.classification_head = ClassificationHead(
                in_channels=self.encoder.out_channels[-1], **aux_params
            )
        else:
            self.classification_head = None

        self.name = "unetplusplus-{}".format(encoder_name)
        self.initialize()


def get_encoder(name, in_channels=3, depth=5, weights=None):
    import os, sys, warnings, errno, tempfile, hashlib, shutil, zipfile
    from tqdm.auto import tqdm
    from urllib.parse import urlparse
    from urllib.request import urlopen, Request

    def load_state_dict_from_url(url, model_dir="../external_data/", file_name='efficientnet-b7-dcc49843.pth', map_location=None, progress=True, ):
        r"""Loads the Torch serialized object at the given URL.

        If downloaded file is a zip file, it will be automatically
        decompressed.

        If the object is already present in `model_dir`, it's deserialized and
        returned.
        The default value of `model_dir` is ``<hub_dir>/checkpoints`` where
        `hub_dir` is the directory returned by :func:`~torch.hub.get_dir`.

        Args:
            url (string): URL of the object to download
            model_dir (string, optional): directory in which to save the object
            map_location (optional): a function or a dict specifying how to remap storage locations (see torch.load)
            progress (bool, optional): whether or not to display a progress bar to stderr.
                Default: True
            check_hash(bool, optional): If True, the filename part of the URL should follow the naming convention
                ``filename-<sha256>.ext`` where ``<sha256>`` is the first eight or more
                digits of the SHA256 hash of the contents of the file. The hash is used to
                ensure unique names and to verify the contents of the file.
                Default: False
            file_name (string, optional): name for the downloaded file. Filename from `url` will be used if not set.

        Example:
            # state_dict = torch.hub.load_state_dict_from_url('https://s3.amazonaws.com/pytorch/models/resnet18-5c106cde.pth')
        """

        # Issue warning to move data if old env is set

        def download_url_to_file(url, dst, hash_prefix=None, progress=True):
            r"""Download object at the given URL to a local path.

            Args:
                url (string): URL of the object to download
                dst (string): Full path where object will be saved, e.g. `/tmp/temporary_file`
                hash_prefix (string, optional): If not None, the SHA256 downloaded file should start with `hash_prefix`.
                    Default: None
                progress (bool, optional): whether or not to display a progress bar to stderr
                    Default: True

            Example:
                >>> torch.hub.download_url_to_file('https://s3.amazonaws.com/pytorch/models/resnet18-5c106cde.pth', '/tmp/temporary_file')

            """
            file_size = None
            # We use a different API for python2 since urllib(2) doesn't recognize the CA
            # certificates in older Python
            req = Request(url, headers={"User-Agent": "torch.hub"})
            u = urlopen(req)
            meta = u.info()
            if hasattr(meta, 'getheaders'):
                content_length = meta.getheaders("Content-Length")
            else:
                content_length = meta.get_all("Content-Length")
            if content_length is not None and len(content_length) > 0:
                file_size = int(content_length[0])

            # We deliberately save it in a temp file and move it after
            # download is complete. This prevents a local working checkpoint
            # being overridden by a broken download.
            dst = os.path.expanduser(dst)
            dst_dir = os.path.dirname(dst)
            f = tempfile.NamedTemporaryFile(delete=False, dir=dst_dir)

            try:
                if hash_prefix is not None:
                    sha256 = hashlib.sha256()
                with tqdm(total=file_size, disable=not progress,
                          unit='B', unit_scale=True, unit_divisor=1024) as pbar:
                    while True:
                        buffer = u.read(8192)
                        if len(buffer) == 0:
                            break
                        f.write(buffer)
                        if hash_prefix is not None:
                            sha256.update(buffer)
                        pbar.update(len(buffer))

                f.close()
                if hash_prefix is not None:
                    digest = sha256.hexdigest()
                    if digest[:len(hash_prefix)] != hash_prefix:
                        raise RuntimeError('invalid hash value (expected "{}", got "{}")'
                                           .format(hash_prefix, digest))
                shutil.move(f.name, dst)
            finally:
                f.close()
                if os.path.exists(f.name):
                    os.remove(f.name)

        def _is_legacy_zip_format(filename):
            if zipfile.is_zipfile(filename):
                infolist = zipfile.ZipFile(filename).infolist()
                return len(infolist) == 1 and not infolist[0].is_dir()
            return False

        def _legacy_zip_load(filename, model_dir, map_location):
            warnings.warn('Falling back to the old format < 1.6. This support will be '
                          'deprecated in favor of default zipfile format introduced in 1.6. '
                          'Please redo torch.save() to save it in the new zipfile format.')
            # Note: extractall() defaults to overwrite file if exists. No need to clean up beforehand.
            #       We deliberately don't handle tarfile here since our legacy serialization format was in tar.
            #       E.g. resnet18-5c106cde.pth which is widely used.
            with zipfile.ZipFile(filename) as f:
                members = f.infolist()
                if len(members) != 1:
                    raise RuntimeError('Only one file(not dir) is allowed in the zipfile')
                f.extractall(model_dir)
                extraced_name = members[0].filename
                extracted_file = os.path.join(model_dir, extraced_name)
            return torch.load(extracted_file, map_location=map_location)

        if os.getenv('TORCH_MODEL_ZOO'):
            warnings.warn('TORCH_MODEL_ZOO is deprecated, please use env TORCH_HOME instead')

        if not os.path.exists(model_dir):
            os.makedirs(model_dir)


        filename = os.path.join(model_dir, file_name)
        print(os.curdir)
        print(filename)
        print((file_name is not None))
        print(os.path.exists(filename))
        if file_name is not None and os.path.exists(filename):
            return torch.load(filename, map_location=map_location)
        else:
            parts = urlparse(url)
            filename = os.path.basename(parts.path)
            cached_file = os.path.join(model_dir, filename)
            if not os.path.exists(cached_file):
                sys.stderr.write('Downloading: "{}" to {}\n'.format(url, cached_file))
                hash_prefix = None
                download_url_to_file(url, cached_file, hash_prefix, progress=progress)

            if _is_legacy_zip_format(cached_file):
                return _legacy_zip_load(cached_file, model_dir, map_location)
            return torch.load(cached_file, map_location=map_location)

    try:
        Encoder = encoders[name]["encoder"]
    except KeyError:
        raise KeyError("Wrong encoder name `{}`, supported encoders: {}".format(name, list(encoders.keys())))

    params = encoders[name]["params"]
    params.update(depth=depth)
    encoder = Encoder(**params)

    if weights is not None:
        try:
            settings = encoders[name]["pretrained_settings"][weights]
        except KeyError:
            raise KeyError("Wrong pretrained weights `{}` for encoder `{}`. Avaliable options are: {}".format(
                weights, name, list(encoders[name]["pretrained_settings"].keys()),
            ))
        encoder.load_state_dict(load_state_dict_from_url(settings["url"]))

    encoder.set_in_channels(in_channels)

    return encoder


class UnetPlusPlusDecoder(nn.Module):
    def __init__(
            self,
            encoder_channels,
            decoder_channels,
            n_blocks=5,
            use_batchnorm=True,
            attention_type=None,
            center=False,
    ):
        super().__init__()

        if n_blocks != len(decoder_channels):
            raise ValueError(
                "Model depth is {}, but you provide `decoder_channels` for {} blocks.".format(
                    n_blocks, len(decoder_channels)
                )
            )

        encoder_channels = encoder_channels[1:]  # remove first skip with same spatial resolution
        encoder_channels = encoder_channels[::-1]  # reverse channels to start from head of encoder
        # computing blocks input and output channels
        head_channels = encoder_channels[0]
        self.in_channels = [head_channels] + list(decoder_channels[:-1])
        self.skip_channels = list(encoder_channels[1:]) + [0]
        self.out_channels = decoder_channels
        if center:
            self.center = CenterBlock(
                head_channels, head_channels, use_batchnorm=use_batchnorm
            )
        else:
            self.center = nn.Identity()

        # combine decoder keyword arguments
        kwargs = dict(use_batchnorm=use_batchnorm, attention_type=attention_type)

        blocks = {}
        for layer_idx in range(len(self.in_channels) - 1):
            for depth_idx in range(layer_idx + 1):
                if depth_idx == 0:
                    in_ch = self.in_channels[layer_idx]
                    skip_ch = self.skip_channels[layer_idx] * (layer_idx + 1)
                    out_ch = self.out_channels[layer_idx]
                else:
                    out_ch = self.skip_channels[layer_idx]
                    skip_ch = self.skip_channels[layer_idx] * (layer_idx + 1 - depth_idx)
                    in_ch = self.skip_channels[layer_idx - 1]
                blocks[f'x_{depth_idx}_{layer_idx}'] = DecoderBlock(in_ch, skip_ch, out_ch, **kwargs)
        blocks[f'x_{0}_{len(self.in_channels) - 1}'] = \
            DecoderBlock(self.in_channels[-1], 0, self.out_channels[-1], **kwargs)
        self.blocks = nn.ModuleDict(blocks)
        self.depth = len(self.in_channels) - 1

    def forward(self, *features):

        features = features[1:]  # remove first skip with same spatial resolution
        features = features[::-1]  # reverse channels to start from head of encoder
        # start bulding dense connections
        dense_x = {}
        for layer_idx in range(len(self.in_channels) - 1):
            for depth_idx in range(self.depth - layer_idx):
                if layer_idx == 0:
                    output = self.blocks[f'x_{depth_idx}_{depth_idx}'](features[depth_idx], features[depth_idx + 1])
                    dense_x[f'x_{depth_idx}_{depth_idx}'] = output
                else:
                    dense_l_i = depth_idx + layer_idx
                    cat_features = [dense_x[f'x_{idx}_{dense_l_i}'] for idx in range(depth_idx + 1, dense_l_i + 1)]
                    cat_features = torch.cat(cat_features + [features[dense_l_i + 1]], dim=1)
                    dense_x[f'x_{depth_idx}_{dense_l_i}'] = \
                        self.blocks[f'x_{depth_idx}_{dense_l_i}'](dense_x[f'x_{depth_idx}_{dense_l_i - 1}'],
                                                                  cat_features)
        dense_x[f'x_{0}_{self.depth}'] = self.blocks[f'x_{0}_{self.depth}'](dense_x[f'x_{0}_{self.depth - 1}'])
        return dense_x[f'x_{0}_{self.depth}']


class CenterBlock(nn.Sequential):
    def __init__(self, in_channels, out_channels, use_batchnorm=True):
        conv1 = md.Conv2dReLU(
            in_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm,
        )
        conv2 = md.Conv2dReLU(
            out_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm,
        )
        super().__init__(conv1, conv2)


class DecoderBlock(nn.Module):
    def __init__(
            self,
            in_channels,
            skip_channels,
            out_channels,
            use_batchnorm=True,
            attention_type=None,
    ):
        super().__init__()
        self.conv1 = md.Conv2dReLU(
            in_channels + skip_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm,
        )

        self.attention1 = md.Attention(attention_type, in_channels=in_channels + skip_channels)
        self.conv2 = md.Conv2dReLU(
            out_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm,
        )
        self.attention2 = md.Attention(attention_type, in_channels=out_channels)

    def forward(self, x, skip=None):
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        if skip is not None:
            x = torch.cat([x, skip], dim=1)
            x = self.attention1(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.attention2(x)
        return x


class SegmentationHead_deepsuper(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, activation=None, upsampling=1):
        super().__init__()
        self.s1 = nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=kernel_size // 2),
                                nn.UpsamplingBilinear2d(scale_factor=upsampling) if upsampling > 1 else nn.Identity(),
                                Activation(activation))
        self.s2 = nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=kernel_size // 2),
                                nn.UpsamplingBilinear2d(scale_factor=upsampling) if upsampling > 1 else nn.Identity(),
                                Activation(activation))
        self.s3 = nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=kernel_size // 2),
                                nn.UpsamplingBilinear2d(scale_factor=upsampling) if upsampling > 1 else nn.Identity(),
                                Activation(activation))
        self.s4 = nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=kernel_size // 2),
                                nn.UpsamplingBilinear2d(scale_factor=upsampling) if upsampling > 1 else nn.Identity(),
                                Activation(activation))

    def forward(self, x):
        x4 = self.s1(x[0])
        x3 = self.s2(x[1])
        x2 = self.s3(x[2])
        x1 = self.s4(x[3])
        return [x4, x3, x2, x1]


class UnetPlusPlusDecoder_deepsuper(nn.Module):
    def __init__(
            self,
            encoder_channels,
            decoder_channels,
            n_blocks=5,
            use_batchnorm=True,
            attention_type=None,
            center=False,
    ):
        super().__init__()

        if n_blocks != len(decoder_channels):
            raise ValueError(
                "Model depth is {}, but you provide `decoder_channels` for {} blocks.".format(
                    n_blocks, len(decoder_channels)
                )
            )

        encoder_channels = encoder_channels[1:]  # remove first skip with same spatial resolution
        encoder_channels = encoder_channels[::-1]  # reverse channels to start from head of encoder
        # computing blocks input and output channels
        head_channels = encoder_channels[0]
        self.in_channels = [head_channels] + list(decoder_channels[:-1])
        self.skip_channels = list(encoder_channels[1:]) + [0]
        self.out_channels = decoder_channels
        if center:
            self.center = CenterBlock(
                head_channels, head_channels, use_batchnorm=use_batchnorm
            )
        else:
            self.center = nn.Identity()

        # combine decoder keyword arguments
        kwargs = dict(use_batchnorm=use_batchnorm, attention_type=attention_type)

        blocks = {}
        for layer_idx in range(len(self.in_channels) - 1):
            for depth_idx in range(layer_idx + 1):
                if depth_idx == 0:
                    in_ch = self.in_channels[layer_idx]
                    skip_ch = self.skip_channels[layer_idx] * (layer_idx + 1)
                    out_ch = self.out_channels[layer_idx]
                else:
                    out_ch = self.skip_channels[layer_idx]
                    skip_ch = self.skip_channels[layer_idx] * (layer_idx + 1 - depth_idx)
                    in_ch = self.skip_channels[layer_idx - 1]
                blocks[f'x_{depth_idx}_{layer_idx}'] = DecoderBlock(in_ch, skip_ch, out_ch, **kwargs)
        blocks[f'x_{-1}_{len(self.in_channels) - 1}'] = \
            DecoderBlock(self.in_channels[-1], 0, self.out_channels[-1], **kwargs)
        blocks[f'x_{-1}_{len(self.in_channels) - 2}'] = \
            DecoderBlock(self.in_channels[-1] * 2, 0, self.out_channels[-1], **kwargs)
        blocks[f'x_{-1}_{len(self.in_channels) - 3}'] = \
            DecoderBlock(self.in_channels[-1] * 2, 0, self.out_channels[-1], **kwargs)
        blocks[f'x_{-1}_{len(self.in_channels) - 4}'] = \
            DecoderBlock(self.in_channels[-1] * 2, 0, self.out_channels[-1], **kwargs)
        self.blocks = nn.ModuleDict(blocks)
        self.depth = len(self.in_channels) - 1

    def forward(self, *features):

        features = features[1:]  # remove first skip with same spatial resolution
        features = features[::-1]  # reverse channels to start from head of encoder
        # start bulding dense connections
        dense_x = {}
        for layer_idx in range(len(self.in_channels) - 1):
            for depth_idx in range(self.depth - layer_idx):
                if layer_idx == 0:
                    output = self.blocks[f'x_{depth_idx}_{depth_idx}'](features[depth_idx], features[depth_idx + 1])
                    dense_x[f'x_{depth_idx}_{depth_idx}'] = output
                else:
                    dense_l_i = depth_idx + layer_idx
                    cat_features = [dense_x[f'x_{idx}_{dense_l_i}'] for idx in range(depth_idx + 1, dense_l_i + 1)]
                    cat_features = torch.cat(cat_features + [features[dense_l_i + 1]], dim=1)
                    dense_x[f'x_{depth_idx}_{dense_l_i}'] = \
                        self.blocks[f'x_{depth_idx}_{dense_l_i}'](dense_x[f'x_{depth_idx}_{dense_l_i - 1}'],
                                                                  cat_features)
        dense_x[f'x_{-1}_{self.depth - 1}'] = self.blocks[f'x_{-1}_{self.depth}'](dense_x[f'x_{0}_{self.depth - 1}'])
        dense_x[f'x_{-1}_{self.depth - 2}'] = self.blocks[f'x_{-1}_{self.depth - 1}'](
            dense_x[f'x_{1}_{self.depth - 1}'])
        dense_x[f'x_{-1}_{self.depth - 3}'] = self.blocks[f'x_{-1}_{self.depth - 2}'](
            dense_x[f'x_{2}_{self.depth - 1}'])
        dense_x[f'x_{-1}_{self.depth - 4}'] = self.blocks[f'x_{-1}_{self.depth - 3}'](
            dense_x[f'x_{3}_{self.depth - 1}'])
        return [dense_x[f'x_{-1}_{self.depth - 1}'], dense_x[f'x_{-1}_{self.depth - 2}'],
                dense_x[f'x_{-1}_{self.depth - 3}'], dense_x[f'x_{-1}_{self.depth - 4}']]
