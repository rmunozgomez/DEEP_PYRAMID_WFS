import importlib
import torch
import torch.nn as nn
import torch.nn.functional as F


# =============================================================================
# Modelos WFS livianos / medios
# =============================================================================

class BasicBlockWFS(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1, activation="silu"):
        super().__init__()

        self.conv1 = nn.Conv2d(
            in_ch,
            out_ch,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(out_ch)

        self.conv2 = nn.Conv2d(
            out_ch,
            out_ch,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )
        self.bn2 = nn.BatchNorm2d(out_ch)

        if stride != 1 or in_ch != out_ch:
            self.skip = nn.Sequential(
                nn.Conv2d(
                    in_ch,
                    out_ch,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(out_ch),
            )
        else:
            self.skip = nn.Identity()

        if activation == "relu":
            self.act = nn.ReLU(inplace=True)
        elif activation == "silu":
            self.act = nn.SiLU(inplace=True)
        else:
            raise ValueError(f"activation no soportada: {activation}")

    def forward(self, x):
        identity = self.skip(x)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.act(x)

        x = self.conv2(x)
        x = self.bn2(x)

        x = x + identity
        x = self.act(x)

        return x


class ResNetWFS(nn.Module):
    """
    CNN residual compacta para WFS.

    Entrada:
        [B, in_chans, resolution, resolution]

    Salida:
        [B, num_classes]

    Recomendado para:
        input_shape=(B, 4, 32, 32)
    """

    def __init__(
        self,
        in_chans=4,
        num_classes=68,
        resolution=32,
        base_ch=64,
        activation="silu",
    ):
        super().__init__()

        if resolution < 16:
            raise ValueError(f"resolution demasiado baja para ResNetWFS: {resolution}")

        self.in_chans = int(in_chans)
        self.num_classes = int(num_classes)
        self.resolution = int(resolution)
        self.base_ch = int(base_ch)

        if activation == "relu":
            act = nn.ReLU(inplace=True)
        elif activation == "silu":
            act = nn.SiLU(inplace=True)
        else:
            raise ValueError(f"activation no soportada: {activation}")

        self.stem = nn.Sequential(
            nn.Conv2d(
                self.in_chans,
                base_ch,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(base_ch),
            act,
        )

        self.features = nn.Sequential(
            BasicBlockWFS(base_ch, base_ch, stride=1, activation=activation),
            BasicBlockWFS(base_ch, base_ch, stride=1, activation=activation),

            BasicBlockWFS(base_ch, base_ch * 2, stride=2, activation=activation),
            BasicBlockWFS(base_ch * 2, base_ch * 2, stride=1, activation=activation),

            BasicBlockWFS(base_ch * 2, base_ch * 3, stride=2, activation=activation),
            BasicBlockWFS(base_ch * 3, base_ch * 3, stride=1, activation=activation),

            BasicBlockWFS(base_ch * 3, base_ch * 4, stride=2, activation=activation),
            BasicBlockWFS(base_ch * 4, base_ch * 4, stride=1, activation=activation),
        )

        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(base_ch * 4, 256),
            nn.SiLU(inplace=True) if activation == "silu" else nn.ReLU(inplace=True),
            nn.Linear(256, self.num_classes),
        )

    def forward(self, x):
        x = self.stem(x)
        x = self.features(x)
        x = self.head(x)
        return x


class TinyResNetWFS(nn.Module):
    """
    Versión más liviana de ResNetWFS.

    Usa la misma arquitectura, pero con base_ch=32.
    """

    def __init__(
        self,
        in_chans=4,
        num_classes=68,
        resolution=32,
        base_ch=32,
        activation="silu",
    ):
        super().__init__()

        self.model = ResNetWFS(
            in_chans=in_chans,
            num_classes=num_classes,
            resolution=resolution,
            base_ch=base_ch,
            activation=activation,
        )

    def forward(self, x):
        return self.model(x)


# =============================================================================
# RepVGG-like WFS
# =============================================================================

class RepVGGBlockWFS(nn.Module):
    """
    Bloque tipo RepVGG simple.

    Nota:
        Esta versión es fácil de entrenar y exportar.
        No incluye todavía función de reparametrización a deploy.
        Aun así, para probar contra GCViT/ConvNeXt es simple y estable.
    """

    def __init__(self, in_ch, out_ch, stride=1, activation="silu"):
        super().__init__()

        self.conv3 = nn.Sequential(
            nn.Conv2d(
                in_ch,
                out_ch,
                kernel_size=3,
                stride=stride,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(out_ch),
        )

        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_ch,
                out_ch,
                kernel_size=1,
                stride=stride,
                padding=0,
                bias=False,
            ),
            nn.BatchNorm2d(out_ch),
        )

        if stride == 1 and in_ch == out_ch:
            self.identity = nn.BatchNorm2d(out_ch)
        else:
            self.identity = None

        if activation == "relu":
            self.act = nn.ReLU(inplace=True)
        elif activation == "silu":
            self.act = nn.SiLU(inplace=True)
        else:
            raise ValueError(f"activation no soportada: {activation}")

    def forward(self, x):
        out = self.conv3(x) + self.conv1(x)

        if self.identity is not None:
            out = out + self.identity(x)

        out = self.act(out)
        return out


class RepVGGWFS(nn.Module):
    """
    Red tipo RepVGG para WFS.

    Entrada:
        [B, in_chans, resolution, resolution]

    Salida:
        [B, num_classes]

    Buena opción para TensorRT porque usa principalmente Conv + BN + activación.
    """

    def __init__(
        self,
        in_chans=4,
        num_classes=68,
        resolution=32,
        base_ch=64,
        activation="silu",
    ):
        super().__init__()

        if resolution < 16:
            raise ValueError(f"resolution demasiado baja para RepVGGWFS: {resolution}")

        self.in_chans = int(in_chans)
        self.num_classes = int(num_classes)
        self.resolution = int(resolution)
        self.base_ch = int(base_ch)

        self.stem = nn.Sequential(
            nn.Conv2d(
                self.in_chans,
                base_ch,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(base_ch),
            nn.SiLU(inplace=True) if activation == "silu" else nn.ReLU(inplace=True),
        )

        self.features = nn.Sequential(
            RepVGGBlockWFS(base_ch, base_ch, stride=1, activation=activation),
            RepVGGBlockWFS(base_ch, base_ch, stride=1, activation=activation),

            RepVGGBlockWFS(base_ch, base_ch * 2, stride=2, activation=activation),
            RepVGGBlockWFS(base_ch * 2, base_ch * 2, stride=1, activation=activation),

            RepVGGBlockWFS(base_ch * 2, base_ch * 3, stride=2, activation=activation),
            RepVGGBlockWFS(base_ch * 3, base_ch * 3, stride=1, activation=activation),

            RepVGGBlockWFS(base_ch * 3, base_ch * 4, stride=2, activation=activation),
            RepVGGBlockWFS(base_ch * 4, base_ch * 4, stride=1, activation=activation),
        )

        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(base_ch * 4, 256),
            nn.SiLU(inplace=True) if activation == "silu" else nn.ReLU(inplace=True),
            nn.Linear(256, self.num_classes),
        )

    def forward(self, x):
        x = self.stem(x)
        x = self.features(x)
        x = self.head(x)
        return x


# =============================================================================
# Additional compact WFS models
# =============================================================================

def _make_activation(activation="silu"):
    if activation == "relu":
        return nn.ReLU(inplace=True)
    if activation == "silu":
        return nn.SiLU(inplace=True)
    raise ValueError(f"activation no soportada: {activation}")


class MicroResNetWFS(TinyResNetWFS):
    """
    Variante micro de TinyResNetWFS.

    Igual arquitectura que TinyResNetWFS, pero con base_ch=16 por defecto.
    Recomendada para probar máxima velocidad con input_shape=(B,4,32,32).
    """

    def __init__(
        self,
        in_chans=4,
        num_classes=68,
        resolution=32,
        base_ch=16,
        activation="silu",
    ):
        super().__init__(
            in_chans=in_chans,
            num_classes=num_classes,
            resolution=resolution,
            base_ch=base_ch,
            activation=activation,
        )


class WideTinyResNetWFS(TinyResNetWFS):
    """
    Variante más ancha de TinyResNetWFS.

    Igual arquitectura que TinyResNetWFS, pero con base_ch=48 por defecto.
    Útil para verificar si TinyResNetWFS está limitado por capacidad.
    """

    def __init__(
        self,
        in_chans=4,
        num_classes=68,
        resolution=32,
        base_ch=48,
        activation="silu",
    ):
        super().__init__(
            in_chans=in_chans,
            num_classes=num_classes,
            resolution=resolution,
            base_ch=base_ch,
            activation=activation,
        )


class ShallowResNetWFS(nn.Module):
    """
    ResNet WFS más superficial.

    Diseñada para entradas pequeñas tipo (B,4,32,32), donde demasiados
    downsamplings pueden eliminar información espacial fina de las pupilas.
    """

    def __init__(
        self,
        in_chans=4,
        num_classes=68,
        resolution=32,
        base_ch=32,
        activation="silu",
    ):
        super().__init__()

        if resolution < 16:
            raise ValueError(f"resolution demasiado baja para ShallowResNetWFS: {resolution}")

        act = _make_activation(activation)

        self.in_chans = int(in_chans)
        self.num_classes = int(num_classes)
        self.resolution = int(resolution)
        self.base_ch = int(base_ch)

        self.stem = nn.Sequential(
            nn.Conv2d(in_chans, base_ch, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(base_ch),
            act,
        )

        self.features = nn.Sequential(
            BasicBlockWFS(base_ch, base_ch, stride=1, activation=activation),
            BasicBlockWFS(base_ch, base_ch * 2, stride=2, activation=activation),
            BasicBlockWFS(base_ch * 2, base_ch * 4, stride=2, activation=activation),
        )

        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(base_ch * 4, 128),
            _make_activation(activation),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        x = self.stem(x)
        x = self.features(x)
        x = self.head(x)
        return x


class ConvNeXtBlockWFS(nn.Module):
    """
    Bloque tipo ConvNeXt compacto usando BatchNorm2d.

    Usa depthwise convolution + pointwise MLP convolucional.
    Se evita LayerNorm para mantenerlo simple y amistoso con TensorRT.
    """

    def __init__(self, ch, mlp_ratio=2, activation="silu"):
        super().__init__()

        hidden = int(ch * mlp_ratio)

        self.dwconv = nn.Conv2d(
            ch,
            ch,
            kernel_size=7,
            stride=1,
            padding=3,
            groups=ch,
            bias=False,
        )
        self.bn = nn.BatchNorm2d(ch)

        self.pwconv = nn.Sequential(
            nn.Conv2d(ch, hidden, kernel_size=1, bias=False),
            _make_activation(activation),
            nn.Conv2d(hidden, ch, kernel_size=1, bias=False),
        )

    def forward(self, x):
        return x + self.pwconv(self.bn(self.dwconv(x)))


class TinyConvNeXtWFS(nn.Module):
    """
    ConvNeXt-like compacto para WFS.

    Alternativa liviana a torchvision ConvNeXtTiny para input_shape=(B,4,32,32).
    """

    def __init__(
        self,
        in_chans=4,
        num_classes=68,
        resolution=32,
        base_ch=32,
        activation="silu",
    ):
        super().__init__()

        if resolution < 16:
            raise ValueError(f"resolution demasiado baja para TinyConvNeXtWFS: {resolution}")

        self.in_chans = int(in_chans)
        self.num_classes = int(num_classes)
        self.resolution = int(resolution)
        self.base_ch = int(base_ch)

        self.stem = nn.Sequential(
            nn.Conv2d(in_chans, base_ch, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(base_ch),
            _make_activation(activation),
        )

        self.stage1 = nn.Sequential(
            ConvNeXtBlockWFS(base_ch, activation=activation),
            ConvNeXtBlockWFS(base_ch, activation=activation),
        )

        self.down1 = nn.Sequential(
            nn.Conv2d(base_ch, base_ch * 2, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(base_ch * 2),
            _make_activation(activation),
        )

        self.stage2 = nn.Sequential(
            ConvNeXtBlockWFS(base_ch * 2, activation=activation),
            ConvNeXtBlockWFS(base_ch * 2, activation=activation),
        )

        self.down2 = nn.Sequential(
            nn.Conv2d(base_ch * 2, base_ch * 4, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(base_ch * 4),
            _make_activation(activation),
        )

        self.stage3 = nn.Sequential(
            ConvNeXtBlockWFS(base_ch * 4, activation=activation),
        )

        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(base_ch * 4, 128),
            _make_activation(activation),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        x = self.stem(x)
        x = self.stage1(x)
        x = self.down1(x)
        x = self.stage2(x)
        x = self.down2(x)
        x = self.stage3(x)
        x = self.head(x)
        return x


class DSConvBlockWFS(nn.Module):
    """
    Depthwise-separable convolution block.

    Bloque tipo MobileNet: depthwise 3x3 + pointwise 1x1.
    """

    def __init__(self, in_ch, out_ch, stride=1, activation="silu"):
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv2d(
                in_ch,
                in_ch,
                kernel_size=3,
                stride=stride,
                padding=1,
                groups=in_ch,
                bias=False,
            ),
            nn.BatchNorm2d(in_ch),
            _make_activation(activation),
            nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_ch),
            _make_activation(activation),
        )

    def forward(self, x):
        return self.block(x)


class MobileNetWFS(nn.Module):
    """
    MobileNet-like WFS.

    Muy rápida para inferencia. Buena candidata si la prioridad es Hz.
    """

    def __init__(
        self,
        in_chans=4,
        num_classes=68,
        resolution=32,
        base_ch=32,
        activation="silu",
    ):
        super().__init__()

        if resolution < 16:
            raise ValueError(f"resolution demasiado baja para MobileNetWFS: {resolution}")

        self.in_chans = int(in_chans)
        self.num_classes = int(num_classes)
        self.resolution = int(resolution)
        self.base_ch = int(base_ch)

        self.stem = nn.Sequential(
            nn.Conv2d(in_chans, base_ch, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(base_ch),
            _make_activation(activation),
        )

        self.features = nn.Sequential(
            DSConvBlockWFS(base_ch, base_ch, stride=1, activation=activation),
            DSConvBlockWFS(base_ch, base_ch * 2, stride=2, activation=activation),
            DSConvBlockWFS(base_ch * 2, base_ch * 2, stride=1, activation=activation),
            DSConvBlockWFS(base_ch * 2, base_ch * 4, stride=2, activation=activation),
            DSConvBlockWFS(base_ch * 4, base_ch * 4, stride=1, activation=activation),
        )

        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(base_ch * 4, 128),
            _make_activation(activation),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        x = self.stem(x)
        x = self.features(x)
        x = self.head(x)
        return x


class PupilMixerWFS(nn.Module):
    """
    CNN compacta orientada a WFS multicanal.

    Pensada para entradas tipo (B,4,32,32), donde los 4 canales representan
    pupilas/crops. Primero codifica los canales y luego mezcla features.
    """

    def __init__(
        self,
        in_chans=4,
        num_classes=68,
        resolution=32,
        base_ch=32,
        activation="silu",
    ):
        super().__init__()

        if resolution < 16:
            raise ValueError(f"resolution demasiado baja para PupilMixerWFS: {resolution}")

        self.in_chans = int(in_chans)
        self.num_classes = int(num_classes)
        self.resolution = int(resolution)
        self.base_ch = int(base_ch)

        self.pupil_encoder = nn.Sequential(
            nn.Conv2d(in_chans, base_ch, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(base_ch),
            _make_activation(activation),
            nn.Conv2d(base_ch, base_ch, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(base_ch),
            _make_activation(activation),
        )

        self.mixer = nn.Sequential(
            nn.Conv2d(base_ch, base_ch * 2, kernel_size=1, bias=False),
            nn.BatchNorm2d(base_ch * 2),
            _make_activation(activation),
        )

        self.features = nn.Sequential(
            BasicBlockWFS(base_ch * 2, base_ch * 2, stride=1, activation=activation),
            BasicBlockWFS(base_ch * 2, base_ch * 4, stride=2, activation=activation),
            BasicBlockWFS(base_ch * 4, base_ch * 4, stride=1, activation=activation),
            BasicBlockWFS(base_ch * 4, base_ch * 4, stride=2, activation=activation),
        )

        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(base_ch * 4, 128),
            _make_activation(activation),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        x = self.pupil_encoder(x)
        x = self.mixer(x)
        x = self.features(x)
        x = self.head(x)
        return x



# =============================================================================
# FastWFS: flexible low-latency WFS model
# =============================================================================

class FastWFSBlock(nn.Module):
    """
    Bloque rápido Conv-BN-ReLU.

    Pensado para TensorRT/float32:
        - Conv2d bias=False
        - BatchNorm2d fusionable
        - ReLU inplace
        - sin SiLU, LayerNorm, attention ni operaciones dinámicas

    Para latencia baja, mantener base_ch pequeño.
    """

    def __init__(self, in_ch, out_ch, stride=1, kernel_size=3):
        super().__init__()
        padding = kernel_size // 2

        self.block = nn.Sequential(
            nn.Conv2d(
                in_ch,
                out_ch,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                bias=False,
            ),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class FastWFS(nn.Module):
    """
    Red WFS rápida y flexible.

    Entrada:
        [B, in_chans, resolution, resolution]

    Salida:
        [B, num_classes]

    Uso recomendado:
        input_shape=(1, 4, 32, 32)
        output_dim=97
        dtype=torch.float32

    También permite cambiar:
        - in_chans desde input_shape
        - resolution desde input_shape
        - num_classes desde output_dim

    Diseño:
        Conv-BN-ReLU con downsampling progresivo hasta un mapa pequeño,
        luego GAP + MLP compacto.

    Nota:
        Para TensorRT y alta velocidad, exportar con shape fijo.
    """

    def __init__(
        self,
        in_chans=4,
        num_classes=97,
        resolution=32,
        base_ch=24,
        head_ch=128,
        min_spatial=4,
        max_downsamples=3,
    ):
        super().__init__()

        if resolution < 8:
            raise ValueError(f"resolution demasiado baja para FastWFS: {resolution}")

        self.in_chans = int(in_chans)
        self.num_classes = int(num_classes)
        self.resolution = int(resolution)
        self.base_ch = int(base_ch)
        self.head_ch = int(head_ch)
        self.min_spatial = int(min_spatial)
        self.max_downsamples = int(max_downsamples)

        if self.min_spatial < 1:
            raise ValueError(f"min_spatial debe ser >= 1. Recibido: {self.min_spatial}")

        if self.max_downsamples < 0:
            raise ValueError(f"max_downsamples debe ser >= 0. Recibido: {self.max_downsamples}")

        layers = []

        # Stem: conserva resolución.
        c = self.base_ch
        layers.append(FastWFSBlock(self.in_chans, c, stride=1, kernel_size=3))
        layers.append(FastWFSBlock(c, c, stride=1, kernel_size=3))

        spatial = self.resolution
        n_down = 0

        # Downsampling progresivo.
        # Ejemplo resolution=32:
        #   32 -> 16 -> 8 -> 4
        while spatial > self.min_spatial and n_down < self.max_downsamples:
            c_next = c + self.base_ch
            layers.append(FastWFSBlock(c, c_next, stride=2, kernel_size=3))
            layers.append(FastWFSBlock(c_next, c_next, stride=1, kernel_size=3))

            c = c_next
            spatial = (spatial + 1) // 2
            n_down += 1

        self.out_ch = c
        self.out_spatial = spatial

        self.features = nn.Sequential(*layers)

        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(self.out_ch, self.head_ch),
            nn.ReLU(inplace=True),
            nn.Linear(self.head_ch, self.num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.head(x)
        return x



# =============================================================================
# ModelManager
# =============================================================================

class ModelManager(nn.Module):
    _MODEL_REGISTRY = {
        "GcVit": ("NN.GcVit", "GCViT"),
        "ConvNeXtTiny": ("torchvision.models", "convnext_tiny"),

        # Modelos definidos en este mismo archivo
        "ResNetWFS": ("__local__", "ResNetWFS"),
        "TinyResNetWFS": ("__local__", "TinyResNetWFS"),
        "RepVGGWFS": ("__local__", "RepVGGWFS"),

        # Nuevos modelos locales WFS
        "MicroResNetWFS": ("__local__", "MicroResNetWFS"),
        "WideTinyResNetWFS": ("__local__", "WideTinyResNetWFS"),
        "ShallowResNetWFS": ("__local__", "ShallowResNetWFS"),
        "TinyConvNeXtWFS": ("__local__", "TinyConvNeXtWFS"),
        "MobileNetWFS": ("__local__", "MobileNetWFS"),
        "PupilMixerWFS": ("__local__", "PupilMixerWFS"),
        "FastWFS": ("__local__", "FastWFS"),
    }

    _LOCAL_MODELS = {
        "ResNetWFS",
        "TinyResNetWFS",
        "RepVGGWFS",
        "MicroResNetWFS",
        "WideTinyResNetWFS",
        "ShallowResNetWFS",
        "TinyConvNeXtWFS",
        "MobileNetWFS",
        "PupilMixerWFS",
        "FastWFS",
    }

    def __init__(
        self,
        device="cpu",
        dtype=torch.float32,
        nnModel="GcVit",
        input_shape=(1, 1, 128, 128),  # (B,C,N,N)
        output_dim=68,
        wts=None,
    ):
        super().__init__()

        self.device = device
        self.dtype = dtype
        self.nnModel = nnModel
        self.input_shape = tuple(input_shape)
        self.output_dim = int(output_dim)
        self.wts = wts

        if len(self.input_shape) != 4:
            raise ValueError(f"input_shape debe ser (B,C,N,N). Recibido: {self.input_shape}")

        _, C, N1, N2 = self.input_shape
        if N1 != N2:
            raise ValueError(f"Se espera input cuadrado (N==N). Recibido: {self.input_shape}")

        self.in_chans = int(C)
        self.N = int(N1)
        self.num_classes = self.output_dim

        # Mantenemos la lógica original
        if self.in_chans == 1:
            self.nnModel_input_type = "full_frame"
            self.res_nnModel = self.N
            self.channels_nnRes = None
        else:
            self.nnModel_input_type = "channels"
            self.channels_nnRes = self.N
            self.res_nnModel = None

        ModelClass = self._import_model_class(self.nnModel)

        # ------------------------------------------------------------------
        # full_frame: C == 1
        # ------------------------------------------------------------------
        if self.nnModel_input_type == "full_frame":

            # --------------------------------------------------------------
            # Modelos locales WFS: sirven para cualquier C y resolución >=16
            # --------------------------------------------------------------
            if self.nnModel in self._LOCAL_MODELS:
                self.NN = ModelClass(
                    in_chans=self.in_chans,
                    num_classes=self.num_classes,
                    resolution=self.res_nnModel,
                )

            # --------------------------------------------------------------
            # GCViT: se deja con la misma lógica original
            # --------------------------------------------------------------
            elif self.nnModel == "GcVit":
                if self.res_nnModel == 128:
                    self.NN = ModelClass(
                        num_classes=self.num_classes,
                        depths=[2, 2, 6, 2],
                        num_heads=[2, 4, 8, 16],
                        window_size=[4, 4, 8, 4],
                        resolution=self.res_nnModel,
                        in_chans=self.in_chans,
                        dim=64,
                        mlp_ratio=3,
                        drop_path_rate=0.2,
                    )
                elif self.res_nnModel == 256:
                    self.NN = ModelClass(
                        num_classes=self.num_classes,
                        depths=[2, 2, 6, 2],
                        num_heads=[2, 4, 8, 16],
                        window_size=[8, 8, 16, 8],
                        resolution=self.res_nnModel,
                        in_chans=self.in_chans,
                        dim=64,
                        mlp_ratio=3,
                        drop_path_rate=0.2,
                    )
                else:
                    raise ValueError(f"Resolución no soportada (full_frame): {self.res_nnModel}")

            elif self.nnModel == "ConvNeXtTiny":
                if self.res_nnModel < 32:
                    raise ValueError(
                        f"ConvNeXtTiny requiere al menos 32x32. Recibido: {self.res_nnModel}x{self.res_nnModel}"
                    )

                try:
                    self.NN = ModelClass(weights=None)
                except TypeError:
                    # Compatibilidad con versiones antiguas de torchvision
                    self.NN = ModelClass(pretrained=False)

                # Reemplazar primera conv para soportar in_chans != 3
                old_conv = self.NN.features[0][0]
                self.NN.features[0][0] = nn.Conv2d(
                    in_channels=self.in_chans,
                    out_channels=old_conv.out_channels,
                    kernel_size=old_conv.kernel_size,
                    stride=old_conv.stride,
                    padding=old_conv.padding,
                    dilation=old_conv.dilation,
                    groups=old_conv.groups,
                    bias=(old_conv.bias is not None),
                    padding_mode=old_conv.padding_mode,
                )

                # Reemplazar capa final para output_dim
                old_head = self.NN.classifier[2]
                self.NN.classifier[2] = nn.Linear(
                    in_features=old_head.in_features,
                    out_features=self.num_classes,
                    bias=(old_head.bias is not None),
                )

            else:
                raise ValueError(f"Modelo no manejado en full_frame: {self.nnModel}")

        # ------------------------------------------------------------------
        # channels: C > 1
        # ------------------------------------------------------------------
        elif self.nnModel_input_type == "channels":

            # --------------------------------------------------------------
            # Modelos locales WFS: recomendados para input_shape=(B,4,32,32)
            # --------------------------------------------------------------
            if self.nnModel in self._LOCAL_MODELS:
                self.NN = ModelClass(
                    in_chans=self.in_chans,
                    num_classes=self.num_classes,
                    resolution=self.channels_nnRes,
                )

            elif self.nnModel == "GcVit":
                if self.channels_nnRes % 32 != 0:
                    raise ValueError(
                        f"channels_nnRes debe ser múltiplo de 32 para GCViT. Recibido: {self.channels_nnRes}"
                    )
                window = self.channels_nnRes // 32
                self.NN = ModelClass(
                    num_classes=self.num_classes,
                    depths=[2, 2, 6, 2],
                    num_heads=[2, 4, 8, 16],
                    window_size=[window, window, window * 2, window],
                    resolution=self.channels_nnRes,
                    in_chans=self.in_chans,
                    dim=64,
                    mlp_ratio=3,
                    drop_path_rate=0.2,
                )

            elif self.nnModel == "ConvNeXtTiny":
                if self.channels_nnRes < 32:
                    raise ValueError(
                        f"ConvNeXtTiny requiere al menos 32x32. Recibido: {self.channels_nnRes}x{self.channels_nnRes}"
                    )

                try:
                    self.NN = ModelClass(weights=None)
                except TypeError:
                    self.NN = ModelClass(pretrained=False)

                # Reemplazar primera conv para soportar cualquier cantidad de canales
                old_conv = self.NN.features[0][0]
                self.NN.features[0][0] = nn.Conv2d(
                    in_channels=self.in_chans,
                    out_channels=old_conv.out_channels,
                    kernel_size=old_conv.kernel_size,
                    stride=old_conv.stride,
                    padding=old_conv.padding,
                    dilation=old_conv.dilation,
                    groups=old_conv.groups,
                    bias=(old_conv.bias is not None),
                    padding_mode=old_conv.padding_mode,
                )

                # Reemplazar capa final para output_dim
                old_head = self.NN.classifier[2]
                self.NN.classifier[2] = nn.Linear(
                    in_features=old_head.in_features,
                    out_features=self.num_classes,
                    bias=(old_head.bias is not None),
                )

            else:
                raise ValueError(f"Modelo no manejado en channels: {self.nnModel}")

        else:
            raise ValueError(f"Tipo de input no soportado: {self.nnModel_input_type}")

        # Igual que antes: modelo en float32 y a device
        self.NN = self.NN.to(torch.float32).to(self.device)

        # y luego dtype final si quieres
        if self.dtype is not None:
            self.NN = self.NN.to(self.dtype)

        # Si te pasan wts "puros", los carga
        if self.wts is not None:
            obj = torch.load(self.wts, map_location=self.device)
            if isinstance(obj, dict) and "state_dict" in obj:
                obj = obj["state_dict"]
            self.NN.load_state_dict(obj)

    def _import_model_class(self, name: str):
        if name not in self._MODEL_REGISTRY:
            raise ValueError(f"Modelo no soportado: {name}. Disponibles: {list(self._MODEL_REGISTRY.keys())}")

        module_path, class_name = self._MODEL_REGISTRY[name]

        # Modelos definidos en este mismo archivo
        if module_path == "__local__":
            if class_name not in globals():
                raise ValueError(f"Clase local no encontrada: {class_name}")
            return globals()[class_name]

        module = importlib.import_module(module_path)
        return getattr(module, class_name)

    def forward(self, x):
        return self.NN(x)

    # ---------------------------------------------------------------------
    # Guardar/cargar el objeto completo (arquitectura+pesos+specs)
    # ---------------------------------------------------------------------
    def save_full(self, path: str):
        """
        Guarda el ModelManager COMPLETO (incluye NN, input_shape, output_dim, etc).
        """
        torch.save(self, path)

    @staticmethod
    def load_full(path: str, device="cpu", dtype=None):
        """
        Carga el ModelManager COMPLETO sin reconstruir nada.
        - device: dónde quieres dejarlo (independiente del guardado).
        - dtype: si lo pasas, fuerza dtype final (opcional).
        """
        # map_location mueve los tensores al device durante el load
        model = torch.load(path, map_location=device, weights_only=False)

        if not isinstance(model, ModelManager):
            raise TypeError(f"El archivo no contiene un ModelManager. Tipo: {type(model)}")

        # Asegura coherencia de atributos + mueve módulos
        model.device = device
        if dtype is not None:
            model.dtype = dtype

        model = model.to(device)
        if dtype is not None:
            model = model.to(dtype)

        # Por si tu NN se usa directamente
        model.NN = model.NN.to(device)
        if dtype is not None:
            model.NN = model.NN.to(dtype)

        return model