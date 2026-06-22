"""
models/ser_model.py — CNN-BiLSTM-Attention model for Speech Emotion Recognition.

Architecture flow:
    Input          (B, 1, 120, 130)   — 1 channel, 120 freq bins, 130 time frames
    ↓ CNN Extractor                   — learns local spectro-temporal patterns
    ↓ Reshape      (B, T', C'×H')     — convert CNN feature maps to sequences
    ↓ BiLSTM       (B, T', 512)       — capture long-range temporal dynamics
    ↓ Self-Attention (B, T', 512)     — weight emotionally salient frames
    ↓ Global Avg Pool (B, 512)        — aggregate over time
    ↓ FC + Dropout  (B, 256)
    Output         (B, 8)             — logits for 8 emotion classes

Why CNN first?
    MFCCs are 2D maps (frequency × time). CNNs excel at detecting local
    patterns (formant transitions, pitch rises) just like they detect edges in
    images — before the LSTM handles the global sequence.

Why BiLSTM?
    After CNN flattens each time-step, BiLSTM reads the sequence in both
    directions so each hidden state sees past AND future context — critical
    for emotion that unfolds non-linearly over time.
"""
import torch
import torch.nn as nn

from models.attention import SelfAttention
from config import Config


class ConvBlock(nn.Module):
    """
    One convolutional block:  Conv2d → BatchNorm → ReLU → MaxPool → Dropout2d

    padding="same" preserves spatial dimensions through the conv layer;
    MaxPool halves them.  After 4 blocks with pool=(2,2):
        Height: 120 → 60 → 30 → 15 → 7
        Width : 130 → 65 → 32 → 16 → 8
    """

    def __init__(
        self,
        in_channels:  int,
        out_channels: int,
        kernel_size:  tuple = Config.KERNEL_SIZE,
        pool_size:    tuple = Config.POOL_SIZE,
        dropout:      float = 0.2,
    ) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(
                in_channels, out_channels,
                kernel_size=kernel_size,
                padding="same",
                bias=False,              # BatchNorm absorbs the bias term
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=pool_size),
            nn.Dropout2d(p=dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class SERModel(nn.Module):
    """
    Speech Emotion Recognition — CNN-BiLSTM-Attention classifier.

    Args:
        num_classes:   Number of emotion classes (default 8).
        feature_dim:   MFCC feature height, i.e. n_mfcc×3 = 120.
        time_frames:   Number of time frames = 130.
        cnn_channels:  Channel list for each Conv block.
        lstm_hidden:   BiLSTM hidden size per direction.
        lstm_layers:   Number of stacked BiLSTM layers.
        attention_dim: Projection dim inside self-attention.
        fc_hidden:     Hidden size of the penultimate FC layer.
        dropout:       Dropout probability in the classifier head.
    """

    def __init__(
        self,
        num_classes:   int   = Config.NUM_CLASSES,
        feature_dim:   int   = Config.FEATURE_DIM,
        time_frames:   int   = Config.TIME_FRAMES,
        cnn_channels:  list  = Config.CNN_CHANNELS,
        lstm_hidden:   int   = Config.LSTM_HIDDEN,
        lstm_layers:   int   = Config.LSTM_LAYERS,
        attention_dim: int   = Config.ATTENTION_DIM,
        fc_hidden:     int   = Config.FC_HIDDEN,
        dropout:       float = Config.DROPOUT,
    ) -> None:
        super().__init__()

        # ── CNN Extractor ─────────────────────────────────────────────────────
        cnn_blocks = []
        in_ch = 1
        for out_ch in cnn_channels:
            cnn_blocks.append(ConvBlock(in_ch, out_ch, dropout=0.2))
            in_ch = out_ch
        self.cnn = nn.Sequential(*cnn_blocks)

        # Compute CNN output dimensions with a dummy forward pass
        with torch.no_grad():
            dummy = torch.zeros(1, 1, feature_dim, time_frames)
            cnn_out = self.cnn(dummy)                       # (1, C, H', W')
            _, C, H, W = cnn_out.shape
            self._cnn_flat = C * H       # features per time-step fed to LSTM
            self._cnn_time = W           # number of time-steps into LSTM

        # ── Bidirectional LSTM ────────────────────────────────────────────────
        self.lstm = nn.LSTM(
            input_size=self._cnn_flat,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=True,
            dropout=Config.LSTM_DROPOUT if lstm_layers > 1 else 0.0,
        )
        lstm_out_dim = lstm_hidden * 2            # ×2 for bidirectional

        # ── Self-Attention ────────────────────────────────────────────────────
        self.attention = SelfAttention(lstm_out_dim, attention_dim)

        # ── Classifier Head ───────────────────────────────────────────────────
        self.classifier = nn.Sequential(
            nn.Linear(lstm_out_dim, fc_hidden),
            nn.LayerNorm(fc_hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(fc_hidden, num_classes),
        )

        # Store config for checkpoint serialisation
        self.hparams = dict(
            num_classes=num_classes, feature_dim=feature_dim,
            time_frames=time_frames, cnn_channels=cnn_channels,
            lstm_hidden=lstm_hidden, lstm_layers=lstm_layers,
            attention_dim=attention_dim, fc_hidden=fc_hidden,
            dropout=dropout,
        )

        self._init_weights()

    # ─── Weight Init ──────────────────────────────────────────────────────────
    def _init_weights(self) -> None:
        """Kaiming-Normal for Conv/Linear (ReLU nets); zeros for biases."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    # ─── Forward Pass ─────────────────────────────────────────────────────────
    def forward(self, x: torch.Tensor) -> tuple:
        """
        Args:
            x: Input tensor of shape (B, 1, feature_dim, time_frames).

        Returns:
            logits:  (B, num_classes)
            weights: (B, T, T)  — attention weight matrix for inspection.
        """
        B = x.size(0)

        # 1. CNN: (B,1,F,T) → (B,C,H',W')
        cnn_out = self.cnn(x)
        _, C, H, W = cnn_out.shape

        # 2. Reshape to sequence: (B, W', C×H')
        seq = cnn_out.permute(0, 3, 1, 2).reshape(B, W, C * H)

        # 3. BiLSTM: (B,T,D_in) → (B,T,D_lstm)
        lstm_out, _ = self.lstm(seq)

        # 4. Self-Attention: (B,T,D) → (B,T,D)
        attended, attn_weights = self.attention(lstm_out)

        # 5. Global average pooling over time: (B,T,D) → (B,D)
        pooled = attended.mean(dim=1)

        # 6. Classify: (B,D) → (B,num_classes)
        logits = self.classifier(pooled)

        return logits, attn_weights

    # ─── Utility ──────────────────────────────────────────────────────────────
    def count_parameters(self) -> int:
        """Return total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def summary(self) -> None:
        """Print a compact architecture summary."""
        total = self.count_parameters()
        print(f"\n{'─'*50}")
        print(f"  SERModel — CNN-BiLSTM-Attention")
        print(f"{'─'*50}")
        print(f"  CNN output   : (B, {self._cnn_time}, {self._cnn_flat})")
        print(f"  BiLSTM out   : (B, {self._cnn_time}, {Config.LSTM_HIDDEN*2})")
        print(f"  Classifier   : {Config.LSTM_HIDDEN*2} → {Config.FC_HIDDEN} → {Config.NUM_CLASSES}")
        print(f"  Parameters   : {total:,}")
        print(f"  Device       : {Config.DEVICE}")
        print(f"{'─'*50}\n")