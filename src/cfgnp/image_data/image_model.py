import copy
import torch
import torch.nn as nn

from cfgnp.graph_approach.causal_graph import CausalGraphFactory
from medical_diffusion.models.embedders.latent_embedders import VAE
from cfgnp.models import GNNCFNPSeq, CFSamplingModel
from cfgnp.models.size_invariant_model import CFGNPSizeInvariant
from cfgnp.util.util import IMAGE_RESIZE

USE_SPLIT_OBS_ENCODING = True
OBS_ENCODING_SPLITS = 3


def factual_image_from_batch(batch, num_features: int) -> torch.Tensor:
    """The factual X-ray of every graph in `batch`, as `[B, 3, IMAGE_RESIZE, IMAGE_RESIZE]`.
    """
    x_orig = batch.x_orig.reshape((batch.batch_size, -1, 3))
    return x_orig[:, num_features:].reshape((-1, IMAGE_RESIZE, IMAGE_RESIZE, 3)).permute(0, 3, 1, 2)


class VisionCFNP(nn.Module):
    def __init__(
        self,
        vae_path,
        target_classifiers,  # dict[int, nn.Module] or list[nn.Module]
        num_features,
        output_nodes,
        node_dim,
        in_features: int,
        hidden_dim: int,
        num_nodes: int,
        num_obs: int,
        num_count,
        iterations=3,
        mog_comp=3,
        num_heads_att=5,
        freeze_vae: bool = True,
        freeze_classifiers: bool = True,
        size_invariant: bool = True,
    ):
        super().__init__()
        self.num_features = num_features

        # ---- VAE ----
        ckpt = torch.load(vae_path, map_location="cpu", weights_only=False)

        self.vae = VAE(
            in_channels=3,
            out_channels=3,
            emb_channels=8,
            spatial_dims=2,
            hid_chs=[64, 128, 256, 512],
            kernel_sizes=[3, 3, 3, 3],
            strides=[1, 2, 2, 2],
            deep_supervision=1,
            use_attention="none",
            loss=torch.nn.MSELoss,
            embedding_loss_weight=1e-6,
        )
        self.vae.load_state_dict(ckpt["state_dict"])

        self.freeze_vae = freeze_vae
        if freeze_vae:
            for p in self.vae.parameters():
                p.requires_grad = False
            self.vae.eval()

        # ---- CFNP ----
        if size_invariant:
            model_class = CFGNPSizeInvariant
        else:
            model_class = GNNCFNPSeq
        self.cfnp = model_class(
            in_features=in_features,
            hidden_dim=hidden_dim,
            num_nodes=num_nodes,
            num_obs=num_obs,
            num_count=num_count,
            iterations=iterations,
            mog_comp=mog_comp,
            num_heads_att=num_heads_att,
        )
        self.cfnp_sample = CFSamplingModel(self.cfnp)

        self.match_dim = nn.Sequential(
            nn.Linear(3, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
        )

        # ---- Classifiers ----
        if isinstance(target_classifiers, dict):
            sorted_items = sorted(target_classifiers.items())
            self.target_classifiers = nn.ModuleList([m for _, m in sorted_items])
        else:
            self.target_classifiers = nn.ModuleList(target_classifiers)

        # Store class counts from classifiers
        self.class_counts = []
        for i, clf in enumerate(self.target_classifiers):
            if not hasattr(clf, "num_classes"):
                raise ValueError(f"Classifier {i} must have attribute `num_classes`")
            self.class_counts.append(int(clf.num_classes))

        self.max_classes = max(self.class_counts)

        if freeze_classifiers:
            for p in self.target_classifiers.parameters():
                p.requires_grad = False

    def _encode_obs_images(self, obs_image):
        """
        Encode obs images either in one pass or split into chunks depending on
        the global USE_SPLIT_OBS_ENCODING flag.
        """
        if not USE_SPLIT_OBS_ENCODING:
            with torch.no_grad():
                with torch.autocast(device_type="cuda", enabled=False):
                    return self.vae.encode(obs_image).flatten(2)

        encoded_chunks = []
        with torch.no_grad():
            for chunk in torch.chunk(obs_image, OBS_ENCODING_SPLITS, dim=0):
                if chunk.numel() == 0:
                    continue
                with torch.autocast(device_type="cuda", enabled=False):
                    encoded_chunks.append(self.vae.encode(chunk).flatten(2))

        return torch.cat(encoded_chunks, dim=0)

    def forward(self, batch):
        """
        Returns:
            logits:     [B, 1, T, C_max, 1, 1]
            output_img: [B, 3, IMAGE_RESIZE, IMAGE_RESIZE], the decoded counterfactual
        """
        output_img = self.predict_image(batch)
        # Need to unsqueeze the target predictions as we usually have mog output but the classified targets do not
        logits = self.classify_targets_padded(output_img).unsqueeze(1).unsqueeze(-1).unsqueeze(-1)  # [B, 1, T, C_max, 1, 1]
        return logits, output_img

    def classify_targets_padded(self, output_img):
        B = output_img.shape[0]
        T = len(self.target_classifiers)
        C_max = self.max_classes

        device = output_img.device
        dtype = output_img.dtype

        logits_padded = torch.zeros(B, T, C_max, device=device, dtype=dtype)

        for t, (clf, c) in enumerate(zip(self.target_classifiers, self.class_counts)):
            logits = clf(output_img)  # [B, c]

            if logits.shape[-1] != c:
                raise ValueError(
                    f"Classifier {t} output mismatch: got {logits.shape[-1]}, expected {c}"
                )

            logits_padded[:, t, :c] = logits

        return logits_padded

    def predict_image(self, batch):
        x_orig = batch.x_orig.reshape((batch.batch_size, -1, 3))
        x_obs = batch.x_obs.reshape((batch.batch_size, -1, IMAGE_RESIZE * IMAGE_RESIZE + self.num_features, 3))
        x_obs = x_obs.nan_to_num(nan=0.0, posinf=0.0, neginf=0.0)
        x_int = batch.x_int.reshape((batch.batch_size, -1, 3))
        num_obs = x_obs.shape[1]

        # extract image features. other features are tripled due to channel number 3.
        z_image = factual_image_from_batch(batch, self.num_features)  # (B, 3, IMAGE_RESIZE, IMAGE_RESIZE)
        obs_image = x_obs[:, :, self.num_features:].reshape((-1, IMAGE_RESIZE, IMAGE_RESIZE, 3)).permute(0, 3, 1, 2)  # (B * num_obs, 3, IMAGE_RESIZE, IMAGE_RESIZE)
        int_image = x_int[:, self.num_features:].reshape((-1, IMAGE_RESIZE, IMAGE_RESIZE, 3)).permute(0, 3, 1, 2)  # (B, 3, IMAGE_RESIZE, IMAGE_RESIZE)

        with torch.no_grad():
            z_image_enc = self.vae.encode(z_image).flatten(2)   # (B, 8, 16*16)
            int_image_enc = self.vae.encode(int_image).flatten(2)

        obs_image_enc = self._encode_obs_images(obs_image)
        if obs_image_enc.isnan().any():
            raise ValueError("NaN values found in obs_image_enc after encoding.")

        z_match = self.match_dim(x_orig[:, :self.num_features])
        int_match = self.match_dim(x_int[:, :self.num_features])

        obs_flat = x_obs.reshape((x_obs.shape[0] * x_obs.shape[1], -1, 3))
        obs_match = self.match_dim(obs_flat[:, :self.num_features])

        z_enc = torch.cat(
            [z_match[:, :self.num_features, :z_image_enc.shape[-1]], z_image_enc],
            dim=1,
        )
        int_enc = torch.cat(
            [int_match[:, :self.num_features, :int_image_enc.shape[-1]], int_image_enc],
            dim=1,
        )
        obs_enc = torch.cat(
            [
                obs_match[:, :self.num_features, :obs_image_enc.shape[-1]],
                obs_image_enc,
            ],
            dim=1,
        )
        if obs_enc.isnan().any():
            raise ValueError("NaN values found in obs_enc after concatenation.")

        # Avoid mutating original batch
        batch_local = copy.copy(batch)
        batch_local.x_orig = z_enc
        batch_local.x_obs = obs_enc
        batch_local.x_int = int_enc
        batch_local.batch = torch.arange(batch_local.batch_size, device=batch_local.x_orig.device).repeat_interleave(z_enc.shape[1] * 2)

        single_dmsm = CausalGraphFactory.get_causal_graph("full").dual_graph(z_enc.shape[1])
        num_nodes = z_enc.shape[1] * 2
        E = single_dmsm.size(1)

        offsets = torch.arange(batch_local.batch_size, device=single_dmsm.device) * num_nodes
        offsets = offsets.repeat_interleave(E)

        edge_index = single_dmsm.repeat(1, batch_local.batch_size)
        edge_index += offsets.unsqueeze(0)

        model_output = self.cfnp_sample(batch_local, True)
        delta_enc = model_output.squeeze(1)[:, self.num_features:]
        output_img_enc = z_image_enc + delta_enc

        lat_mean = z_image_enc.mean(dim=-1, keepdim=True)
        lat_std = z_image_enc.std(dim=-1, keepdim=True)
        out_mean = output_img_enc.mean(dim=-1, keepdim=True)
        out_std = output_img_enc.std(dim=-1, keepdim=True)
        output_img_enc = (output_img_enc - out_mean) / (out_std + 1e-6) * lat_std + lat_mean

        output_img = self.vae.decode(output_img_enc.reshape(output_img_enc.shape[0], 8, 16, 16))
        output_img = torch.tanh(output_img)

        return output_img