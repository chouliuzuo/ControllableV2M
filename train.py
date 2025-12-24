import torch
import torch.nn as nn
import yaml
import schedulefree
import torch.optim as optim
from tqdm import tqdm
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from dataloader.v2m import MultiModalDataset
from model.v2m_new import V2M
import os
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.multiprocessing as mp
from torch.cuda.amp import autocast, GradScaler
import typing as tp

class TorchAutocast:
    def __init__(self, enabled: bool, *args, **kwargs):
        self.autocast = torch.autocast(*args, **kwargs) if enabled else None

    def __enter__(self):
        if self.autocast is None:
            return
        try:
            self.autocast.__enter__()
        except RuntimeError:
            device = self.autocast.device
            dtype = self.autocast.fast_dtype
            raise RuntimeError(
                f"There was an error autocasting with dtype={dtype} device={device}\n"
                "If you are on the FAIR Cluster, you might need to use autocast_dtype=float16"
            )

    def __exit__(self, *args, **kwargs):
        if self.autocast is None:
            return
        self.autocast.__exit__(*args, **kwargs)


def _compute_cross_entropy(
     logits: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor = None,
    ) -> tp.Tuple[torch.Tensor, tp.List[torch.Tensor]]:
        """Compute cross entropy between multi-codebook targets and model's logits.
        The cross entropy is computed per codebook to provide codebook-level cross entropy.
        Valid timesteps for each of the codebook are pulled from the mask, where invalid
        timesteps are set to 0.

        Args:
            logits (torch.Tensor): Model's logits of shape [B, K, T, card].
            targets (torch.Tensor): Target codes, of shape [B, K, T].
            mask (torch.Tensor): Mask for valid target codes, of shape [B, K, T].
        Returns:
            ce (torch.Tensor): Cross entropy averaged over the codebooks
            ce_per_codebook (list of torch.Tensor): Cross entropy per codebook (detached).
        """
        B, K, T = targets.shape
        assert logits.shape[:-1] == targets.shape
        assert mask.shape == targets.shape
        ce = torch.zeros([], device=targets.device)
        ce_per_codebook: tp.List[torch.Tensor] = []
        for k in range(K):
            logits_k = logits[:, k, ...].contiguous().view(-1, logits.size(-1))  # [B x T, card]
            targets_k = targets[:, k, ...].contiguous().view(-1)  # [B x T]
            mask_k = mask[:, k, ...].contiguous().view(-1)  # [B x T]
            ce_targets = targets_k[mask_k]
            ce_logits = logits_k[mask_k]
            q_ce = F.cross_entropy(ce_logits, ce_targets)
            ce += q_ce
            ce_per_codebook.append(q_ce.detach())
        # average cross entropy across codebooks
        ce = ce / K
        return ce#, ce_per_codebook

def train(rank, o_model, train_dataset, test_dataset, model_config):
    o_model.cuda(rank)
    model = o_model
    scaler = GradScaler()
    optimizer = schedulefree.AdamWScheduleFree(model.parameters(), lr=model_config['train']['lr'],  eps=1e-8, weight_decay=0.1)
    if rank == 0:
        tensorboard_dir = "./checkpoint/tensorboard"
        writer = SummaryWriter(tensorboard_dir)
    save = model_config['train']['save_per_epoch']
    batch_size = model_config['train']['batch_size']
    n_epochs = model_config['train']['epochs']

    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, 
                              shuffle=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, 
                              shuffle=True)
    
    for epoch in range(n_epochs):
        pbar = tqdm(
            total=int(len(train_dataset) / batch_size / world_size) + 1,
            desc=f"Epoch {epoch + 1}/{n_epochs}",
            postfix=dict,
        )
        model.train()
        optimizer.train()
        train_loss = 0.0
        train_target_loss = 0.0
        train_predict_loss = 0.0
        train_music_loss = 0.0
        train_pitch_loss = 0.0
        train_loudness_loss = 0.0
        train_chroma_loss = 0.0
        train_spectral_loss = 0.0
        for iteration, item in enumerate(train_loader):
            video, music = item
            semantics, colors, starts, moves, areas = video
            semantics = semantics.cuda(rank)
            colors = colors.cuda(rank)
            starts = starts.cuda(rank)
            moves = moves.cuda(rank)
            areas = areas.cuda(rank)

            pitch, chroma, loudness, spectral_centroid, music_target = music
            music_target = music_target.squeeze(1).cuda(rank)
            loudness = loudness.cuda(rank)
            pitch = pitch.cuda(rank)
            chroma = chroma.cuda(rank)
            spectral_centroid = spectral_centroid.cuda(rank)

            with TorchAutocast(enabled=True, device_type='cuda', dtype=torch.bfloat16):
                pitch_prediction, chroma_prediction, loudness_prediction, spectral_prediction, music_out, gt_loss, pre_loss = model(semantics, colors, starts, moves, areas, music_target)
                # pitch_prediction, chroma_prediction, loudness_prediction, spectral_prediction, music_out, gt_loss, pre_loss = model(video, music_target)
                logits = music_out.logits
                mask = music_out.mask
                if model_config['train']['loss_type'] == 'l1':
                    loss = nn.L1Loss()
                elif model_config['train']['loss_type'] == 'l2':
                    loss = nn.MSELoss()
                else:
                    raise NotImplementedError

                music_loss = _compute_cross_entropy(logits, music_target, mask)
                target_loss = gt_loss
                predict_loss = pre_loss
                pitch_loss = loss(pitch, pitch_prediction)
                loudness_loss = loss(loudness, loudness_prediction)
                chroma_loss = loss(chroma, chroma_prediction)
                spectral_loss = loss(spectral_centroid, spectral_prediction)
            
                total_loss = music_loss + pitch_loss + loudness_loss + chroma_loss+ target_loss + predict_loss + spectral_loss
                pbar.set_postfix(
                    **{
                        "train": total_loss.item(),
                        "music": music_loss.item(),
                        "target": target_loss.item(),
                        "predict": predict_loss.item(),
                        "pitch": pitch_loss.item(),
                        "loud": loudness_loss.item(),
                        "chroma": chroma_loss.item(),
                        "spec": spectral_loss.item(),
                    }
                )
                train_loss += total_loss.item()
                train_music_loss += music_loss.item()
                train_target_loss += target_loss.item()
                train_predict_loss += predict_loss.item()
                train_pitch_loss += pitch_loss.item()
                train_loudness_loss += loudness_loss.item()
                train_chroma_loss += chroma_loss.item()
                train_spectral_loss += spectral_loss.item()
            
            scaler.scale(total_loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            pbar.update(1)
            
        if rank==0:
            writer.add_scalar(
                "loss / train_loss", train_loss / len(train_loader), global_step=epoch
            )
            writer.add_scalar(
                "loss / train_music_loss", train_music_loss / len(train_loader), global_step=epoch
            )
            writer.add_scalar(
                "loss / train_target_loss", train_target_loss / len(train_loader), global_step=epoch
            )
            writer.add_scalar(
                "loss / train_predict_loss", train_predict_loss / len(train_loader), global_step=epoch
            )
            writer.add_scalar(
                "loss / train_pitch_loss", train_pitch_loss / len(train_loader), global_step=epoch
            )
            writer.add_scalar(
                "loss / train_loudness_loss", train_loudness_loss / len(train_loader), global_step=epoch
            )
            writer.add_scalar(
                "loss / train_chroma_loss", train_chroma_loss / len(train_loader), global_step=epoch
            )
            writer.add_scalar(
                "loss / train_spectral_loss", train_spectral_loss / len(train_loader), global_step=epoch
            )
        pbar.close()
        with torch.no_grad():
            pbar = tqdm(
                total=int(len(test_dataset) / batch_size / world_size) + 1,
                desc=f"Epoch {epoch + 1}/{n_epochs}",
                postfix=dict,
                mininterval=0.3,
            )
            model.eval()
            optimizer.eval()
            test_loss = 0.0
            test_music_loss = 0.0
            test_target_loss = 0.0
            test_predict_loss = 0.0
            test_pitch_loss = 0.0
            test_loudness_loss = 0.0
            test_chroma_loss = 0.0
            test_spectral_loss = 0.0
            for iteration, item in enumerate(test_loader):
                video, music = item
                semantics, colors, starts, moves, areas = video
                semantics = semantics.cuda(rank)
                colors = colors.cuda(rank)
                starts = starts.cuda(rank)
                moves = moves.cuda(rank)
                areas = areas.cuda(rank)
                
                
                pitch, chroma, loudness, spectral_centroid, music_target = music
                music_target = music_target.squeeze(1).cuda(rank)
                loudness = loudness.cuda(rank)
                pitch = pitch.cuda(rank)
                chroma = chroma.cuda(rank)
                spectral_centroid = spectral_centroid.cuda(rank)
                with TorchAutocast(enabled=True, device_type='cuda', dtype=torch.bfloat16):
                    pitch_prediction, chroma_prediction, loudness_prediction, spectral_prediction, music_out, gt_loss, pre_loss = model(semantics, colors, starts, moves, areas, music_target)

                    logits = music_out.logits
                    mask = music_out.mask
                    music_loss = _compute_cross_entropy(logits, music_target, mask).item()
                    target_loss = gt_loss.item()
                    predict_loss = pre_loss.item()
                    pitch_loss = loss(pitch, pitch_prediction).item()
                    loudness_loss = loss(loudness, loudness_prediction).item()
                    chroma_loss = loss(chroma, chroma_prediction).item()
                    spectral_loss = loss(spectral_centroid, spectral_prediction).item()

                    total_loss = music_loss + target_loss + predict_loss+ pitch_loss + loudness_loss + chroma_loss + spectral_loss 
                    pbar.set_postfix(
                        **{
                            "test": total_loss,
                            "music": music_loss,
                            "target": target_loss,
                            "predict": predict_loss,
                            "pitch": pitch_loss,
                            "loud": loudness_loss,
                            "chroma": chroma_loss,
                            "spec": spectral_loss,
                        }
                    )
                    test_loss += total_loss
                    test_music_loss += music_loss
                    test_target_loss += target_loss
                    test_predict_loss += predict_loss
                    test_pitch_loss += pitch_loss
                    test_loudness_loss += loudness_loss
                    test_chroma_loss += chroma_loss
                    test_spectral_loss += spectral_loss
                    
        if rank==0:
            writer.add_scalar(
                "loss / test_loss", test_loss / len(test_loader), global_step=epoch
            )
            writer.add_scalar(
                "loss / test_target_loss", test_target_loss / len(test_loader), global_step=epoch
            )
            writer.add_scalar(
                "loss / test_predict_loss", test_predict_loss / len(test_loader), global_step=epoch
            )
            writer.add_scalar(
                "loss / test_music_loss", test_music_loss / len(test_loader), global_step=epoch
            )
            writer.add_scalar(
                "loss / test_pitch_loss", test_pitch_loss / len(test_loader), global_step=epoch
            )
            writer.add_scalar(
                "loss / test_loudness_loss", test_loudness_loss / len(test_loader), global_step=epoch
            )
            writer.add_scalar(
                "loss / test_chroma_loss", test_chroma_loss / len(test_loader), global_step=epoch
            )
            writer.add_scalar(
                "loss / test_spectral_loss", test_spectral_loss / len(test_loader), global_step=epoch
            )
        pbar.close()
        if (epoch + 1) % save == 0 and rank==0:
            checkpoint = {
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scaler_state_dict': scaler.state_dict(),
            }
            torch.save(checkpoint, f"./checkpoint/{epoch+1}.pth")

if __name__ == "__main__":
    world_size = 1
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

    model_config = yaml.load(open("./config/v2m.yaml", "r"), Loader=yaml.FullLoader)
    dataset = MultiModalDataset(model_config['path']['video_path'],
                           model_config['path']['preprocessed_path'],
                                    model_config['path']['music_target_path'])

    # train test split
    torch.manual_seed(0)
    train_size = int(0.85 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(
        dataset, [train_size, test_size]
    )

    model = V2M(model_config)
    
    # dict = torch.load('./state_dict.bin')
    # model.music_generator.decoder.load_state_dict(dict['best_state'], strict=False)

    train(0, model, train_dataset, test_dataset, model_config)