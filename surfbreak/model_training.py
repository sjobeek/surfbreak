import torch
import pytorch_lightning as pl
from surfbreak.waveform_models import LitSirenNet, LitWaveCNN
from surfbreak.datasets import WaveformVideoDataset, WaveformChunkDataset, InferredWaveformDataset
from datetime import datetime


wf_labeling_training_video = '../data/shirahama_1590387334_SURF-93cm.ts'

def train_basic_waveform_model(training_video, max_epochs=20):
    wf_net_kwargs = dict(
        hidden_features=256,
        hidden_layers=3,
        first_omega_0=2.5,
        hidden_omega_0=11,
        squared_slowness=0.20,
        steps_per_vid_chunk=100,
        learning_rate=1e-4,
        grad_loss_scale=0,
        wavefunc_loss_scale=5.5e-9,
        wavespeed_loss_scale=4e-4, 
        xrange=(0,400),
        timerange=(0,3*30),
        chunk_duration=30,
        chunk_stride=30
    )

    wf_train_video_dataset = WaveformVideoDataset(wf_labeling_training_video, ydim=120, xrange=wf_net_kwargs['xrange'], timerange=wf_net_kwargs['timerange'], 
                                                  time_chunk_duration_s=wf_net_kwargs['chunk_duration'], time_chunk_stride_s=wf_net_kwargs['chunk_stride'], time_axis_scale=0.5)

    wavefunc_model = LitSirenNet(wf_labeling_training_video, **wf_net_kwargs, vid_dataset=wf_train_video_dataset)

    tb_logger = pl.loggers.TensorBoardLogger('logs/', name="pipeline_wf")
    trainer = pl.Trainer(logger=tb_logger, limit_val_batches=3,
                         max_epochs=max_epochs, # 20 
                         gpus=1 if torch.cuda.is_available() else None,)

    pl.seed_everything(42)
    trainer.fit(wavefunc_model)
    
    now = datetime.now() # current date and time
    date_time = now.strftime("%m-%d-%Y_%H-%M-%S")
    checkpoint_filepath = '__graphchain_cache__/pipeline_wf_'+date_time+'.ckpt'
    trainer.save_checkpoint(checkpoint_filepath)
    
    return checkpoint_filepath


def train_wavefront_detection_cnn(video_filepath, wf_model_checkpoint, max_epochs=20):
    pl.seed_everything(42)
                                 # Params from optimization run: fo 2.4967  ho 10.969  ss 0.20492  wfls 5.4719e-9  wsls 0.00043457
    wavecnn_model = LitWaveCNN(video_filepath=video_filepath, 
                                 wf_model_checkpoint=wf_model_checkpoint, 
                                 learning_rate=1e-4, xrange=(0,400), timerange=(0,90), chunk_duration=30, chunk_stride=30,
                                 n_input_channels=2)

    tb_logger = pl.loggers.TensorBoardLogger('logs/', name="pipeline_cnn")
    trainer = pl.Trainer(logger=tb_logger, limit_val_batches=3,
                         max_epochs=max_epochs, 
                         gpus=1 if torch.cuda.is_available() else None,)


    trainer.fit(wavecnn_model)
    
    now = datetime.now() # current date and time
    date_time = now.strftime("%m-%d-%Y_%H-%M-%S")
    checkpoint_filepath = '__graphchain_cache__/pipeline_cnn_'+date_time+'.ckpt'
    trainer.save_checkpoint(checkpoint_filepath)
    
    return checkpoint_filepath