import torch
import pytorch_lightning as pl
from surfbreak.waveform_models import WaveformNet, LitWaveCNN
from surfbreak.datasets import WaveformVideoDataset, WaveformChunkDataset, InferredWaveformDataset
from datetime import datetime


wf_labeling_training_video = '../data/shirahama_1590387334_SURF-93cm.ts'

# TODO: Test this replacement (from nbs/11_simple_wave_detector.ipynb)
def train_wavefront_detection_cnn(video_filepath, wf_model_checkpoint, max_epochs=20):
    pl.seed_everything(42)

    tb_logger = pl.loggers.TensorBoardLogger('logs/', name="wavecnn")
    checkpoint_callback = ModelCheckpoint(save_top_k=2, verbose=True,
                                        monitor='val_loss', mode='min')

    trainer = pl.Trainer(logger=tb_logger, #limit_val_batches=15,
                        ccheckpoint_callbackoint_callback=checkpoint_callback,
                        max_epochs=20, 
                        gpus=1 if torch.cuda.is_available() else None,)

    wavecnn_model = LitWaveCNN(video_filepath=video_filepath, 
                               learning_rate=1e-4, timerange=(0,4*60), val_timerange=(4*60,5*60))
        
    trainer.fit(wavecnn_model)
    
    now = datetime.now() # current date and time
    date_time = now.strftime("%m-%d-%Y_%H-%M-%S")
    checkpoint_filepath = '__graphchain_cache__/pipeline_cnn_'+date_time+'.ckpt'
    trainer.save_checkpoint(checkpoint_filepath)
    
    return checkpoint_filepath