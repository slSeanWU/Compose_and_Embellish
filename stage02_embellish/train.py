import sys, os, time, random
sys.path.append('./model')

from dataloader import REMISkylineToMidiTransformerDataset, pickle_load
from torch.utils.data import DataLoader

from torch import optim
import torch
import numpy as np

import yaml
train_conf_path = sys.argv[1]
train_conf = yaml.load(open(train_conf_path, 'r'), Loader=yaml.FullLoader)

train_steps = 0
train_conf_: dict = train_conf['training']

gpuid = train_conf_['gpuid']
torch.cuda.set_device(gpuid)

warmup_steps = train_conf_['warmup_steps']
max_lr = train_conf_['lr']
min_lr = train_conf_['lr_scheduler']['eta_min']
lr_decay_steps = train_conf_['lr_scheduler']['T_max']
redraw_prob = train_conf_.get('feat_redraw_prob', 0.0)
max_epochs = train_conf_['num_epochs']
batch_size = train_conf['data_loader']['batch_size']
train_split = train_conf['data_loader']['train_split']
val_split = train_conf['data_loader']['val_split']

ckpt_dir = train_conf_['ckpt_dir']
pretrained_param_path = train_conf_['trained_params']
pretrained_optimizer_path = train_conf_['trained_optim']
ckpt_interval = train_conf_['ckpt_interval']
val_interval = 1
log_interval = train_conf_['log_interval']
accum_steps = train_conf_.get("accum_steps", 1)

use_chord_emb = False

def log_epoch(log_file, log_data, is_init=False):
  if is_init:
    with open(log_file, 'w') as f:
      f.write('{:4} {:8} {:12} {:12}\n'.format('ep', 'steps', 'recons_loss', 'ep_time'))

  with open(log_file, 'a') as f:
    f.write('{:<4} {:<8} {:<12} {:<12}\n'.format(
      log_data['ep'], log_data['steps'], round(log_data['recons_loss'], 5), round(log_data['time'], 2)
    ))


def train_model(
  epoch, model, dloader, optim, sched, model_type: str = "performer"
):
  model.train()
  recons_loss_rec = 0.
  accum_samples = 0

  print ('[epoch {:03d}] training ...'.format(epoch))
  print ('[epoch {:03d}] # batches = {}'.format(epoch, len(dloader)))
  st = time.time()

  for batch_idx, batch_samples in enumerate(dloader):
    batch_dec_inp = batch_samples['dec_input'].cuda(gpuid)
    batch_dec_tgt = batch_samples['dec_target'].cuda(gpuid)
    batch_track_mask = batch_samples['track_mask'].cuda(gpuid)
    batch_inp_lens = batch_samples['length']

    if use_chord_emb:
      batch_chord_mhot = batch_samples['chords_mhot'].cuda(gpuid).float()
    else:
      batch_chord_mhot = None

    global train_steps
    train_steps += 1

    # get logits from model
    if model_type == "performer":
      omit_feature_map_draw = random.random() > redraw_prob
      dec_logits = model(
                    batch_dec_inp, 
                    seg_inp=batch_track_mask,
                    chord_inp=batch_chord_mhot,
                    attn_kwargs={'omit_feature_map_draw': omit_feature_map_draw}
                  )
    elif model_type == "gpt2":
      omit_feature_map_draw = True  # dummy var here
      dec_logits = model(
                    batch_dec_inp,
                    seg_inp=batch_track_mask,
                    chord_inp=batch_chord_mhot,
                  )
    losses = model.compute_loss(dec_logits, batch_dec_tgt)
    
    # clip gradient & update model
    if accum_steps > 1:
      losses['total_loss'] /= accum_steps
    losses['total_loss'].backward()

    if (train_steps % accum_steps) == 0:
      torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
      optim.step()
      optim.zero_grad()
      recons_loss_rec += batch_samples['id'].size(0) * losses['recons_loss'].item() * accum_steps * accum_steps
      accum_samples += batch_samples['id'].size(0) * accum_steps

      print (' -- epoch {:03d} | batch {:03d}/{:03d}: len: {}\n   * loss = {:.4f}, step = {}, time_elapsed = {:.2f} secs | redraw: {}'.format(
        epoch, batch_idx + 1, len(dloader), batch_inp_lens, 
        recons_loss_rec / accum_samples, train_steps,
        time.time() - st, (not omit_feature_map_draw)
      ))

    # anneal learning rate
    if (train_steps // accum_steps) < warmup_steps:
      curr_lr = max_lr * train_steps / (warmup_steps * accum_steps)
      optim.param_groups[0]['lr'] = curr_lr
    else:
      sched.step((train_steps // accum_steps - warmup_steps))


    if not train_steps % log_interval:
      log_data = {
        'ep': epoch,
        'steps': train_steps,
        'recons_loss': recons_loss_rec / accum_samples,
        'time': time.time() - st
      }
      log_epoch(
        os.path.join(ckpt_dir, 'log.txt'), log_data, is_init=not os.path.exists(os.path.join(ckpt_dir, 'log.txt'))
      )
  
  print ('[epoch {:03d}] training completed\n  -- loss = {:.4f}\n  -- time elapsed = {:.2f} secs.'.format(
    epoch, recons_loss_rec / accum_samples, time.time() - st
  ))
  log_data = {
    'ep': epoch,
    'steps': train_steps,
    'recons_loss': recons_loss_rec / accum_samples,
    'time': time.time() - st
  }
  log_epoch(
    os.path.join(ckpt_dir, 'log.txt'), log_data, is_init=not os.path.exists(os.path.join(ckpt_dir, 'log.txt'))
  )

  return recons_loss_rec / accum_samples

def validate(model, dloader, rounds=1, model_type="performer"):
  model.eval()
  loss_rec = []

  with torch.no_grad():
    for r in range(rounds):
      print (' >> validating ... (round {})'.format(r+1))
      for batch_idx, batch_samples in enumerate(dloader):
        batch_dec_inp = batch_samples['dec_input'].cuda(gpuid)
        batch_dec_tgt = batch_samples['dec_target'].cuda(gpuid)
        batch_track_mask = batch_samples['track_mask'].cuda(gpuid)

        if use_chord_emb:
          batch_chord_mhot = batch_samples['chords_mhot'].cuda(gpuid).float()
        else:
          batch_chord_mhot = None

        if model_type == "performer":
          omit_feature_map_draw = random.random() > redraw_prob
          dec_logits = model(
                        batch_dec_inp,
                        seg_inp=batch_track_mask,
                        chord_inp=batch_chord_mhot,
                        attn_kwargs={'omit_feature_map_draw': omit_feature_map_draw}
                      )
        elif model_type == "gpt2":
          dec_logits = model(
                        batch_dec_inp,
                        seg_inp=batch_track_mask,
                        chord_inp=batch_chord_mhot,
                      )

        losses = model.compute_loss(dec_logits, batch_dec_tgt)
        if not batch_idx % 5:
          print ('  -- batch #{:03d}: valloss = {:.3f}'.format(
                  batch_idx + 1, losses['recons_loss'].item()
                ))
        loss_rec.append(losses['recons_loss'].item())
    
  return loss_rec

if __name__ == "__main__":
  model_conf = train_conf['model']
  model_type = model_conf.get("type", "performer")

  if model_type == "performer":
    from model.music_performer import MusicPerformer
    model_klass = MusicPerformer
  elif model_type == "gpt2":
    from model.music_gpt2 import MusicGPT2
    model_klass = MusicGPT2
  else:
    raise NotImplementedError("Unsuppported model:", model_type)

  if 'use_chord_emb' in model_conf and model_conf['use_chord_emb'] == True:
    use_chord_emb = True
  print ('[use chord emb]', use_chord_emb)

  dset = REMISkylineToMidiTransformerDataset(
    train_conf['data_loader']['data_path'],
    train_conf['data_loader']['vocab_path'], 
    do_augment=train_conf['data_loader']['do_augment'] if 'do_augment' in train_conf['data_loader'] else True, 
    model_dec_seqlen=train_conf['model']['max_len'], 
    pieces=pickle_load(train_split),
    pad_to_same=True,
    use_chord_mhot=use_chord_emb
  )
  val_dset = REMISkylineToMidiTransformerDataset(
    train_conf['data_loader']['data_path'],
    train_conf['data_loader']['vocab_path'], 
    do_augment=False, 
    model_dec_seqlen=train_conf['model']['max_len'], 
    pieces=pickle_load(val_split),
    pad_to_same=True,
    use_chord_mhot=use_chord_emb
  )
  print ('[info] # training pieces:', len(dset.pieces), dset.do_augment)
  dloader = DataLoader(
              dset, batch_size=batch_size, shuffle=True, num_workers=8
            )
  val_dloader = DataLoader(
                  val_dset, batch_size=batch_size, shuffle=True, num_workers=8
                )

  if model_type == "performer":
    model = model_klass(
        dset.vocab_size, model_conf['n_layer'], model_conf['n_head'], 
        model_conf['d_model'], model_conf['d_ff'], model_conf['d_embed'],
        use_segment_emb=model_conf['use_segemb'], n_segment_types=2,
        favor_feature_dims=model_conf['feature_map']['n_dims'],
        use_chord_mhot_emb=use_chord_emb
      ).cuda(gpuid)
  elif model_type == "gpt2":
    model = model_klass(
      dset.vocab_size, model_conf['n_layer'], model_conf['n_head'], 
      model_conf['d_model'], model_conf['d_ff'], model_conf['d_embed'],
      use_segment_emb=model_conf['use_segemb'], n_segment_types=2,
      use_chord_mhot_emb=use_chord_emb
    ).cuda(gpuid)


  if pretrained_param_path:
    pretrained_dict = torch.load(pretrained_param_path)
    pretrained_dict = {
      k:v for k, v in pretrained_dict.items() if 'feature_map.omega' not in k
    }
    model_state_dict = model.state_dict()
    model_state_dict.update(pretrained_dict)
    model.load_state_dict(model_state_dict)

  model.train()
  n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
  print ('# params:', n_params)
  print ('segemb:', model.segemb)

  opt_params = filter(lambda p: p.requires_grad, model.parameters())
  optimizer = optim.Adam(opt_params, lr=max_lr)
  scheduler = optim.lr_scheduler.CosineAnnealingLR(
    optimizer, lr_decay_steps, eta_min=min_lr
  )
  if pretrained_optimizer_path:
    optimizer.load_state_dict(
      torch.load(pretrained_optimizer_path)
    )

  params_dir = os.path.join(ckpt_dir, 'params/')
  optimizer_dir = os.path.join(ckpt_dir, 'optim/')
  if not os.path.exists(ckpt_dir):
    os.makedirs(ckpt_dir)
  if not os.path.exists(params_dir):
    os.makedirs(params_dir)
  if not os.path.exists(optimizer_dir):
    os.makedirs(optimizer_dir)

  for ep in range(max_epochs):
    loss = train_model(ep+1, model, dloader, optimizer, scheduler, model_type=model_type)
    if not (ep + 1) % ckpt_interval:
      torch.save(model.state_dict(),
        os.path.join(params_dir, 'ep{:03d}_loss{:.3f}_params.pt'.format(ep+1, loss))
      )
      torch.save(optimizer.state_dict(),
        os.path.join(optimizer_dir, 'ep{:03d}_loss{:.3f}_optim.pt'.format(ep+1, loss))
      )
    if not (ep + 1) % val_interval:
      val_losses = validate(model, val_dloader, model_type=model_type)
      with open(os.path.join(ckpt_dir, 'valloss.txt'), 'a') as f:
        f.write("ep{:03d} | loss: {:.3f} valloss: {:.3f} (+/- {:.3f})\n".format(
          ep + 1, loss, np.mean(val_losses), np.std(val_losses)
        ))
