import sys, os, time
from itertools import chain
sys.path.append('./model')

import torch
import yaml
import numpy as np

from dataloader import REMISkylineToMidiTransformerDataset, pickle_load, KEY_TO_IDX
from model.music_performer import MusicPerformer
from convert2midi import event_to_midi

train_conf_path = sys.argv[1]
gen_leadsheet_dir = sys.argv[2]
out_dir = sys.argv[3]

train_conf = yaml.load(open(train_conf_path, 'r'), Loader=yaml.FullLoader)

inference_param_path = train_conf['training']['inference_params']
train_conf_ = train_conf['training']
gpuid = train_conf_['gpuid']
torch.cuda.set_device(gpuid)

max_bars = 128
max_dec_inp_len = 2048

use_chords_mhot = 'use_chord_emb' in train_conf['model'] and train_conf['model']['use_chord_emb']

temp, top_p = 1.1, 0.99
n_pieces = 20
samp_per_piece = 1

major_map = [0, 4, 7]
minor_map = [0, 3, 7]
diminished_map = [0, 3, 6]
augmented_map = [0, 4, 8]
dominant_map = [0, 4, 7, 10]
major_seventh_map = [0, 4, 7, 11]
minor_seventh_map = [0, 3, 7, 10]
diminished_seventh_map = [0, 3, 6, 9]
half_diminished_seventh_map = [0, 3, 6, 10]
sus_2_map = [0, 2, 7]
sus_4_map = [0, 5, 7]

chord_maps = {
        'M': major_map,
        'm': minor_map,
        'o': diminished_map,
        '+': augmented_map,
        '7': dominant_map,
        'M7': major_seventh_map,
        'm7': minor_seventh_map,
        'o7': diminished_seventh_map,
        '/o7': half_diminished_seventh_map,
        'sus2': sus_2_map,
        'sus4': sus_4_map
}
chord_maps = {k : np.array(v) for k, v in chord_maps.items()}


###############################################
# sampling utilities
###############################################
def construct_inadmissible_set(tempo_val, event2idx, tolerance=20):
  inadmissibles = []

  for k, i in event2idx.items():
    if 'Tempo' in k and 'Conti' not in k and abs( int(k.split('_')[-1]) - tempo_val ) > tolerance:
      inadmissibles.append(i)

  print (inadmissibles)

  return np.array(inadmissibles)


def temperature(logits, temperature, inadmissibles=12):
  if inadmissibles is not None:
    logits[ inadmissibles ] -= np.inf

  try:
    probs = np.exp(logits / temperature) / np.sum(np.exp(logits / temperature))
    assert np.count_nonzero(np.isnan(probs)) == 0
  except:
    print ('overflow detected, use 128-bit')
    logits = logits.astype(np.float128)
    probs = np.exp(logits / temperature) / np.sum(np.exp(logits / temperature))
    probs = probs.astype(float)
  return probs

def nucleus(probs, p):
    probs /= sum(probs)
    sorted_probs = np.sort(probs)[::-1]
    sorted_index = np.argsort(probs)[::-1]
    cusum_sorted_probs = np.cumsum(sorted_probs)
    after_threshold = cusum_sorted_probs > p
    if sum(after_threshold) > 0:
        last_index = np.where(after_threshold)[0][1]
        candi_index = sorted_index[:last_index]
    else:
        candi_index = sorted_index[:3] # just assign a value
    candi_probs = np.array([probs[i] for i in candi_index], dtype=np.float64)
    candi_probs /= sum(candi_probs)
    word = np.random.choice(candi_index, size=1, p=candi_probs)[0]
    return word


##############################################
# data manipulation utilities
##############################################
def read_generated_events(events_file, event2idx, strip_token='Phrase', 
                          translate_chord=True, pitch_shift=None, assert_tempo=None, 
                          patch_chord_mhot=False):
  events = open(events_file).read().splitlines()

  if not assert_tempo:
    tempo = event2idx[ events[0] ]
  else:
    tempo = event2idx[ 'Tempo_{}'.format(assert_tempo) ]
  if strip_token is not None:
    events = [e for e in events if strip_token not in e]
  events.append('EOS_None')

  bar_pos = np.where( np.array(events) == 'Bar_None' )[0].tolist()
  bar_pos.append(len(events))

  if translate_chord:
    new_events = []
    for e in events:
      if 'Chord' in e:
        if 'N_N' in e:
          new_events.append('Chord_None_None')
        else:
          root = KEY_TO_IDX[ e.split('_')[1] ]
          new_events.append('Chord_{}_{}'.format(root, e.split('_')[-1]))
          # print (root, e.split('_')[1])
      else:
        new_events.append(e)
    assert len(new_events) == len(events)

    events = new_events

  if pitch_shift is not None:
    shifted_events = []
    for e in events:
      if 'Note_Pitch' in e:
        new_pitch = int(e.split('_')[-1]) + pitch_shift
        shifted_events.append( 'Note_Pitch_{}'.format(new_pitch))
      else:
        shifted_events.append(e)
    events = shifted_events
  
  skyline_bars = []
  for st, ed in zip(bar_pos[:-1], bar_pos[1:]):
    bar_skyline_events = [ e for e in events[st : ed] ]
    skyline_bars.append( bar_skyline_events )

  if patch_chord_mhot:
    chords_bars = []
    cur_chord = None

    for bar in range(len(skyline_bars)):
      bar_chords = np.zeros((len(skyline_bars[bar]), 12))
      for j, ev in enumerate(skyline_bars[bar]):
        if ev.split('_')[0] in ['Bar', 'Tempo']:
          continue
        elif ev.split('_')[0] == 'Chord':
          if 'None_None' in ev:
            cur_chord = None
          else:
            cur_chord = ( chord_maps[ ev.split('_')[-1] ] + int(ev.split('_')[1]) ) % 12
        
        if cur_chord is not None:
          bar_chords[j][ cur_chord ] = 1.
          bar_chords[j][ cur_chord[0] ] = 2.

      chords_bars.append( bar_chords )

  for bar in range(len(skyline_bars)):
    skyline_bars[ bar ] = [event2idx[e] for e in skyline_bars[ bar ]]

  if not patch_chord_mhot:
    return tempo, skyline_bars
  else:
    return tempo, skyline_bars, chords_bars

def word2event(word_seq, idx2event):
  return [ idx2event[w] for w in word_seq ]

def extract_skyline_from_val_data(val_input, idx2event, event2idx):
  tempo = val_input[0]

  skyline_starts = np.where( val_input == event2idx['Track_Skyline'] )[0].tolist()
  midi_starts = np.where( val_input == event2idx['Track_Midi'] )[0].tolist()

  assert len(skyline_starts) == len(midi_starts)

  skyline_bars = []
  for st, ed in zip(skyline_starts, midi_starts):
    bar_skyline_events = val_input[ st + 1 : ed ].tolist()
    skyline_bars.append( bar_skyline_events )

  return tempo, skyline_bars

def extract_midi_events_from_generation(events):
  skyline_starts = np.where( np.array(events) == 'Track_Skyline' )[0].tolist()
  midi_starts = np.where( np.array(events) == 'Track_Midi' )[0].tolist()

  midi_bars = []
  for st, ed in zip(midi_starts, skyline_starts[1:] + [len(events)]):
    bar_midi_events = events[ st + 1 : ed ]
    midi_bars.append(bar_midi_events)

  return midi_bars

def get_position_idx(event):
  return int(event.split('_')[-1])


################################################
# main generation function
################################################
def generate_conditional(model, event2idx, idx2event, skyline_events, tempo,
                          max_events=10000, skip_check=False, max_bars=None,
                          temp=1.2, top_p=0.9, inadmissibles=None,
                          use_chords_mhot=False, skyline_chords_mhot=None
                        ):
  # generated = [event2idx['Bar_None']]
  generated = [tempo, event2idx['Track_Skyline']] + skyline_events[0] + [event2idx['Track_Midi']]
  if use_chords_mhot:
    generated_chords_mhot = np.zeros((len(generated), 12))
    generated_chords_mhot[ 2 : -1 ] = skyline_chords_mhot[0]
    # print (len(generated_chords_mhot))
  # print (generated)
  seg_inp = [ 0 for _ in range(len(generated)) ]
  seg_inp[-1] = 1

  target_bars, generated_bars = len(skyline_events), 0
  if max_bars is not None:
    target_bars = min(max_bars, target_bars)

  steps = 0
  time_st = time.time()
  cur_pos = 0
  failed_cnt = 0
  cur_chord = np.zeros((1, 12))

  while generated_bars < target_bars:
    assert len(generated) == len(seg_inp)
    if len(generated) < max_dec_inp_len:
      dec_input = torch.tensor([generated]).long().to(next(model.parameters()).device)
      dec_seg_inp = torch.tensor([seg_inp]).long().to(next(model.parameters()).device)
    else:
      dec_input = torch.tensor([generated[ -max_dec_inp_len : ]]).long().to(next(model.parameters()).device)
      dec_seg_inp = torch.tensor([seg_inp[ -max_dec_inp_len : ]]).long().to(next(model.parameters()).device)

    if use_chords_mhot:
      dec_chords_mhot = torch.tensor([generated_chords_mhot[ -max_dec_inp_len : ]]).float().to(dec_input.device)
    else:
      dec_chords_mhot =None

    # sampling
    logits = model(
              dec_input, 
              seg_inp=dec_seg_inp,
              chord_inp=dec_chords_mhot,
              keep_last_only=True, 
              attn_kwargs={'omit_feature_map_draw': steps > 0}
            )
    logits = (logits[0]).cpu().detach().numpy()
    probs = temperature(logits, temp, inadmissibles=inadmissibles)
    word = nucleus(probs, top_p)
    word_event = idx2event[word]

    if not skip_check:
      if 'Beat' in word_event:
        event_pos = get_position_idx(word_event)
        if not event_pos >= cur_pos:
          failed_cnt += 1
          print ('[info] position not increasing, failed cnt:', failed_cnt)
          if failed_cnt >= 256:
            print ('[FATAL] model stuck, exiting with generated events ...')
            return generated
          continue
        else:
          cur_pos = event_pos
          failed_cnt = 0

    if word_event == 'Track_Skyline':
      steps += 1
      generated.append( word )
      seg_inp.append(0)
      generated_bars += 1
      print ('[info] generated {} bars, #events = {}'.format(generated_bars, len(generated)))

      if generated_bars < target_bars:
        generated.extend( skyline_events[ generated_bars ] )
        seg_inp.extend( [0 for _ in range(len(skyline_events[ generated_bars ]))] )

        generated.append( event2idx['Track_Midi'] )
        seg_inp.append(1)
        cur_pos = 0

        if use_chords_mhot:
          generated_chords_mhot = np.concatenate(
            (generated_chords_mhot, np.zeros((1, 12)), skyline_chords_mhot[generated_bars], np.zeros((1, 12))),
            axis=0
          )

      continue

    if word_event == 'PAD_None' or (word_event == 'EOS_None' and generated_bars < target_bars - 1):
      continue
    elif word_event == 'EOS_None' and generated_bars == target_bars - 1:
      print ('[info] gotten eos')
      generated.append(word)
      break

    generated.append(word)
    seg_inp.append(1)
    if use_chords_mhot:
      if 'Chord' in word_event and 'Conti' not in word_event:
        cur_chord = np.zeros((1, 12))
        if 'None' in word_event:
          pass
        else:
          chord_indices = ( chord_maps[ word_event.split('_')[-1] ] + int(word_event.split('_')[1]) ) % 12
          cur_chord[ 0, chord_indices ] = 1.
          cur_chord[ 0, chord_indices[0] ] = 2.
      
      if word_event not in ['Bar_None', 'Track_Midi', 'Track_Skyline']:
        generated_chords_mhot = np.concatenate(
          (generated_chords_mhot, cur_chord), axis=0
        )
      else:
        generated_chords_mhot = np.concatenate(
          (generated_chords_mhot, np.zeros((1, 12))), axis=0
        )
    steps += 1

    if len(generated) > max_events:
      print ('[info] max events reached')
      break

  print ('-- generated events:', len(generated))
  print ('-- time elapsed  : {:.2f} secs'.format(time.time() - time_st))
  print ('-- time per event: {:.2f} secs'.format((time.time() - time_st) / len(generated)))
  return generated[:-1]


if __name__ == '__main__':
  val_split = train_conf['data_loader']['val_split']
  dset = REMISkylineToMidiTransformerDataset(
    train_conf['data_loader']['data_path'],
    train_conf['data_loader']['vocab_path'], 
    do_augment=False, 
    model_dec_seqlen=train_conf['model']['max_len'], 
    pieces=pickle_load(val_split),
    pad_to_same=True,
    use_chord_mhot=use_chords_mhot
  )

  model_conf = train_conf['model']
  model = model = MusicPerformer(
      dset.vocab_size, model_conf['n_layer'], model_conf['n_head'], 
      model_conf['d_model'], model_conf['d_ff'], model_conf['d_embed'],
      use_segment_emb=model_conf['use_segemb'], n_segment_types=2,
      favor_feature_dims=model_conf['feature_map']['n_dims'],
      use_chord_mhot_emb=use_chords_mhot
    ).cuda()

  pretrained_dict = torch.load(inference_param_path) #, map_location='cpu')
  pretrained_dict = {
    k:v for k, v in pretrained_dict.items() if 'feature_map.omega' not in k
  }
  model_state_dict = model.state_dict()
  model_state_dict.update(pretrained_dict)
  model.load_state_dict(model_state_dict)

  model.eval()
  print ('[info] model loaded')

  if not os.path.exists(out_dir):
    os.makedirs(out_dir)

  skyline_files = [ x for x in os.listdir(gen_leadsheet_dir) if '.txt' in x ]
  print ('[# pieces]', len(skyline_files))

  for samp in range(samp_per_piece):
    for p in range(min(n_pieces, len(skyline_files))):
      out_name = skyline_files[p].split('.')[0] + '_2stage_samp{:02d}'.format(samp + 1)
      if os.path.exists(os.path.join(out_dir, out_name + '.mid')):
        print ('[info] {} exists, skipping ...'.format(out_name) )
        continue

      # print (out_name)
      if not use_chords_mhot:
        tempo, skyline_events = read_generated_events(os.path.join(gen_leadsheet_dir, skyline_files[p]), 
                                                    dset.event2idx)
        skyline_chords_mhot = None
      else:
        tempo, skyline_events, skyline_chords_mhot = read_generated_events(
                                                      os.path.join(gen_leadsheet_dir, skyline_files[p]), 
                                                      dset.event2idx, patch_chord_mhot=use_chords_mhot
                                                    )
    
      with torch.no_grad():
        generated = generate_conditional(model, dset.event2idx, dset.idx2event, skyline_events, tempo, max_bars=max_bars,
                                         temp=temp, top_p=top_p, inadmissibles=None, 
                                         use_chords_mhot=use_chords_mhot,
                                         skyline_chords_mhot=skyline_chords_mhot
                                        )

      for i in range(min(max_bars, len(skyline_events))):
        skyline_events[i] = word2event(skyline_events[i], dset.idx2event)
      
      generated = word2event(generated, dset.idx2event)
      generated = extract_midi_events_from_generation(generated)

      event_to_midi(
        [ dset.idx2event[tempo] ] + \
        list(chain(*skyline_events[:max_bars])),
        mode='skyline',
        output_midi_path=os.path.join(
          out_dir, '{}_skyline.mid'.format(skyline_files[p].split('.')[0])
        )
      )
      event_to_midi(
        list(chain(*generated[:max_bars])),
        mode='full',
        output_midi_path=os.path.join(out_dir, out_name + '.mid')
      )
