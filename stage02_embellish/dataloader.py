   
import os, random
import pickle
from glob import glob

import torch
import numpy as np
from torch.utils.data import Dataset

IDX_TO_KEY = {
  9: 'A',
  10: 'A#',
  11: 'B',
  0: 'C',
  1: 'C#',
  2: 'D',
  3: 'D#',
  4: 'E',
  5: 'F',
  6: 'F#',
  7: 'G',
  8: 'G#'
}
KEY_TO_IDX = {
  v:k for k, v in IDX_TO_KEY.items()
}

def get_chord_tone(chord_event):
  tone = chord_event['value'].split('_')[0]
  return int(tone)

def transpose_chord(chord_event, n_keys):
  if chord_event['value'] == 'None_None' or chord_event['value'] == 'Conti_Conti':
    return chord_event

  orig_tone_idx = get_chord_tone(chord_event)
  new_tone_idx = (orig_tone_idx + 12 + n_keys) % 12
  new_chord_value = chord_event['value'].replace(
    '{}_'.format(orig_tone_idx), '{}_'.format(new_tone_idx)
  )
  new_chord_event = {'name': chord_event['name'], 'value': new_chord_value}
  
  # print ('[n key = {:4}] {} --> {}'.format(n_keys, chord_event['value'], new_chord_event['value']))

  return new_chord_event

def check_extreme_pitch(raw_events):
  low, high = 128, 0
  for ev in raw_events:
    if ev['name'] == 'Note_Pitch':
      low = min(low, int(ev['value']))
      high = max(high, int(ev['value']))

  return low, high

def transpose_events(raw_events, n_keys):
  transposed_raw_events = []

  for ev in raw_events:
    if ev['name'] == 'Note_Pitch':
      transposed_raw_events.append(
        {'name': ev['name'], 'value': ev['value'] + n_keys}
      )
    elif ev['name'] == 'Chord':
      transposed_raw_events.append(
        transpose_chord(ev, n_keys)
      )
    else:
      transposed_raw_events.append(ev)

  assert len(transposed_raw_events) == len(raw_events)
  return transposed_raw_events

def pickle_load(path):
  return pickle.load(open(path, 'rb'))

def convert_event(event_seq, event2idx, to_ndarr=True):
  if isinstance(event_seq[0], dict):
    event_seq = [event2idx['{}_{}'.format(e['name'], e['value'])] for e in event_seq]
  else:
    event_seq = [event2idx[e] for e in event_seq]

  if to_ndarr:
    return np.array(event_seq)
  else:
    return event_seq

class REMISkylineToMidiTransformerDataset(Dataset):
  def __init__(self, data_dir, vocab_file, 
               model_dec_seqlen=10240, model_max_bars=None,
               pieces=[], do_augment=True, augment_range=range(-6, 7), 
               min_pitch=21, max_pitch=108, pad_to_same=True,
               appoint_st_bar=None, dec_end_pad_value=None,
               use_chord_mhot=False
              ):
    self.vocab_file = vocab_file
    self.read_vocab()

    self.model_dec_seqlen = model_dec_seqlen
    self.model_max_bars = model_max_bars
    self.data_dir = data_dir
    self.pieces = pieces
    self.use_chord_mhot = use_chord_mhot
    self.build_dataset()

    self.do_augment = do_augment
    self.augment_range = augment_range
    self.min_pitch, self.max_pitch = min_pitch, max_pitch
    self.pad_to_same = pad_to_same

    self.appoint_st_bar = appoint_st_bar
    if dec_end_pad_value == 'EOS':
      self.dec_end_pad_value = self.eos_token
    else:
      self.dec_end_pad_value = self.pad_token

  def read_vocab(self):
    vocab = pickle_load(self.vocab_file)[0]
    self.idx2event = pickle_load(self.vocab_file)[1]
    orig_vocab_size = len(vocab)
    self.event2idx = vocab
    self.bar_token = self.event2idx['Bar_None']
    self.eos_token = self.event2idx['EOS_None']
    self.pad_token = orig_vocab_size
    self.vocab_size = self.pad_token + 1
  
  def build_dataset(self):
    if not self.pieces:
      self.pieces = sorted( glob(os.path.join(self.data_dir, '*.pkl')) )
    else:
      self.pieces = sorted( [os.path.join(self.data_dir, p) for p in self.pieces] )

    self.piece_skyline_pos = []
    self.piece_midi_pos = []
    self.piece_admissible_stbars = []

    for i, p in enumerate(self.pieces):
      piece_data = pickle_load(p)
      skyline_pos, midi_pos = piece_data[0], piece_data[1]
      piece_evs = piece_data[2]
      if not i % 200:
        print ('[preparing data] now at #{}'.format(i))

      self.piece_skyline_pos.append(skyline_pos)
      self.piece_midi_pos.append(midi_pos)

      if len(piece_evs) <= self.model_dec_seqlen:
        self.piece_admissible_stbars.append([0])
      else:
        _admissible_stbars = []
        for bar in range(len(self.piece_skyline_pos[-1])):
          if len(piece_evs) - self.piece_skyline_pos[-1][bar][0] >= 0.5 * self.model_dec_seqlen:
            _admissible_stbars.append(bar)
          else:
            break
        self.piece_admissible_stbars.append(_admissible_stbars)

  def get_sample_from_file(self, piece_idx):
    if self.use_chord_mhot:
      piece_evs, piece_chords = pickle_load(self.pieces[piece_idx])[-2:]
    else:
      piece_evs = pickle_load(self.pieces[piece_idx])[2]

    piece_skyline_pos = self.piece_skyline_pos[piece_idx]
    piece_midi_pos = self.piece_midi_pos[piece_idx]
    # n_bars = len(piece_midi_pos)
    st_bar = random.choice(self.piece_admissible_stbars[ piece_idx ])

    if not self.use_chord_mhot:
      return piece_evs, piece_skyline_pos, piece_midi_pos, st_bar
    else:
      return piece_evs, piece_skyline_pos, piece_midi_pos, piece_chords, st_bar

  def pad_sequence(self, seq, maxlen, pad_value=None):
    if pad_value is None:
      pad_value = self.pad_token

    if len(seq) < maxlen:
      seq.extend( [pad_value for _ in range(maxlen- len(seq))] )

    return seq

  def pad_chords(self, chords, maxlen):
    if chords is None:
      return None

    if len(chords) >= maxlen:
      return chords

    chords = np.concatenate(
      (chords, np.zeros((maxlen - len(chords), 12))),
      axis=0
    )

    return chords

  def pitch_augment(self, bar_events):
    bar_min_pitch, bar_max_pitch = check_extreme_pitch(bar_events)
    
    n_keys = random.choice(self.augment_range)
    while bar_min_pitch + n_keys < self.min_pitch or bar_max_pitch + n_keys > self.max_pitch:
      n_keys = random.choice(self.augment_range)

    augmented_bar_events = transpose_events(bar_events, n_keys)
    return augmented_bar_events

  def make_target_and_mask(self, inp_tokens, skyline_pos, midi_pos, st_bar):
    tgt = np.full_like(inp_tokens, fill_value=self.pad_token)
    track_mask = np.zeros_like(inp_tokens)

    for bidx in range(st_bar, len(skyline_pos)):
      offset =  - skyline_pos[st_bar][0] + 1 

      track_mask[ midi_pos[bidx][0] + offset : midi_pos[bidx][1] + offset ] = 1
      if bidx != len(skyline_pos) - 1:
        tgt[ midi_pos[bidx][0] + offset : midi_pos[bidx][1] + offset ] = inp_tokens[ midi_pos[bidx][0] + 1 + offset: midi_pos[bidx][1] + 1 + offset ]
      else:
        tgt[ midi_pos[bidx][0] + offset : midi_pos[bidx][1] - 1 + offset ] = inp_tokens[ midi_pos[bidx][0] + 1 + offset : midi_pos[bidx][1] + offset ]
        tgt[ midi_pos[bidx][1] - 1 + offset ] = self.eos_token

    return tgt, track_mask

  def __len__(self):
    return len(self.pieces)

  def __getitem__(self, idx):
    if torch.is_tensor(idx):
      idx = idx.tolist()

    if not self.use_chord_mhot:
      piece_events, skyline_pos, midi_pos, st_bar = self.get_sample_from_file(idx)
      piece_chords = None
    else:
      piece_events, skyline_pos, midi_pos, piece_chords, st_bar = self.get_sample_from_file(idx)

    if self.do_augment:
      piece_events = self.pitch_augment(piece_events)

    # print ('poly and rfreq', polyph_cls, rfreq_cls)
    piece_tokens = convert_event(
      [piece_events[0]] + piece_events[ self.piece_skyline_pos[idx][st_bar][0] : ], 
      self.event2idx, to_ndarr=False
    )
    length = len(piece_tokens)

    if self.pad_to_same:
      inp = self.pad_sequence(piece_tokens, self.model_dec_seqlen)
      piece_chords = self.pad_chords(piece_chords, self.model_dec_seqlen)

    inp = np.array(inp, dtype=int)

    target, track_mask = self.make_target_and_mask(inp, skyline_pos, midi_pos, st_bar)
    # piece_tokens.append( self.eos_token )

    inp = inp[:self.model_dec_seqlen]
    target = target[:self.model_dec_seqlen]
    track_mask = track_mask[:self.model_dec_seqlen]

    if piece_chords is not None:
      piece_chords = piece_chords[ : self.model_dec_seqlen]
    else:
      piece_chords = 0

    # print (bar_pos)
    # print ('[no. {:06d}] len: {:4} | last: {}'.format(idx, len(bar_tokens), self.idx2event[ bar_tokens[-1] ]))

    # target = np.array(inp[1:], dtype=int)
    # inp = np.array(inp[:-1], dtype=int)
    # assert len(inp) == len(target)

    return {
      'id': idx,
      'piece_id': self.pieces[idx].split('/')[-1].replace('.pkl', ''),
      'dec_input': inp,
      'dec_target': target,
      'chords_mhot': piece_chords,
      'track_mask': track_mask,
      'length': min(length, self.model_dec_seqlen)
    }
