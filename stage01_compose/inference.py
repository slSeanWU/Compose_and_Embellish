import sys
import os
import random
import pickle
sys.path.append('./model/')
sys.path.append('./')

import yaml
import torch
import numpy as np

from model.plain_transformer import PlainTransformer
from convert2midi_remi import skyline_event_to_midi, TempoEvent
from utils import pickle_load
from inference_utils import generate_plain_xl

config_path = sys.argv[1]
out_dir = sys.argv[2]
n_pieces = int(sys.argv[3]) if len(sys.argv) > 3 else 20

config = yaml.load(open(config_path, 'r'), Loader=yaml.FullLoader)
ckpt_dir = config['output']['ckpt_dir']

max_bars = 128
temp = 1.2
top_p = 0.97
max_dec_len = 2400
print ('[nucleus parameters] t = {}, p = {}'.format(temp, top_p))

torch.cuda.device(config['device'])

# for generation w/ melody prompts
use_prompt = False
prompt_bars = 8


def read_vocab(vocab_file):
  event2idx, idx2event = pickle_load(vocab_file)
  orig_vocab_size = len(event2idx)
  pad_token = orig_vocab_size
  event2idx['PAD_None'] = pad_token
  vocab_size = pad_token + 1

  return event2idx, idx2event, vocab_size


def dump_midi(words, idx2event, output_midi_path=None, 
              rfreq_cls=None, polyph_cls=None, output_event_path=None,
              return_tempo=False, enforce_tempo_val=None):
  events = [idx2event[w] for w in words]

  if output_event_path is not None:
    f = open(output_event_path, 'w')
    if rfreq_cls is not None:
      f.write('[rhymfreq] ')
      f.write(str(rfreq_cls))
      f.write('\n')
    if polyph_cls is not None:
      f.write('[polyph  ] ')
      f.write(str(polyph_cls))
      f.write('\n')
      f.write('======================================================================\n')
    print (*events, sep='\n', file=f)

  if return_tempo:
    return skyline_event_to_midi(events, output_midi_path=output_midi_path, return_tempo=True)[1]
  elif enforce_tempo_val is not None:
    skyline_event_to_midi(events, output_midi_path=output_midi_path, enforce_tempo=True, enforce_tempo_val=enforce_tempo_val)
  else:
    skyline_event_to_midi(events, output_midi_path=output_midi_path)


def get_leadsheet_prompt(data_dir, piece, prompt_n_bars):
  bar_pos, evs = pickle_load(
    os.path.join(data_dir, piece + '.pkl')
  )

  prompt_evs = [
    '{}_{}'.format(x['name'], x['value']) for x in evs[ : bar_pos[prompt_n_bars] + 1 ]
  ]
  assert len( np.where( np.array(prompt_evs) == 'Bar_None' )[0] ) == prompt_n_bars + 1
  target_bars = len(bar_pos)

  return prompt_evs, target_bars


if __name__ == '__main__':
  if not os.path.exists(out_dir):
    os.makedirs(out_dir)

  event2idx, idx2event, vocab_size = \
    read_vocab(config['data']['vocab_path'])

  if use_prompt:
    prompt_pieces = pickle_load(config['data']['val_split'])
    prompt_pieces = [x for x in prompt_pieces if os.path.exists(
      os.path.join(config['data']['data_dir'], x + '.pkl')
    )]
    if len(prompt_pieces) > n_pieces:
      prompt_pieces = random.sample(prompt_pieces, n_pieces)

    pickle.dump(
        prompt_pieces,
        open(os.path.join(out_dir, 'sampled_pieces.pkl'), 'wb')
      )
    prompts = []
    for p in prompt_pieces:
      prompts.append(
        get_leadsheet_prompt(
          config['data']['data_dir'], p,
          prompt_bars
        )
      )


  mconf = config['model']
  model = PlainTransformer(
            mconf['d_word_embed'],
            vocab_size,
            mconf['decoder']['n_layer'],
            mconf['decoder']['n_head'],
            mconf['decoder']['d_model'],
            mconf['decoder']['d_ff'],
            mconf['decoder']['tgt_len'],
            mconf['decoder']['tgt_len'],
            dec_dropout=mconf['decoder']['dropout'],
            pre_lnorm=mconf['pre_lnorm']
          ).cuda()
  print ('[info] # params:', sum(p.numel() for p in model.parameters() if p.requires_grad))

  pretrained_dict = torch.load(config['inference_param_path'], map_location='cpu')
  model.load_state_dict( pretrained_dict )
  model.eval()

  generated_pieces = 0
  total_pieces = n_pieces
  gen_times = []

  while generated_pieces < n_pieces:
    piece_id = generated_pieces + 1

    out_name = 'samp_{:02d}'.format(piece_id)
    if os.path.exists(os.path.join(out_dir, out_name + '.mid')):
      print ('[info] {} exists, skipping ...'.format(out_name))
      continue

    if not use_prompt:
      tempo_range = range(65, 165, 3)
      tempo = random.choice(
        tempo_range
      )
      orig_tempos = [
        TempoEvent(tempo, 0, 0)
      ]
      print ('[global tempo]', orig_tempos[0].tempo)
    else:
      target_bars = prompts[p][1]
      orig_tempos = [
        TempoEvent(int(prompts[p][0][0].split('_')[-1]), 0, 0)
      ]

    print (' -- generating leadsheet #{} of {}'.format(
      generated_pieces + 1, total_pieces
    ))


    if not use_prompt:
      gen_words, t_sec = generate_plain_xl(
                            model,
                            event2idx, idx2event,
                            max_events=max_dec_len, max_bars=max_bars,
                            primer=['Tempo_{}'.format(orig_tempos[0].tempo), 'Bar_None'],
                            temp=temp, top_p=top_p
                          )
    else:
      gen_words, t_sec = generate_plain_xl(
                            model,
                            event2idx, idx2event,
                            max_events=max_dec_len, max_bars=target_bars,
                            primer=prompts[p][0],
                            temp=temp, top_p=top_p,
                            prompt_bars=prompt_bars
                          )

    if gen_words is None: # model failed repeatedly
      continue
    if len(gen_words) >= max_dec_len:
      continue
    if len( np.where( np.array(gen_words) == event2idx[ 'Bar_None' ] )[0] ) >= max_bars:
      continue

    dump_midi(
      gen_words, idx2event, 
      os.path.join(out_dir, out_name + '.mid'), 
      output_event_path=os.path.join(out_dir, out_name + '.txt'),
      enforce_tempo_val=orig_tempos
    )

    gen_times.append(t_sec)
    generated_pieces += 1

  print ('[info] finished generating {} pieces, avg. time: {:.2f} +/- {:.2f} secs.'.format(
    generated_pieces, np.mean(gen_times), np.std(gen_times)
  ))
