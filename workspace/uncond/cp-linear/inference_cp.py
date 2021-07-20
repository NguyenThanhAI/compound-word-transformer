import sys
import os

import argparse

import math
import time
import glob
import datetime
import random
import pickle
import json
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

import torch.optim as optim
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import Dataset, DataLoader

from fast_transformers.builders import TransformerEncoderBuilder
from fast_transformers.builders import RecurrentEncoderBuilder
from fast_transformers.masking import TriangularCausalMask

import miditoolkit
from miditoolkit.midi.containers import Marker, Instrument, TempoChange, Note

import saver


def get_args():

    parser = argparse.ArgumentParser()

    parser.add_argument("--model_dir", type=str, default=None)
    parser.add_argument("--path_dictionary", type=str, default=None)
    parser.add_argument("--d_model", type=int, default=512)
    parser.add_argument("--n_layer", type=int, default=12)
    parser.add_argument("--n_head", type=int, default=8)
    parser.add_argument("--num_songs", type=int, default=1)
    parser.add_argument("--beat_resol", type=int, default=480)

    args = parser.parse_args()

    return args


def write_midi(words, path_outfile, word2event, bar_resol, tick_resol):
    class_keys = word2event.keys()
    # words = np.load(path_infile)
    midi_obj = miditoolkit.midi.parser.MidiFile()

    bar_cnt = 0
    cur_pos = 0

    all_notes = []

    cnt_error = 0
    for i in range(len(words)):
        vals = []
        for kidx, key in enumerate(class_keys):
            vals.append(word2event[key][words[i][kidx]])
        # print(vals)

        if vals[3] == 'Metrical':
            if vals[2] == 'Bar':
                bar_cnt += 1
            elif 'Beat' in vals[2]:
                beat_pos = int(vals[2].split('_')[1])
                cur_pos = bar_cnt * bar_resol + beat_pos * tick_resol

                # chord
                if vals[1] != 'CONTI' and vals[1] != 0:
                    midi_obj.markers.append(
                        Marker(text=str(vals[1]), time=cur_pos))

                if vals[0] != 'CONTI' and vals[0] != 0:
                    tempo = int(vals[0].split('_')[-1])
                    midi_obj.tempo_changes.append(
                        TempoChange(tempo=tempo, time=cur_pos))
            else:
                pass
        elif vals[3] == 'Note':

            try:
                pitch = vals[4].split('_')[-1]
                duration = vals[5].split('_')[-1]
                velocity = vals[6].split('_')[-1]

                if int(duration) == 0:
                    duration = 60
                end = cur_pos + int(duration)

                all_notes.append(
                    Note(
                        pitch=int(pitch),
                        start=cur_pos,
                        end=end,
                        velocity=int(velocity))
                )
            except:
                continue
        else:
            pass

    # save midi
    piano_track = Instrument(0, is_drum=False, name='piano')
    piano_track.notes = all_notes
    midi_obj.instruments = [piano_track]
    midi_obj.dump(path_outfile)


################################################################################
# Sampling
################################################################################
# -- temperature -- #
def softmax_with_temperature(logits, temperature):
    probs = np.exp(logits / temperature) / np.sum(np.exp(logits / temperature))
    return probs


def weighted_sampling(probs):
    probs /= sum(probs)
    sorted_probs = np.sort(probs)[::-1]
    sorted_index = np.argsort(probs)[::-1]
    word = np.random.choice(sorted_index, size=1, p=sorted_probs)[0]
    return word


# -- nucleus -- #
def nucleus(probs, p):
    probs /= (sum(probs) + 1e-5)
    sorted_probs = np.sort(probs)[::-1]
    sorted_index = np.argsort(probs)[::-1]
    cusum_sorted_probs = np.cumsum(sorted_probs)
    after_threshold = cusum_sorted_probs > p
    if sum(after_threshold) > 0:
        last_index = np.where(after_threshold)[0][0] + 1
        candi_index = sorted_index[:last_index]
    else:
        candi_index = sorted_index[:]
    candi_probs = [probs[i] for i in candi_index]
    candi_probs /= sum(candi_probs)
    word = np.random.choice(candi_index, size=1, p=candi_probs)[0]
    return word


def sampling(logit, p=None, t=1.0):
    logit = logit.squeeze().cpu().numpy()
    probs = softmax_with_temperature(logits=logit, temperature=t)

    if p is not None:
        cur_word = nucleus(probs, p=p)
    else:
        cur_word = weighted_sampling(probs)
    return cur_word


################################################################################
# Model
################################################################################


def network_paras(model):
    # compute only trainable params
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    return params


class Embeddings(nn.Module):
    def __init__(self, n_token, d_model):
        super(Embeddings, self).__init__()
        self.lut = nn.Embedding(n_token, d_model)
        self.d_model = d_model

    def forward(self, x):
        return self.lut(x) * math.sqrt(self.d_model)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=20000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class TransformerModel(nn.Module):
    def __init__(self, n_token, d_model, n_layer, n_head, is_training=True):
        super(TransformerModel, self).__init__()

        # --- params config --- #
        self.n_token = n_token
        self.d_model = d_model
        self.n_layer = n_layer  #
        self.dropout = 0.1
        self.n_head = n_head  #
        self.d_head = self.d_model // self.n_head
        self.d_inner = 2048
        self.loss_func = nn.CrossEntropyLoss(reduction='none')
        self.emb_sizes = [128, 256, 64, 32, 512, 128, 128]

        # --- modules config --- #
        # embeddings
        print('>>>>>:', self.n_token)
        self.word_emb_tempo = Embeddings(self.n_token[0], self.emb_sizes[0])
        self.word_emb_chord = Embeddings(self.n_token[1], self.emb_sizes[1])
        self.word_emb_barbeat = Embeddings(self.n_token[2], self.emb_sizes[2])
        self.word_emb_type = Embeddings(self.n_token[3], self.emb_sizes[3])
        self.word_emb_pitch = Embeddings(self.n_token[4], self.emb_sizes[4])
        self.word_emb_duration = Embeddings(self.n_token[5], self.emb_sizes[5])
        self.word_emb_velocity = Embeddings(self.n_token[6], self.emb_sizes[6])
        self.pos_emb = PositionalEncoding(self.d_model, self.dropout)

        # linear
        self.in_linear = nn.Linear(np.sum(self.emb_sizes), self.d_model)

        # encoder
        if is_training:
            # encoder (training)
            self.transformer_encoder = TransformerEncoderBuilder.from_kwargs(
                n_layers=self.n_layer,
                n_heads=self.n_head,
                query_dimensions=self.d_model // self.n_head,
                value_dimensions=self.d_model // self.n_head,
                feed_forward_dimensions=2048,
                activation='gelu',
                dropout=0.1,
                attention_type="causal-linear",
            ).get()
        else:
            # encoder (inference)
            print(' [o] using RNN backend.')
            self.transformer_encoder = RecurrentEncoderBuilder.from_kwargs(
                n_layers=self.n_layer,
                n_heads=self.n_head,
                query_dimensions=self.d_model // self.n_head,
                value_dimensions=self.d_model // self.n_head,
                feed_forward_dimensions=2048,
                activation='gelu',
                dropout=0.1,
                attention_type="causal-linear",
            ).get()

        # blend with type
        self.project_concat_type = nn.Linear(self.d_model + 32, self.d_model)

        # individual output
        self.proj_tempo = nn.Linear(self.d_model, self.n_token[0])
        self.proj_chord = nn.Linear(self.d_model, self.n_token[1])
        self.proj_barbeat = nn.Linear(self.d_model, self.n_token[2])
        self.proj_type = nn.Linear(self.d_model, self.n_token[3])
        self.proj_pitch = nn.Linear(self.d_model, self.n_token[4])
        self.proj_duration = nn.Linear(self.d_model, self.n_token[5])
        self.proj_velocity = nn.Linear(self.d_model, self.n_token[6])

    def compute_loss(self, predict, target, loss_mask):
        loss = self.loss_func(predict, target)
        loss = loss * loss_mask
        loss = torch.sum(loss) / torch.sum(loss_mask)
        return loss

    def train_step(self, x, target, loss_mask):
        h, y_type = self.forward_hidden(x)
        y_tempo, y_chord, y_barbeat, y_pitch, y_duration, y_velocity = self.forward_output(h, target)

        # reshape (b, s, f) -> (b, f, s)
        y_tempo = y_tempo[:, ...].permute(0, 2, 1)
        y_chord = y_chord[:, ...].permute(0, 2, 1)
        y_barbeat = y_barbeat[:, ...].permute(0, 2, 1)
        y_type = y_type[:, ...].permute(0, 2, 1)
        y_pitch = y_pitch[:, ...].permute(0, 2, 1)
        y_duration = y_duration[:, ...].permute(0, 2, 1)
        y_velocity = y_velocity[:, ...].permute(0, 2, 1)

        # loss
        loss_tempo = self.compute_loss(
            y_tempo, target[..., 0], loss_mask)
        loss_chord = self.compute_loss(
            y_chord, target[..., 1], loss_mask)
        loss_barbeat = self.compute_loss(
            y_barbeat, target[..., 2], loss_mask)
        loss_type = self.compute_loss(
            y_type, target[..., 3], loss_mask)
        loss_pitch = self.compute_loss(
            y_pitch, target[..., 4], loss_mask)
        loss_duration = self.compute_loss(
            y_duration, target[..., 5], loss_mask)
        loss_velocity = self.compute_loss(
            y_velocity, target[..., 6], loss_mask)

        return loss_tempo, loss_chord, loss_barbeat, loss_type, loss_pitch, loss_duration, loss_velocity

    def forward_hidden(self, x, memory=None, is_training=True):
        '''
        linear transformer: b x s x f
        x.shape=(bs, nf)
        '''

        # embeddings
        emb_tempo = self.word_emb_tempo(x[..., 0])
        emb_chord = self.word_emb_chord(x[..., 1])
        emb_barbeat = self.word_emb_barbeat(x[..., 2])
        emb_type = self.word_emb_type(x[..., 3])
        emb_pitch = self.word_emb_pitch(x[..., 4])
        emb_duration = self.word_emb_duration(x[..., 5])
        emb_velocity = self.word_emb_velocity(x[..., 6])

        embs = torch.cat(
            [
                emb_tempo,
                emb_chord,
                emb_barbeat,
                emb_type,
                emb_pitch,
                emb_duration,
                emb_velocity,
            ], dim=-1)

        emb_linear = self.in_linear(embs)
        pos_emb = self.pos_emb(emb_linear)

        # assert False

        # transformer
        if is_training:
            # mask
            attn_mask = TriangularCausalMask(pos_emb.size(1), device=x.device)
            h = self.transformer_encoder(pos_emb, attn_mask)  # y: b x s x d_model

            # project type
            y_type = self.proj_type(h)
            return h, y_type
        else:
            pos_emb = pos_emb.squeeze(0)
            h, memory = self.transformer_encoder(pos_emb, memory=memory)  # y: s x d_model

            # project type
            y_type = self.proj_type(h)
            return h, y_type, memory

    def forward_output(self, h, y):
        '''
        for training
        '''
        tf_skip_type = self.word_emb_type(y[..., 3])

        # project other
        y_concat_type = torch.cat([h, tf_skip_type], dim=-1)
        y_ = self.project_concat_type(y_concat_type)

        y_tempo = self.proj_tempo(y_)
        y_chord = self.proj_chord(y_)
        y_barbeat = self.proj_barbeat(y_)
        y_pitch = self.proj_pitch(y_)
        y_duration = self.proj_duration(y_)
        y_velocity = self.proj_velocity(y_)

        return y_tempo, y_chord, y_barbeat, y_pitch, y_duration, y_velocity

    def froward_output_sampling(self, h, y_type):
        '''
        for inference
        '''
        # sample type
        y_type_logit = y_type[0, :]
        cur_word_type = sampling(y_type_logit, p=0.90)

        type_word_t = torch.from_numpy(
            np.array([cur_word_type])).long().cuda().unsqueeze(0)

        tf_skip_type = self.word_emb_type(type_word_t).squeeze(0)

        # concat
        y_concat_type = torch.cat([h, tf_skip_type], dim=-1)
        y_ = self.project_concat_type(y_concat_type)

        # project other
        y_tempo = self.proj_tempo(y_)
        y_chord = self.proj_chord(y_)
        y_barbeat = self.proj_barbeat(y_)

        y_pitch = self.proj_pitch(y_)
        y_duration = self.proj_duration(y_)
        y_velocity = self.proj_velocity(y_)

        # sampling gen_cond
        cur_word_tempo = sampling(y_tempo, t=1.2, p=0.9)
        cur_word_barbeat = sampling(y_barbeat, t=1.2)
        cur_word_chord = sampling(y_chord, p=0.99)
        cur_word_pitch = sampling(y_pitch, p=0.9)
        cur_word_duration = sampling(y_duration, t=2, p=0.9)
        cur_word_velocity = sampling(y_velocity, t=5)

        # collect
        next_arr = np.array([
            cur_word_tempo,
            cur_word_chord,
            cur_word_barbeat,
            cur_word_type,
            cur_word_pitch,
            cur_word_duration,
            cur_word_velocity,
        ])
        return next_arr

    def inference_from_scratch(self, dictionary):
        event2word, word2event = dictionary
        classes = word2event.keys()

        def print_word_cp(cp):
            result = [word2event[k][cp[idx]] for idx, k in enumerate(classes)]

            for r in result:
                print('{:15s}'.format(str(r)), end=' | ')
            print('')

        init = np.array([
            [0, 0, 1, 1, 0, 0, 0],  # bar
        ])

        cnt_token = len(init)
        with torch.no_grad():
            final_res = []
            memory = None
            h = None

            cnt_bar = 1
            init_t = torch.from_numpy(init).long().cuda()
            print('------ initiate ------')
            for step in range(init.shape[0]):
                print_word_cp(init[step, :])
                input_ = init_t[step, :].unsqueeze(0).unsqueeze(0)
                final_res.append(init[step, :][None, ...])

                h, y_type, memory = self.forward_hidden(
                    input_, memory, is_training=False)

            print('------ generate ------')
            while (True):
                # sample others
                next_arr = self.froward_output_sampling(h, y_type)
                final_res.append(next_arr[None, ...])
                print('bar:', cnt_bar, end='  ==')
                print_word_cp(next_arr)

                # forward
                input_ = torch.from_numpy(next_arr).long().cuda()
                input_ = input_.unsqueeze(0).unsqueeze(0)
                h, y_type, memory = self.forward_hidden(
                    input_, memory, is_training=False)

                # end of sequence
                if word2event['type'][next_arr[3]] == 'EOS':
                    break

                if word2event['bar-beat'][next_arr[2]] == 'Bar':
                    cnt_bar += 1

        print('\n--------[Done]--------')
        final_res = np.concatenate(final_res)
        print(final_res.shape)
        return final_res


def generate(model_dir, path_dictionary, path_gendir, d_model, n_layer, n_head, num_songs, bar_resol, tick_resol):
    # path
    #path_ckpt = info_load_model[0]  # path to ckpt dir
    #loss = info_load_model[1]  # loss
    #name = 'loss_' + str(loss)
    #path_saved_ckpt = os.path.join(path_ckpt, name + '_params.pt')
    for dirs, _, files in os.walk(model_dir):
        for file in files:
            name, ext = os.path.splitext(file)

            if name.endswith("_params") and ext == "pt":
                path_saved_ckpt = os.path.join(dirs, file)
                break
        break

    # load
    dictionary = pickle.load(open(path_dictionary, 'rb'))
    event2word, word2event = dictionary

    # outdir
    if not os.path.exists(path_gendir):
        os.makedirs(path_gendir, exist_ok=True)

    # config
    n_class = []
    for key in event2word.keys():
        n_class.append(len(dictionary[0][key]))

    # init model
    net = TransformerModel(n_class, d_model=d_model, n_layer=n_layer, n_head=n_head, is_training=False)
    net.to(torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
    net.eval()

    # load model
    print('[*] load model from:', path_saved_ckpt)
    net.load_state_dict(torch.load(path_saved_ckpt))

    # gen
    start_time = time.time()
    song_time_list = []
    words_len_list = []

    cnt_tokens_all = 0
    sidx = 0
    while sidx < num_songs:
        try:
            start_time = time.time()
            print('current idx:', sidx)
            path_outfile = os.path.join(path_gendir, 'get_{}.mid'.format(str(sidx)))

            res = net.inference_from_scratch(dictionary)
            write_midi(res, path_outfile, word2event, bar_resol=bar_resol, tick_resol=tick_resol)

            song_time = time.time() - start_time
            word_len = len(res)
            print('song time:', song_time)
            print('word_len:', word_len)
            words_len_list.append(word_len)
            song_time_list.append(song_time)

            sidx += 1
        except KeyboardInterrupt:
            raise ValueError(' [x] terminated.')
        except:
            continue

    print('ave token time:', sum(words_len_list) / sum(song_time_list))
    print('ave song time:', np.mean(song_time_list))

    runtime_result = {
        'song_time': song_time_list,
        'words_len_list': words_len_list,
        'ave token time:': sum(words_len_list) / sum(song_time_list),
        'ave song time': float(np.mean(song_time_list)),
    }

    with open('runtime_stats.json', 'w') as f:
        json.dump(runtime_result, f)


if __name__ == '__main__':
    args = get_args()

    model_dir = args.model_dir
    path_dictionary = args.path_dictionary
    path_gendir = args.path_gendir
    d_model = args.d_model
    n_layer = args.n_layer
    n_head = args.n_head
    num_songs = args.num_songs
    beat_resol = args.beat_resol

    bar_resol = beat_resol * 4
    tick_resol = beat_resol // 4

    generate(model_dir=model_dir, path_dictionary=path_dictionary, path_gendir=path_gendir, d_model=d_model,
             n_layer=n_layer, n_head=n_head, num_songs=num_songs, bar_resol=bar_resol, tick_resol=tick_resol)
