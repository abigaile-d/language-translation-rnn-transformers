import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn.utils.rnn import pad_sequence
from torchtext.data.metrics import bleu_score


from data import MAX_LENGTH, SOS_token, EOS_token


PADDING_VALUE=0


class TransformersTools():
    def __init__(self, device):
        self.device = device
    
    def collate(list_of_samples):
        src_seqs = [x[0] for x in list_of_samples]
        src_seqs = pad_sequence(src_seqs)
        src_mask = torch.eq(src_seqs, PADDING_VALUE)
        
        tgt_seqs = [x[1] for x in list_of_samples]
        tgt_seqs = pad_sequence(tgt_seqs)
        tgt_seqs = torch.cat((torch.tensor([SOS_token] * src_seqs.shape[1])[None, :], tgt_seqs), dim=0)

        return src_seqs, src_mask, tgt_seqs

    def translate(self, model, src_seqs, src_mask):
        with torch.no_grad():
            src_seqs, src_mask = src_seqs.to(self.device), src_mask.to(self.device)

            tgt_seq_sos = torch.tensor([SOS_token] * src_seqs.shape[1])
            tgt_seq_sos = tgt_seq_sos[None, :]
            tgt_seq_part = tgt_seq_sos
            z = model.encoder(src_seqs, mask=src_mask)
            for i in range(0, MAX_LENGTH):
                outputs = model.decoder(tgt_seq_part, z, src_mask=src_mask)
                _, out_seqs = torch.max(outputs, dim=2)
                tgt_seq_part = torch.cat((tgt_seq_part, out_seqs[-1:]), dim=0)
            return tgt_seq_part[1:]
    
    def seq_to_tokens(self, seq, lang):
        'Convert a sequence of word indices into a list of words (strings).'
        sentence = []
        for i in seq:
            if i == EOS_token:
                break
            sentence.append(lang.index2word[i.item()])
        return(sentence)

    def seq_to_string(self, seq, lang):
        'Convert a sequence of word indices into a sentence string.'
        return(' '.join(self.seq_to_tokens(seq, lang)))
    
    def compute_bleu_score(self, model, dataloader, output_lang, max_batch=5):
        candidate_corpus = []
        references_corpus = []
        for i, (src_seqs, src_mask, tgt_seqs) in enumerate(dataloader):
            out_seqs = self.translate(model, src_seqs, src_mask)
            candidate_corpus.extend([self.seq_to_tokens(seq, output_lang) for seq in out_seqs.T])
            references_corpus.extend([[self.seq_to_tokens(seq, output_lang)] for seq in tgt_seqs[1:].T])
            if i == max_batch:
                break

        score = bleu_score(candidate_corpus, references_corpus)
        return score


class EncoderBlock(nn.Module):
    def __init__(self, n_features, n_heads, n_hidden=64, dropout=0.1):
        super(EncoderBlock, self).__init__()

        self.self_attention = nn.MultiheadAttention(n_features, n_heads)
        self.dropout1 = nn.Dropout(dropout)
        self.layer_norm1 = nn.LayerNorm(n_features)
        
        self.mlp = nn.Sequential(
            nn.Linear(n_features, n_hidden),
            nn.Dropout(dropout),
            nn.ReLU(),
            nn.Linear(n_hidden, n_features)
        )
        self.dropout2 = nn.Dropout(dropout)
        self.layer_norm2 = nn.LayerNorm(n_features)

    def forward(self, x, mask):
        att, att_w = self.self_attention(x, x, x, key_padding_mask=mask.T)
        z = self.dropout1(att)
        z = self.layer_norm2(x + z)
        
        x = z
        z = self.mlp(z)
        z = self.dropout2(z)
        z = self.layer_norm2(x + z)
        
        return z


class PositionalEncoding(nn.Module):
    """This implementation is the same as in the Annotated transformer blog post
        See https://nlp.seas.harvard.edu/2018/04/03/attention.html for more detail.
    """
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        assert (d_model % 2) == 0, 'd_model should be an even number.'
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class Encoder(nn.Module):
    def __init__(self, src_vocab_size, n_blocks, n_features, n_heads, n_hidden=64, dropout=0.1):
        super(Encoder, self).__init__()

        self.embedding = nn.Embedding(src_vocab_size, n_features, padding_idx=PADDING_VALUE)
        self.positional_encoding = PositionalEncoding(n_features, max_len=MAX_LENGTH)
        self.encoder_blocks = nn.ModuleList([EncoderBlock(n_features=n_features, n_heads=n_heads, n_hidden=n_hidden, dropout=dropout) for _ in range(n_blocks)])

    def forward(self, x, mask):
        z = self.embedding(x)
        z = self.positional_encoding(z)
    
        for block in self.encoder_blocks:
            z = block(z, mask=mask)
        return z


class DecoderBlock(nn.Module):
    def __init__(self, n_features, n_heads, n_hidden=64, dropout=0.1):
        super(DecoderBlock, self).__init__()

        self.self_attention = nn.MultiheadAttention(n_features, n_heads)
        self.dropout1 = nn.Dropout(dropout)
        self.layer_norm1 = nn.LayerNorm(n_features)
        
        self.enc_attention = nn.MultiheadAttention(n_features, n_heads)
        self.dropout2 = nn.Dropout(dropout)
        self.layer_norm2 = nn.LayerNorm(n_features)
        
        self.mlp = nn.Sequential(
            nn.Linear(n_features, n_hidden),
            nn.Dropout(dropout),
            nn.ReLU(),
            nn.Linear(n_hidden, n_features)
        )
        self.dropout3 = nn.Dropout(dropout)
        self.layer_norm3 = nn.LayerNorm(n_features)
        

    def forward(self, y, z, src_mask, tgt_mask):
        att, att_w = self.self_attention(y, y, y, attn_mask=tgt_mask)
        out = self.dropout1(att)
        out = self.layer_norm1(y + out)
        
        y = out
        att, att_w = self.enc_attention(out, z, z, key_padding_mask=src_mask.T)
        out = self.dropout2(att)
        out = self.layer_norm2(y + out)
        
        y = out
        out = self.mlp(out)
        out = self.dropout3(out)
        out = self.layer_norm3(y + out)
        
        return out


class Decoder(nn.Module):
    def __init__(self, tgt_vocab_size, n_blocks, n_features, n_heads, n_hidden=64, dropout=0.1):
        super(Decoder, self).__init__()

        self.embedding = nn.Embedding(tgt_vocab_size, n_features, padding_idx=PADDING_VALUE)
        self.positional_encoding = PositionalEncoding(n_features, max_len=MAX_LENGTH)
        self.decoder_blocks = nn.ModuleList([DecoderBlock(n_features=n_features, n_heads=n_heads, n_hidden=n_hidden, dropout=dropout) for _ in range(n_blocks)])
        self.fc = nn.Linear(n_features, tgt_vocab_size)
    
    def subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1).float()
        mask = mask.masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def forward(self, y, z, src_mask):
        out = self.embedding(y)
        out = self.positional_encoding(out)
    
        tgt_mask = self.subsequent_mask(y.size(0))
        for block in self.decoder_blocks:
            out = block(out, z, src_mask=src_mask, tgt_mask=tgt_mask)
            
        out = F.log_softmax(self.fc(out), dim=2)
        
        return out


class Transformers(nn.Module):
    def __init__(self, input_n_words, output_n_words, n_blocks=3, n_features=256, n_heads=16, n_hidden=1024):
        super(Transformers, self).__init__()
        self.encoder = Encoder(src_vocab_size=input_n_words, n_blocks=n_blocks, 
                               n_features=n_features, n_heads=n_heads, n_hidden=n_hidden)
        self.decoder = Decoder(tgt_vocab_size=output_n_words, n_blocks=n_blocks, 
                               n_features=n_features, n_heads=n_heads, n_hidden=n_hidden)
    
    def forward(self, src_seqs, tgt_seqs, src_mask):
        z = self.encoder(src_seqs, mask=src_mask)
        outputs = self.decoder(tgt_seqs[:-1], z, src_mask=src_mask)
        return outputs