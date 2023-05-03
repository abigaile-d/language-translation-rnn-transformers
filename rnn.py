import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
from torchtext.data.metrics import bleu_score

from data import MAX_LENGTH, SOS_token, EOS_token


class RNNTools():
    def __init__(self, device):
        self.device = device

    def collate(list_of_samples):
        sorted_samples = sorted(list_of_samples, key=lambda x: x[0].shape[0], reverse=True)

        src_seqs = [x[0] for x in sorted_samples]
        src_seq_lengths = [len(x) for x in src_seqs]
        src_seqs = pad_sequence(src_seqs)

        tgt_seqs = [x[1] for x in sorted_samples]
        tgt_seqs = pad_sequence(tgt_seqs)
        
        return src_seqs, src_seq_lengths, tgt_seqs

    def translate(self, rnn, pad_src_seqs, src_seq_lengths):
        with torch.no_grad():
            pad_src_seqs = pad_src_seqs.to(self.device)

            rnn.eval()

            outputs = rnn(pad_src_seqs, None, src_seq_lengths, False)
            _, out_seqs = torch.max(outputs, dim=2)

            return out_seqs
    
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

    def compute_bleu_score(self, model, dataloader, output_lang):
        candidate_corpus = []
        references_corpus = []
        for pad_src_seqs, src_seq_lengths, pad_tgt_seqs in dataloader:
            out_seqs = self.translate(model, pad_src_seqs, src_seq_lengths)
            candidate_corpus.extend([self.seq_to_tokens(seq, output_lang) for seq in out_seqs.T])
            references_corpus.extend([[self.seq_to_tokens(seq, output_lang)] for seq in pad_tgt_seqs.T])

        score = bleu_score(candidate_corpus, references_corpus)
        return score


class Encoder(nn.Module):
    def __init__(self, src_dictionary_size, embed_size, hidden_size):
        super(Encoder, self).__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(src_dictionary_size, embed_size)
        self.gru = nn.GRU(input_size=embed_size, hidden_size=hidden_size)

    def forward(self, pad_seqs, seq_lengths, hidden):
        outputs = self.embedding(pad_seqs)
        outputs = pack_padded_sequence(outputs, seq_lengths)
        outputs, hidden = self.gru(outputs, hidden)
        outputs = pad_packed_sequence(outputs)[0]
        return outputs, hidden

    def init_hidden(self, batch_size=1):
        return torch.zeros(1, batch_size, self.hidden_size)


class Decoder(nn.Module):
    def __init__(self, tgt_dictionary_size, embed_size, hidden_size):
        super(Decoder, self).__init__()
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(tgt_dictionary_size, embed_size)
        self.gru = nn.GRU(input_size=embed_size, hidden_size=hidden_size)
        self.out = nn.Linear(hidden_size, tgt_dictionary_size)

    def forward(self, hidden, pad_tgt_seqs=None, teacher_forcing=False):
        if pad_tgt_seqs is None:
            assert not teacher_forcing, 'Cannot use teacher forcing without a target sequence.'


        seq_len = MAX_LENGTH
        if pad_tgt_seqs is not None:
            seq_len = pad_tgt_seqs.shape[0]
        outputs = []
        
        input_word = torch.tensor([SOS_token] * hidden.shape[1])[None, :]
        for i in range(seq_len):
            embedded = F.relu(self.embedding(input_word))
            pred_word, hidden = self.gru(embedded, hidden)
            pred_word = F.log_softmax(self.out(pred_word), dim=2)
            if teacher_forcing:
                input_word = pad_tgt_seqs[i][None, :]
            else:
                _, input_word = torch.max(pred_word, 2)
            outputs.append(pred_word)
        
        outputs = torch.cat(outputs)

        return outputs, hidden


class RNN(nn.Module):
    def __init__(self, input_n_words, output_n_words, embed_size=256, hidden_size=256):
        super(RNN, self).__init__()
        self.encoder = Encoder(input_n_words, embed_size, hidden_size)
        self.decoder = Decoder(output_n_words, embed_size, hidden_size)
    
    def forward(self, src_seqs, tgt_seqs, src_seq_lengths, teacher_forcing):
        hidden = self.encoder.init_hidden(batch_size=src_seqs.shape[1])
        _, hidden = self.encoder(src_seqs, src_seq_lengths, hidden)
        outputs, hidden = self.decoder(hidden, tgt_seqs, teacher_forcing=teacher_forcing)

        return outputs
    
