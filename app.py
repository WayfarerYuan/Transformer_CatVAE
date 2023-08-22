import streamlit as st
import torch.nn.functional as F
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
import re

# Assuming TransformerEncoder and TransformerDecoder are defined above
EMBEDDING_DIM = 16
HIDDEN_DIM = 16
LATENT_DIM = 16 # Dimension of the latent space
SEQ_LEN = 16 # Max length of the sequence

# Gumbel softmax temperature
TAU = 1.0

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device('mps')
torch.random.manual_seed(1024)

# Pass embeded into decoder instead of using the original x
class TransformerEncoder(nn.Module):
    def __init__(self, d_model=EMBEDDING_DIM, nhead=4, num_layers=2):
        super(TransformerEncoder, self).__init__()
        self.embedding = nn.Embedding(VOCAB_SIZE, d_model)
        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model, nhead), num_layers
        )
        self.fc_logits = nn.Linear(d_model, LATENT_DIM)

    def forward(self, x):
        embedded = self.embedding(x).permute(1, 0, 2)  # Transformer expects seq_len, batch, features
        transformed = self.transformer_encoder(embedded)
        # Use the final state to predict logits for latent space
        logits = self.fc_logits(transformed[-1])
        return logits, embedded


class TransformerDecoder(nn.Module):
    def __init__(self, d_model=EMBEDDING_DIM, nhead=4, num_layers=2):
        super(TransformerDecoder, self).__init__()
        self.embedding = nn.Embedding(VOCAB_SIZE, d_model)
        self.transformer_decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(d_model, nhead), num_layers
        )
        self.fc_out = nn.Linear(d_model, VOCAB_SIZE)
        self.fc_z = nn.Linear(LATENT_DIM, d_model)  # Convert z to feature size for transformer

    def forward(self, embedded, z):
        # embedded = self.embedding(x).permute(1, 0, 2) # Transformer expects [seq_len, batch, features], permute函数用于改变张量的维度顺序
        z_adjusted = self.fc_z(z).unsqueeze(0)
        output = self.transformer_decoder(embedded, z_adjusted)
        return self.fc_out(output.permute(1, 0, 2))


class TransformerCVAE(nn.Module):
    def __init__(self):
        super(TransformerCVAE, self).__init__()
        self.encoder = TransformerEncoder()
        self.decoder = TransformerDecoder()

    def reparameterize(self, logits):
        return F.gumbel_softmax(logits, tau=TAU, hard=False, dim=-1)

    def forward(self, x):
        logits, emb = self.encoder(x)
        z = self.reparameterize(logits)
        return self.decoder(emb, z), logits
    
def load_and_preprocess_wikitext(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        text = f.read()

    # Use regular expressions to split the text into sentences
    sentences = re.split(r'(?<=[.!?])\s+', text)
    sentences = [sentence.strip() for sentence in sentences]
    
    return sentences

train_file_path = "wikitext-2/wiki.train.tokens"
test_file_path = "wikitext-2/wiki.test.tokens"
val_file_path = "wikitext-2/wiki.valid.tokens"

wikitext_sentences_train = load_and_preprocess_wikitext(train_file_path)
wikitext_sentences_test = load_and_preprocess_wikitext(test_file_path)
wikitext_sentences_val = load_and_preprocess_wikitext(val_file_path)

# Hyperparameters
BATCH_SIZE = 32
PAD_TOKEN = "<PAD>"
UNK_TOKEN = "<UNK>"

# Tokenize the data
tokens = [word for sentence in wikitext_sentences_train for word in sentence.split()]

# Build vocabulary
vocab = [PAD_TOKEN, UNK_TOKEN] + list(set(tokens))
word_index = {word: index for index, word in enumerate(vocab)}
# 添加新的tokens
SOS_TOKEN = '<SOS>'
EOS_TOKEN = '<EOS>'
word_index[SOS_TOKEN] = len(word_index)
word_index[EOS_TOKEN] = len(word_index)
vocab = {v: k for k, v in word_index.items()}
# Convert tokens to integers
def tokenize_and_encode(text):
    return [word_index.get(word, word_index[UNK_TOKEN]) for word in text.split()]

encoded_data_train = [tokenize_and_encode(sentence) for sentence in wikitext_sentences_train]

# Create a PyTorch Dataset
class WikiDataset(Dataset):
    def __init__(self, data, sequence_length):
        self.data = data
        self.sequence_length = sequence_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        if len(sample) < self.sequence_length:
            sample.extend([word_index[PAD_TOKEN]] * (self.sequence_length - len(sample)))
        else:
            sample = sample[:self.sequence_length]
        return torch.tensor(sample)

# dataset = WikiDataset(encoded_data_train, SEQUENCE_LENGTH)
# dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
# Split the data into train and validation sets
dataset = WikiDataset(encoded_data_train, SEQ_LEN)
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=True)

VOCAB_SIZE = len(vocab)




class MultiMultiSignalingGame:
    def __init__(self, senders: list, receivers: list, optimizer, criterion):
        self.senders = senders
        self.receivers = receivers
        self.optimizer = optimizer
        self.criterion = criterion

    def play_round(self, states):
        all_decoded_outputs = []
        all_logits = []
        
        for i, sender in enumerate(self.senders):
            # Sender encodes the state
            logits, emb = sender(states[i])
            all_logits.append(logits)
            z = F.gumbel_softmax(logits, tau=TAU, hard=False, dim=-1)
            
            # Each receiver decodes the signal from the sender
            for receiver in self.receivers:
                decoded_output = receiver(emb, z)
                all_decoded_outputs.append(decoded_output)
      
        # Calculate loss
        loss = self.compute_loss(states, all_decoded_outputs, all_logits, beta=1.0)
        
        # Update model parameters
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # Capture the input and output sentences
        _, input_sentence_ids = torch.max(states[0], dim=1)
        input_sentence_ids = input_sentence_ids.cpu().numpy()
        input_sentence = ' '.join([vocab[idx] for idx in input_sentence_ids])

        _, output_sentence_ids = torch.max(all_decoded_outputs[0][0], dim=1)
        output_sentence_ids = output_sentence_ids.cpu().numpy()
        output_sentence = ' '.join([vocab[idx] for idx in output_sentence_ids])

        return loss.item(), input_sentence, output_sentence

    def compute_loss(self, original_states, decoded_states, logits, beta):
        recon_loss = sum([self.criterion(decoded_state.view(-1, VOCAB_SIZE), original_state.view(-1))
                          for original_state, decoded_state in zip(original_states * len(self.receivers), decoded_states)])
        
        # Calculate KLD loss
        kld_losses = []
        for logit in logits:
            mean, logvar = torch.chunk(logit, 2, dim=-1)
            kld_loss = -0.5 * torch.sum(1 + logvar - mean.pow(2) - logvar.exp())
            kld_losses.append(kld_loss)

        return recon_loss + beta * sum(kld_losses)
    

def train_signal_game(NUM_SENDERS, NUM_RECEIVERS, num_rounds):
    senders = [TransformerEncoder().to(device) for _ in range(NUM_SENDERS)]
    receivers = [TransformerDecoder().to(device) for _ in range(NUM_RECEIVERS)]

    params = [list(sender.parameters()) for sender in senders]
    params.extend([list(receiver.parameters()) for receiver in receivers])
    optimizer = torch.optim.Adam([param for sublist in params for param in sublist], lr=0.001)
    criterion = torch.nn.CrossEntropyLoss()

    game = MultiMultiSignalingGame(senders, receivers, optimizer, criterion)

    losses = []

    # Use Streamlit's progress bar
    progress_bar = st.progress(0)

    for round in range(num_rounds):
        states = [torch.randint(VOCAB_SIZE, (BATCH_SIZE, 16)).to(device) for _ in range(NUM_SENDERS)]
        loss = game.play_round(states)
        losses.append(loss)
        progress_bar.progress(round / num_rounds)

    # Display the plot directly using Streamlit
    plt.plot(losses, label='losses')
    plt.xlabel('Round')
    plt.ylabel('Loss')
    plt.legend()
    st.pyplot()

# Streamlit UI
st.title('Multi-Multi Signaling Game')

NUM_SENDERS = st.sidebar.slider("NUM_SENDERS", 1, 10, 3)
NUM_RECEIVERS = st.sidebar.slider("NUM_RECEIVERS", 1, 10, 3)
num_rounds = st.sidebar.slider("num_rounds", 1000, 20000, 10000, 1000)

if st.sidebar.button('Start'):
    train_signal_game(NUM_SENDERS, NUM_RECEIVERS, num_rounds)
