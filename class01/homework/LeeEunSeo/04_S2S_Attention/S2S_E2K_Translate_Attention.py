import torch 
import torch.nn as nn
import numpy as np
import random
from  sentence_preprocess import read_language
from langchain_community.embeddings import HuggingFaceEmbeddings
from transformers import AutoTokenizer, AutoModel
from torch.utils.data import TensorDataset, DataLoader
import torch.nn.functional as F
from tqdm.auto import tqdm

SOS_token = 0
EOS_token = 1
# MAX_LENGTH = 3000
device = "cuda" if torch.cuda.is_available() else "cpu"  # GPU가 있으면 CUDA 사용

lang_input, lang_output, pairs = read_language('ENG', 'KOR', reverse=False, verbose=False)
for idx in range(10):
    print(random.choice(pairs))

tokenizer = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-ko-en")
# padding : ������� ���� ���̷� ����.
# truncation : ������ ���̸� �ش� ���� �ִ� ���̷� ������.
# return_tensors : framework
encoded_input = tokenizer(lang_input, padding=True, truncation=True, return_tensors="pt")  
decoded_input = tokenizer(lang_output, padding=True, truncation=True, return_tensors="pt")

input_ids     = encoded_input["input_ids"].to(device)      # (N, src_len)
target_ids    = decoded_input["input_ids"].to(device)      # (N, tgt_len)
dataset = TensorDataset(input_ids, target_ids)
loader  = DataLoader(dataset, batch_size= 4, shuffle=True)

class Encoder(nn.Module):
    def __init__(self,
                vocab_size : int,
                embed_size : int,
                hidden_size,
                dropout = 0.1,
                device = 'cuda'
                ):
        super().__init__()
        print("vocabsize",vocab_size, "embed_size", embed_size, "hidden_size", hidden_size)
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(vocab_size, embed_size, padding_idx=tokenizer.pad_token_id)
        self.GRU = nn.GRU(embed_size, hidden_size, batch_first=True)

    def forward(self, input):
        embedded  = self.embedding(input)
        output, hidden = self.GRU(embedded)
        return output, hidden

class Decoder(nn.Module):
    def __init__(self,
                vocab_size : int,
                embed_size : int,
                hidden_size : int,   #
                max_length = 20,
                device = 'cuda'
                ):
        super().__init__()

        self.hidden_size  = hidden_size
        self.embedding = nn.Embedding(vocab_size, embed_size, padding_idx=tokenizer.pad_token_id).to(device)
        self.GRU  = nn.GRU(embed_size, hidden_size, batch_first=True)
        self.out     = nn.Linear(hidden_size, vocab_size).to(device)
        
        self.max_len = max_length
        self.device = device
        self.to(device)
        
    def forward(self, encoder_outputs, encoder_hidden, target_tensor=None, teacher_forcing_ratio=0.5):
        batch_size  = encoder_outputs.size(0)
        if target_tensor is not None:
            tgt_len = target_tensor.size(1) # sentance_length
        else:
            tgt_len = self.max_len
        vocab_size  = self.out.out_features

        decoder_input  = torch.full((batch_size,1), SOS_token, dtype=torch.long).to(self.device)
        decoder_hidden = encoder_hidden.to(self.device)
         
        outputs = torch.zeros(batch_size, tgt_len, vocab_size, device=self.device)
        
        for sent_idx in range(tgt_len):
            sel_tok = self.forward_1_step(decoder_input, decoder_hidden, encoder_outputs)

            outputs[:, sent_idx, :] = sel_tok.squeeze(1)

            if (target_tensor is not None and random.random() < teacher_forcing_ratio):
                decoder_input = target_tensor[:, sent_idx].unsqueeze(1)
            else:
                top1 = sel_tok.argmax(dim=2)  
                decoder_input = top1 
         
        return outputs

    def forward_1_step(self, input_word, hidden, hidden_from_encoder):
        embedded_input_word = self.embedding(input_word)
        output, hidden1 = self.GRU(embedded_input_word, hidden)
        logits = self.out(output)
        return logits

def save_checkpoint(encoder, decoder, optimizer, epoch, path = "check_point.pth"):
    torch.save({
        'epoch': epoch,
        'encoder_state_dict': encoder.state_dict(),
        'decoder_state_dict': decoder.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()
    }, path)
    print(f"Checkpoint saved to {path}")

embed_size  = 4
hidden_size = 4
src_vocab   = tokenizer.vocab_size
tgt_vocab   = tokenizer.vocab_size

encoder = Encoder(src_vocab, embed_size, hidden_size).to(device)  # 모델을 GPU로 이동
decoder = Decoder(tgt_vocab, embed_size, hidden_size, max_length=target_ids.size(1), device=device).to(device)  # 모델을 GPU로 이동
optimizer = torch.optim.Adam(list(encoder.parameters()) + list(decoder.parameters()), lr=1e-3)
criterion = nn.CrossEntropyLoss(ignore_index=-100)

num_epochs = 10
for epoch in range(1, num_epochs+1):
    encoder.train(); decoder.train()
    total_loss = 0

    loop = tqdm(loader, desc=f"Epoch {epoch}/{num_epochs}", unit="batch")
    for src_ids, tgt_ids in loop:
        src_ids, tgt_ids = src_ids.to(device), tgt_ids.to(device)  # 데이터도 GPU로 이동

        optimizer.zero_grad()
        enc_outs, enc_hidden = encoder(src_ids)
        dec_outputs = decoder(enc_outs, enc_hidden, target_tensor=tgt_ids)

        batch_loss = 0
        for t in range(tgt_ids.size(1)):
            batch_loss += criterion(dec_outputs[:, t, :], tgt_ids[:, t])
        batch_loss = batch_loss / tgt_ids.size(1)

        batch_loss.backward()
        optimizer.step()
        total_loss += batch_loss.item()
        loop.set_postfix(loss=batch_loss.item())
    
    avg_loss = total_loss / len(loader)    
    print(f"Epoch {epoch:2d}  Loss: {avg_loss:.4f}")
    
save_checkpoint(encoder, decoder, optimizer, num_epochs, path='final_model.pth')