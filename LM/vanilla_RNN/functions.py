import torch
import math
import torch.nn as nn
from utils import DEVICE


def collate_fn(data, pad_token):
    def merge(sequences):
        """
        merge from batch * sent_len to batch * max_len
        """
        lengths = [len(seq) for seq in sequences]
        max_len = 1 if max(lengths) == 0 else max(lengths)
        # Pad token is zero in our case
        # So we create a matrix full of PAD_TOKEN (i.e. 0) with the shape
        # batch_size X maximum length of a sequence
        padded_seqs = torch.LongTensor(
            len(sequences), max_len).fill_(pad_token)
        for i, seq in enumerate(sequences):
            end = lengths[i]
            padded_seqs[i, :end] = seq  # We copy each sequence into the matrix
        # We remove these tensors from the computational graph
        padded_seqs = padded_seqs.detach()
        return padded_seqs, lengths

    # Sort data by seq lengths
    data.sort(key=lambda x: len(x["source"]), reverse=True)
    new_item = {}
    for key in data[0].keys():
        new_item[key] = [d[key] for d in data]

    source, _ = merge(new_item["source"]


                      )
    target, lengths = merge(new_item["target"])

    new_item["source"] = source.to(DEVICE)
    new_item["target"] = target.to(DEVICE)
    new_item["number_tokens"] = sum(lengths)
    return new_item


def train_loop(data, optimizer, criterion, model, clip=5):
    model.train()
    loss_array = []
    number_of_tokens = []

    for sample in data:
        optimizer.zero_grad()  # Zeroing the gradient
        output = model(sample['source'])
        loss = criterion(output, sample['target'])
        loss_array.append(loss.item() * sample["number_tokens"])
        number_of_tokens.append(sample["number_tokens"])
        loss.backward()  # Compute the gradient, deleting the computational graph
        # clip the gradient to avoid exploding gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()  # Update the weights

    return sum(loss_array) / sum(number_of_tokens)


def eval_loop(data, eval_criterion, model):
    model.eval()
    loss_to_return = []
    loss_array = []
    number_of_tokens = []

    with torch.no_grad():  # It used to avoid the creation of computational graph
        for sample in data:
            source = sample['source']
            target = sample['target']
            num_tokens = sample['num_tokens']

            output = model(source)
            loss = eval_criterion(output, target)
            loss_array.append(loss.item())
            number_of_tokens.append(num_tokens)

    ppl = math.exp(sum(loss_array) / sum(number_of_tokens))
    loss_to_return = sum(loss_array) / sum(number_of_tokens)
    return ppl, loss_to_return


def ensemble_eval_loop(data, eval_criterion, models):
    """Evaluates an ensemble of models by averaging their log probabilities"""

    for model in models:
        model.eval()

    total_loss = 0
    total_tokens = 0
    log_softmax = nn.LogSoftmax(dim=1)

    with torch.no_grad():
        for sample in data:
            source = sample['source']
            target = sample['target']
            num_tokens = sample['num_tokens']

            log_prob_list = []  # shape: (batch_size * seq_len, vocab)

            for model in models:
                output_logits = model(source)

                # Apply log softmax to get log probabilities
                # logit output shape: (batch_size, vocab_size, seq_len), we need (batch_size * seq_len, vocab_size)
                # target shape: (batch_size, seq_len), we need (batch_size * seq_len)
                # Criterion (CrossEntropyLoss) expects Input: (N, C), Target: (N)
                # Reshaping output_logits before applying softmax:
                batch_size, vocab_size, seq_len = output_logits.shape

                # reshapes the tensor s.t. to have vocab_size columns and automatically infers num. of rows
                log_probs = log_softmax(output_logits.permute(
                    0, 2, 1).reshape(-1, vocab_size))
                log_prob_list.append(log_probs)

            # average the log probabilities (stack and mean)
            avg_log_probs = torch.stack(log_prob_list).mean(dim=0)

            # compute loss using the averages log probabilities
            loss = eval_criterion(avg_log_probs, target.reshape(-1))
            total_loss += loss.item()
            total_tokens += num_tokens

    avg_loss = total_loss / total_tokens
    ppl = math.exp(avg_loss)
    return ppl, avg_loss


def init_weights(mat):
    for m in mat.modules():
        if type(m) in [nn.GRU, nn.LSTM, nn.RNN]:
            for name, param in m.named_parameters():
                if 'weight_ih' in name:
                    for idx in range(4):
                        mul = param.shape[0]//4
                        torch.nn.init.xavier_uniform_(
                            param[idx*mul:(idx+1)*mul])
                elif 'weight_hh' in name:
                    for idx in range(4):
                        mul = param.shape[0]//4
                        torch.nn.init.orthogonal_(param[idx*mul:(idx+1)*mul])
                elif 'bias' in name:
                    param.data.fill_(0)
        else:
            if type(m) in [nn.Linear]:
                torch.nn.init.uniform_(m.weight, -0.01, 0.01)
                if m.bias is not None:
                    m.bias.data.fill_(0.01)
