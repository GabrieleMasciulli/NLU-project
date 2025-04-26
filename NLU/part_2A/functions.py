from utils import DEVICE, PAD_TOKEN
import torch
import torch.nn as nn
from conll import evaluate
from sklearn.metrics import classification_report


def collate_fn(data):
    """
    This function is used to pad the sequences of a batch to the same length
    and move them to the selected device.
    Input:
        data: list of dictionaries containing the data
    Output:
        padded_data: dictionary containing the padded data
    """
    def merge(sequences):
        """
        merge from shape (batch_size, max_len) to shape (batch_size, max_len)
        by using padding tokens

        input:
            sequences: list of list of tokens
        output:
            padded_seqs: tensor of shape (batch_size, max_len)
            lengths: list of lengths of each sequence (before padding)
        """

        lengths = [len(seq) for seq in sequences]
        max_len = 1 if max(lengths) == 0 else max(lengths)
        # We create a matrix full of PAD_TOKEN with the shape
        # (batch_size, max_length_of_a_sequence)
        padded_seqs = torch.LongTensor(
            len(sequences), max_len).fill_(PAD_TOKEN)
        for i, seq in enumerate(sequences):
            end = lengths[i]
            padded_seqs[i, :end] = seq
        padded_seqs = padded_seqs.detach()  # remove from computational graph
        return padded_seqs, lengths

    # sort data by length of utterance in descending order
    data.sort(key=lambda x: len(x['utterance']), reverse=True)
    new_item = {}

    # group all the values for each sample together
    # (all utterances go together, same for intent, slots)
    for key in data[0].keys():  # keys: ['utterance', 'intent', 'slots']
        new_item[key] = [d[key] for d in data]

    # we just need one length for each padded seq,
    # since len(utterance_padded) == len(slots)
    src_utt, _ = merge(new_item['utterance'])
    y_slots, y_lengths = merge(new_item['slots'])
    intents = torch.LongTensor(new_item['intent'])

    # loading the Tensors to the selected device
    src_utt = src_utt.to(DEVICE)
    y_slots = y_slots.to(DEVICE)
    intents = intents.to(DEVICE)

    return {
        'utterances': src_utt,
        'intents': intents,
        'slots': y_slots,
        'slots_len': y_lengths
    }


def init_weights(m: nn.Module):
    """
    This function initializes the weights of the model.
    """
    for module in m.modules():
        if type(module) in [
            nn.GRU,
            nn.LSTM,
            nn.RNN,
        ]:
            for name, param in module.named_parameters():
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
            if type(module) in [nn.Linear]:
                nn.init.uniform_(module.weight.data, -0.1, 0.1)
                if module.bias is not None:
                    module.bias.data.fill_(0.01)


def train_loop(data, optimizer, criterion_slot, criterion_intent, model, clip=5):
    """
    This function performs a training loop over the data.
    """
    model.train()
    loss_arr = []

    for batch in data:
        optimizer.zero_grad()
        slot_logits, intent_logits = model(
            batch['utterances'], batch['slots_len'])
        loss_slot = criterion_slot(slot_logits, batch['slots'])
        loss_intent = criterion_intent(intent_logits, batch['intents'])
        # @todo check if there is a better way to combine the two losses
        loss = loss_slot + loss_intent
        loss_arr.append(loss.item())
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()  # update the weights

    return loss_arr


def eval_loop(data, criterion_slot, criterion_intent, model, lang):
    """
    This function performs an evaluation loop over the data.
    """
    model.eval()
    loss_arr = []

    slot_preds = []
    slot_golds = []

    intent_preds = []
    intent_golds = []

    with torch.no_grad():
        for batch in data:
            slot_logits, intent_logits = model(
                batch['utterances'], batch['slots_len'])
            loss_slot = criterion_slot(slot_logits, batch['slots'])
            loss_intent = criterion_intent(intent_logits, batch['intents'])
            loss = loss_slot + loss_intent
            loss_arr.append(loss.item())

            # intent inference --> get the highest probable class
            out_intents = [lang.id2intent[id]
                           for id in torch.argmax(intent_logits, dim=1).tolist()]
            gold_intents = [lang.id2intent[id]
                            for id in batch['intents'].tolist()]
            intent_preds.extend(out_intents)
            intent_golds.extend(gold_intents)

            # slot inference --> get the highest probable class for each token
            # (batch_size, seq_len)
            out_slots = torch.argmax(slot_logits, dim=1)

            for id_seq, seq in enumerate(out_slots):
                length = batch['slots_len'].tolist()[id_seq]
                # ignore padding tokens
                utt_ids = batch['utterance'][id_seq][:length].tolist()
                gt_ids = batch['slots'][id_seq].tolist()
                gt_slots = [lang.id2slot[id] for id in gt_ids[:length]]
                utterance = [lang.id2word[id] for id in utt_ids]
                to_decode = seq[:length].tolist()

                # build a list of (word, slot) tuples (pairs) for reference and prediction
                slot_preds.append(
                    [(utterance[i], elem) for i, elem in enumerate(gt_slots)])
                tmp_seq = []
                for id_el, elem in enumerate(to_decode):
                    tmp_seq.append((utterance[id_el], lang.id2slot[elem]))
                slot_golds.append(tmp_seq)
    try:
        results = evaluate(slot_golds, slot_preds)
    except Exception as ex:
        # Sometimes the model predicts a class that is not in REF
        print("Warning:", ex)
        gold_set = set([x[1] for x in slot_golds])
        pred_set = set([x[1] for x in slot_preds])
        print(gold_set.difference(pred_set))
        results = {"total": {"f": 0}}

    report_intent = classification_report(
        intent_golds, intent_preds, output_dict=True, zero_division=False)

    return results, report_intent, loss_arr
