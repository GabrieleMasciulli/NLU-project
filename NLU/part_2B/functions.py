import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from sklearn.metrics import accuracy_score, classification_report
from conll import evaluate
from utils import DEVICE, SLOT_PAD_LABEL_ID, Lang, tokenizer

# --- Training Loop ---


def train_loop(model, data_loader: DataLoader, optimizer, scheduler):
    model.train()
    total_loss = 0
    num_batches = len(data_loader)
    progress_bar = tqdm(data_loader, desc="Training", leave=False)

    for batch in progress_bar:
        # Move batch to device
        input_ids = batch['input_ids'].to(DEVICE)
        attention_mask = batch['attention_mask'].to(DEVICE)
        intent_labels = batch['intent_id'].to(DEVICE)
        slot_labels = batch['slot_labels'].to(DEVICE)

        optimizer.zero_grad()

        # Forward pass
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            intent_labels=intent_labels,
            slot_labels=slot_labels,
        )

        # Access loss from the dictionary output
        loss = outputs['loss']

        # Backward pass and optimization
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()  # Update learning rate

        total_loss += loss.item()
        progress_bar.set_postfix({'loss': loss.item()})

    return total_loss / num_batches

# --- Evaluation Loop ---


def eval_loop(model, data_loader: DataLoader, lang: Lang, is_test=False):
    model.eval()
    all_intent_preds = []
    all_intent_labels = []
    slot_preds = []
    slot_golds = []

    progress_bar = tqdm(data_loader, desc="Evaluating", leave=False)

    with torch.no_grad():
        for batch in progress_bar:
            # Move batch to device
            input_ids = batch['input_ids'].to(DEVICE)
            attention_mask = batch['attention_mask'].to(DEVICE)
            intent_labels = batch['intent_id'].to(DEVICE)
            slot_labels = batch['slot_labels'].to(
                DEVICE)  # Shape: (batch, seq_len)

            # Forward pass
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )

            # Get Logits from dictionary output
            intent_logits = outputs['intent_logits']

            # Shape: (batch, seq_len, num_slot_labels)
            slot_logits = outputs['slot_logits']

            # --- Intent Prediction ---
            intent_preds = torch.argmax(intent_logits, dim=1)
            all_intent_preds.extend(intent_preds.cpu().numpy())
            all_intent_labels.extend(intent_labels.cpu().numpy())

            # --- Slot Prediction ---
            # Shape: (batch, seq_len)
            batch_slot_preds_tensor = torch.argmax(slot_logits, dim=-1)

            # Process each sequence in the batch
            for i in range(slot_labels.shape[0]):
                # Get the true length of the sequence using the mask
                seq_len = int(attention_mask[i].sum().item())

                # Get input tokens for this sequence
                input_tokens = input_ids[i][:seq_len].cpu().tolist()

                # Decode tokens to words
                words = tokenizer.convert_ids_to_tokens(input_tokens)

                # Get predictions and gold labels for this sequence
                pred_slot_ids = batch_slot_preds_tensor[i][:seq_len].cpu(
                ).tolist()
                gold_slot_ids = slot_labels[i][:seq_len].cpu().tolist()

                # Convert slot IDs to slot labels and create (word, slot) tuples
                pred_sequence = []
                gold_sequence = []

                for j in range(seq_len):
                    if gold_slot_ids[j] != SLOT_PAD_LABEL_ID:
                        word = words[j]
                        pred_slot = lang.id2slot.get(
                            pred_slot_ids[j], f"UNKNOWN_{pred_slot_ids[j]}")
                        gold_slot = lang.id2slot.get(
                            gold_slot_ids[j], f"UNKNOWN_{gold_slot_ids[j]}")

                        pred_sequence.append((word, pred_slot))
                        gold_sequence.append((word, gold_slot))

                if pred_sequence:  # Only add non-empty sequences
                    slot_preds.append(pred_sequence)
                    slot_golds.append(gold_sequence)

    # --- Calculate Metrics ---
    # Intent Accuracy
    intent_accuracy = accuracy_score(all_intent_labels, all_intent_preds)

    # Slot evaluation using conll evaluate function
    if not slot_golds:
        print("Warning: No valid slot sequences found for evaluation.")
        slot_results = {"total": {"f": 0.0, "p": 0.0, "r": 0.0}}
    else:
        try:
            slot_results = evaluate(slot_golds, slot_preds)
        except Exception as ex:
            print("Warning:", ex)
            gold_set = set([x[1] for seq in slot_golds for x in seq])
            pred_set = set([x[1] for seq in slot_preds for x in seq])
            print("Gold - Pred:", gold_set.difference(pred_set))
            print("Pred - Gold:", pred_set.difference(gold_set))
            slot_results = {"total": {"f": 0.0, "p": 0.0, "r": 0.0}}

    # Intent classification report
    intent_names = [lang.id2intent.get(i, f"UNKNOWN_{i}") for i in sorted(
        set(all_intent_labels + all_intent_preds))]
    intent_report_dict = classification_report(
        all_intent_labels, all_intent_preds,
        target_names=intent_names, output_dict=True, zero_division=0
    )

    results = {
        "intent_acc": intent_accuracy,
        "slot_f1": slot_results["total"]["f"],
        "slot_precision": slot_results["total"]["p"],
        "slot_recall": slot_results["total"]["r"],
        "slot_results": slot_results,
        "intent_report": intent_report_dict
    }
    return results
