import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from sklearn.metrics import accuracy_score, f1_score, classification_report
from utils import DEVICE, SLOT_PAD_LABEL_ID, Lang

# --- Training Loop ---


def train_loop(model, data_loader: DataLoader, optimizer, scheduler, loss_alpha=0.5):
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
            return_dict=True,
            loss_alpha=loss_alpha
        )

        # Access loss from the dictionary output
        loss = outputs['loss']

        # Backward pass and optimization
        loss.backward()
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
    flat_slot_preds = []
    flat_slot_labels = []

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
                return_dict=True
            )

            # Get Logits from dictionary output
            intent_logits = outputs['intent_logits']

            # Shape: (batch, seq_len, num_slot_labels)
            slot_logits = outputs['slot_logits']

            # --- Intent Prediction ---
            intent_preds = torch.argmax(intent_logits, dim=1)
            all_intent_preds.extend(intent_preds.cpu().numpy())
            all_intent_labels.extend(intent_labels.cpu().numpy())

            # --- Slot Prediction using CRF Decode ---
            crf_mask = attention_mask.bool()
            # Ensure mask shape matches slot_logits shape
            if crf_mask.dim() > 2 and crf_mask.shape[1] != slot_logits.shape[1]:
                crf_mask = crf_mask[:, :slot_logits.shape[1]]

            # Decode returns List[List[int]], len(outer_list)=batch_size
            batch_slot_preds = model.crf.decode(slot_logits, mask=crf_mask)

            # Flatten predictions and labels, respecting the mask and PAD ID
            # Iterate through each sequence in the batch
            # Iterate through batch items
            for i in range(slot_labels.shape[0]):
                # Get the true length of the sequence using the mask
                seq_len = int(attention_mask[i].sum().item())
                # Get the predicted sequence for this item (up to true length)
                preds_for_seq = batch_slot_preds[i][:seq_len]
                labels_for_seq = slot_labels[i][:seq_len].cpu().numpy()

                # Iterate through tokens in the sequence
                for j in range(seq_len):
                    if labels_for_seq[j] != SLOT_PAD_LABEL_ID:
                        flat_slot_preds.append(preds_for_seq[j])
                        flat_slot_labels.append(labels_for_seq[j])

    # --- Calculate Metrics ---
    # Intent Accuracy
    intent_accuracy = accuracy_score(all_intent_labels, all_intent_preds)

    # Slot F1 Score - Already flattened and filtered
    # Ensure there are labels to evaluate
    if not flat_slot_labels:
        print("Warning: No valid slot labels found for evaluation.")
        slot_f1_macro = 0.0
        slot_f1_micro = 0.0
        slot_report_str = "No valid slot labels found."
        slot_report_dict = {}
    else:
        # Get unique labels present in the actual data (excluding PAD/IGNORE)
        # Use the already flattened and filtered labels
        present_labels = sorted(list(set(flat_slot_labels)))
        # Map IDs to names for the report, only for labels actually present
        target_names = [lang.id2slot.get(
            idx, f"UNKNOWN_{idx}") for idx in present_labels]

        # Calculate metrics
        slot_f1_macro = f1_score(flat_slot_labels, flat_slot_preds,
                                 labels=present_labels, average='macro', zero_division=0)
        slot_f1_micro = f1_score(flat_slot_labels, flat_slot_preds,
                                 labels=present_labels, average='micro', zero_division=0)
        slot_report_str = classification_report(
            flat_slot_labels,
            flat_slot_preds,
            labels=present_labels,
            target_names=target_names,
            digits=4,
            zero_division=0
        )

        slot_report_dict = classification_report(
            flat_slot_labels,
            flat_slot_preds,
            labels=present_labels,
            target_names=target_names,
            digits=4,
            zero_division=0,
            output_dict=True
        )

    results = {
        "intent_acc": intent_accuracy,
        "slot_f1_macro": slot_f1_macro,
        "slot_f1_micro": slot_f1_micro,
        "slot_report_str": slot_report_str,
        "slot_report_dict": slot_report_dict
    }

    if is_test:
        print("\n--- Test Results ---")
        print(f"Intent Accuracy: {intent_accuracy:.4f}")
        print(f"Slot F1 (Macro): {slot_f1_macro:.4f}")
        print(f"Slot F1 (Micro): {slot_f1_micro:.4f}")
        print("\nSlot Classification Report:")
        print(slot_report_str)

    return results
