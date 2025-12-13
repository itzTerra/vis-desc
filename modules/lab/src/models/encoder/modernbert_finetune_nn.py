from typing import Optional
from transformers import ModernBertPreTrainedModel, ModernBertModel
from transformers.modeling_outputs import SequenceClassifierOutput
import torch.nn as nn
import torch
import numpy as np
import gc
from torch.optim import AdamW
from torch.utils.data import DataLoader
from sklearn.preprocessing import MinMaxScaler
from transformers import get_linear_schedule_with_warmup
from tqdm.auto import tqdm
from peft import get_peft_model, LoraConfig
from text2features import FeatureExtractorPipeline
from models.encoder.common import (
    PersistentMetrics,
    device,
    CustomDataset,
    calculate_metrics,
    SEED,
)

BATCH_SIZE = 64
NUM_EPOCHS = 40
EARLY_STOPPING_PATIENCE = 5


class ModernBertWithFeaturesTrainable(ModernBertPreTrainedModel):
    def __init__(
        self,
        config,
        feature_input_size,
        dropout_rate,
        feature_hidden_size,
        norm_eps=1e-4,
    ):
        super().__init__(config)
        self.config = config
        self.model = ModernBertModel(config)

        self.feature_ff = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(feature_input_size, feature_hidden_size),
            nn.LayerNorm(feature_hidden_size, eps=norm_eps),
            nn.GELU(),
        )

        bert_hidden_size = config.hidden_size

        # self.feature_scale = nn.Parameter(torch.tensor(0.1))  # Start small
        self.fusion_norm = nn.LayerNorm(
            bert_hidden_size + feature_hidden_size, eps=norm_eps
        )

        self.regressor = nn.Sequential(
            nn.Dropout(dropout_rate),
            # nn.Linear(bert_hidden_size + feature_hidden_size, regressor_hidden_size),
            # nn.LayerNorm(regressor_hidden_size, eps=norm_eps),
            # nn.GELU(),
            # nn.Dropout(dropout_rate * 0.5),
            # nn.Linear(regressor_hidden_size, 1),
            nn.Linear(bert_hidden_size + feature_hidden_size, 1),
        )

        self.loss_fct = nn.MSELoss()

        self.post_init()

        if next(self.parameters()).is_meta:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.to_empty(device=device)

        self._init_custom_weights()

        # for name, param in self.named_parameters():
        #     if "model." in name:  # Freeze the entire ModernBERT
        #         param.requires_grad = False
        #     else:  # Only train custom layers
        #         param.requires_grad = True

        # print(
        #     f"Trainable parameters: {sum(p.numel() for p in self.parameters() if p.requires_grad):,}"
        # )
        # print(f"Total parameters: {sum(p.numel() for p in self.parameters()):,}")

    def _init_custom_weights(self):
        for name, module in self.named_modules():
            if isinstance(module, nn.Linear):
                # Use He initialization for layers followed by ReLU
                if "feature_ff" in name:
                    nn.init.kaiming_normal_(
                        module.weight, mode="fan_in", nonlinearity="relu"
                    )
                    # Add small positive bias to prevent all-negative inputs to ReLU
                    if module.bias is not None:
                        nn.init.constant_(module.bias, 0.01)
                else:
                    nn.init.xavier_uniform_(
                        module.weight, gain=0.01
                    )  # Smaller gain for final regressor
                    if module.bias is not None:
                        nn.init.constant_(module.bias, 0)

            elif isinstance(module, nn.LayerNorm):
                nn.init.constant_(module.weight, 1.0)
                nn.init.constant_(module.bias, 0)

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        features=None,
        labels=None,
        **kwargs,
    ):
        # print(
        #     f"Labels range: [{labels.min():.2f}, {labels.max():.2f}], mean: {labels.mean():.2f}"
        # )
        # print(f"Features range: [{features.min():.2f}, {features.max():.2f}]")
        # print(f"Features std: {features.std():.2f}")

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            **kwargs,
        )
        # Use the CLS token's representation (first token)
        cls_embedding = outputs.last_hidden_state[:, 0, :]
        # print(
        #     f"CLS embedding: shape={cls_embedding.shape}, has NaN: {torch.isnan(cls_embedding).any()}"
        # )
        # print(
        #     f"CLS stats: min={cls_embedding.min():.4f}, max={cls_embedding.max():.4f}"
        # )

        feature_embedding = self.feature_ff(features)
        # feature_embedding = feature_embedding * self.feature_scale  # Scale down
        # print(
        #     f"Feature embeddings range: [{feature_embedding.min():.2f}, {feature_embedding.max():.2f}]"
        # )
        concatenated_embedding = torch.cat((cls_embedding, feature_embedding), dim=1)

        concatenated_embedding = self.fusion_norm(concatenated_embedding)

        logits = self.regressor(concatenated_embedding)
        # print(
        #     f"Logits range: [{logits.min():.2f}, {logits.max():.2f}], mean: {logits.mean():.2f}"
        # )

        loss = None
        if labels is not None:
            loss = self.loss_fct(logits.squeeze(), labels)
        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


def check_gradient_flow(model, step, epoch):
    """Check gradient flow through different parts of the model."""
    grad_stats = {
        "embeddings": [],
        "0.attn.Wo": [],
        "0.mlp.Wi": [],
        "0.mlp.Wo": [],
        "21.attn.Wo": [],
        "21.mlp.Wi": [],
        "21.mlp.Wo": [],
        "feature_ff": [],
        "regressor": [],
    }

    for name, param in model.named_parameters():
        if param.grad is not None:
            grad_norm = param.grad.norm().item()
            for key in grad_stats.keys():
                if key in name:
                    grad_stats[key].append(grad_norm)

    # Print summary statistics
    print(f"\n=== Gradient Flow Check (Epoch {epoch}, Step {step}) ===")
    for layer_name, grads in grad_stats.items():
        if grads:
            avg_grad = np.mean(grads)
            max_grad = np.max(grads)
            min_grad = np.min(grads)
            print(
                f"{layer_name:20s}: avg={avg_grad:.6f}, max={max_grad:.6f}, min={min_grad:.6f}, count={len(grads)}"
            )
        else:
            print(f"{layer_name:20s}: NO GRADIENTS")
    print("=" * 60)


def train_finetuned_mbert(
    train_df,
    tokenizer,
    params,
    seed=SEED,
    val_df=None,
    train_lg_df=None,
    metrics: Optional[PersistentMetrics] = None,
):
    """Unified ModernBERT training function.

    Behavior:
      - If `train_lg_df` is provided and not empty, perform Stage 1 pre-training on it (epochs = params['stage1_epochs']).
      - If `val_df` is provided: compute validation loss each epoch, apply early stopping using constant `EARLY_STOPPING_PATIENCE`.
      - If `val_df` is None: no early stopping; train for full NUM_EPOCHS.

    Params dict expected keys:
      - lr_bert, lr_custom, dropout_rate, weight_decay, optimizer_warmup,
                feature_hidden_size, stage1_epochs, stage1_frozen_bert_epochs,
                stage2_epochs, stage2_frozen_bert_epochs.

    Args:
        train_df: DataFrame with training data for stage 2
        tokenizer: The tokenizer to use
        params: Dict with hyperparameters
        seed: Random seed for reproducibility
        val_df: Optional validation DataFrame (enables early stopping)
        train_lg_df: Optional large dataset DataFrame for stage 1
        metrics: Optional PersistentMetrics object for logging
    """
    stage1_epochs = params.get("stage1_epochs", 0)
    stage2_epochs = (
        params.get("stage2_epochs", NUM_EPOCHS) if val_df is None else NUM_EPOCHS
    )
    lr_bert = params["lr_bert"]
    lr_custom = params["lr_custom"]
    dropout_rate = params["dropout_rate"]
    weight_decay = params["weight_decay"]
    optimizer_warmup = params["optimizer_warmup"]
    feature_hidden_size = params["feature_hidden_size"]
    stage1_frozen_bert_epochs = params["stage1_frozen_bert_epochs"]
    stage2_frozen_bert_epochs = params["stage2_frozen_bert_epochs"]

    use_lora = params.get("use_lora", True)
    lora_r = params.get("lora_r", 8)
    lora_alpha = params.get("lora_alpha", 32)
    lora_dropout = params.get("lora_dropout", 0.1)

    model_name = f"finetuned-mbert{'_lg' if train_lg_df is not None else ''}{'_lora' if use_lora else ''}"
    if metrics is None:
        metrics = PersistentMetrics.dummy()
    metrics.update(
        model=model_name,
        params={
            "stage1_epochs": stage1_epochs,
            "stage2_epochs": stage2_epochs,
            "lr_bert": lr_bert,
            "lr_custom": lr_custom,
            "dropout_rate": dropout_rate,
            "weight_decay": weight_decay,
            "optimizer_warmup": optimizer_warmup,
            "feature_hidden_size": feature_hidden_size,
            "stage1_frozen_bert_epochs": stage1_frozen_bert_epochs,
            "stage2_frozen_bert_epochs": stage2_frozen_bert_epochs,
            "use_lora": use_lora,
            "lora_r": lora_r,
            "lora_alpha": lora_alpha,
            "lora_dropout": lora_dropout,
        },
        seed=seed,
    )

    # Scale features
    scaler = MinMaxScaler()
    if train_lg_df is not None and not train_lg_df.empty:
        lg_features_scaled = scaler.fit_transform(
            np.vstack(train_lg_df["features"].values)
        )
        train_lg_df = train_lg_df.copy()
        train_lg_df["features"] = [f for f in np.nan_to_num(lg_features_scaled)]

        if not train_df.empty:
            sm_features_scaled = scaler.transform(
                np.vstack(train_df["features"].values)
            )
            train_df = train_df.copy()
            train_df["features"] = [f for f in np.nan_to_num(sm_features_scaled)]
    else:
        sm_features_scaled = scaler.fit_transform(
            np.vstack(train_df["features"].values)
        )
        train_df = train_df.copy()
        train_df["features"] = [f for f in np.nan_to_num(sm_features_scaled)]

    if val_df is not None:
        val_df = val_df.copy()
        val_features_scaled = scaler.transform(np.vstack(val_df["features"].values))
        val_df["features"] = [f for f in np.nan_to_num(val_features_scaled)]

    g = torch.Generator()
    g.manual_seed(seed)

    # Create validation loader if needed
    val_loader = None
    if val_df is not None:
        val_dataset = CustomDataset(val_df, tokenizer)
        val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

    # Initialize model with frozen BERT (will be optionally unfrozen per-stage)
    model = ModernBertWithFeaturesTrainable.from_pretrained(
        "answerdotai/ModernBERT-base",
        feature_input_size=FeatureExtractorPipeline.FEATURE_COUNT,
        dropout_rate=dropout_rate,
        feature_hidden_size=feature_hidden_size,
    )

    if use_lora:
        peft_config = LoraConfig(
            task_type=None,
            inference_mode=False,
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            target_modules=["Wqkv", "Wo", "Wi"],
        )
        model.model = get_peft_model(model.model, peft_config)
        model.model.print_trainable_parameters()

    for param in model.model.parameters():
        param.requires_grad = False
    model.to(device)

    # Stage 1: Train on large dataset (optional)
    if train_lg_df is not None and not train_lg_df.empty and stage1_epochs > 0:
        print(
            f"Stage 1: Training on large dataset ({len(train_lg_df)} samples) for {stage1_epochs} epochs"
        )
        lg_train_dataset = CustomDataset(train_lg_df, tokenizer)
        lg_train_loader = DataLoader(
            lg_train_dataset, batch_size=BATCH_SIZE, shuffle=True, generator=g
        )
        optimizer = AdamW(
            [
                {"params": model.model.parameters(), "lr": lr_bert},
                {"params": model.feature_ff.parameters(), "lr": lr_custom},
                {"params": model.regressor.parameters(), "lr": lr_custom},
            ],
            weight_decay=weight_decay,
        )
        total_steps = len(lg_train_loader) * stage1_epochs
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=int(total_steps * optimizer_warmup),
            num_training_steps=total_steps,
        )
        model.train()
        for epoch in range(stage1_epochs):
            # Unfreeze BERT after stage1_frozen_bert_epochs
            if epoch == stage1_frozen_bert_epochs:
                print(
                    f"\n[Stage 1] Unfreezing {'LoRA adapters' if use_lora else 'BERT encoder'} at epoch {epoch + 1}"
                )
                if use_lora:
                    for name, param in model.model.named_parameters():
                        if "lora" in name:
                            param.requires_grad = True
                else:
                    for p in model.model.parameters():
                        p.requires_grad = True
                optimizer = AdamW(
                    [
                        {"params": model.model.parameters(), "lr": lr_bert},
                        {"params": model.feature_ff.parameters(), "lr": lr_custom},
                        {"params": model.regressor.parameters(), "lr": lr_custom},
                    ],
                    weight_decay=weight_decay,
                )
                remaining_steps = len(lg_train_loader) * (stage1_epochs - epoch)
                scheduler = get_linear_schedule_with_warmup(
                    optimizer,
                    num_warmup_steps=int(remaining_steps * optimizer_warmup),
                    num_training_steps=remaining_steps,
                )
            for batch in tqdm(
                lg_train_loader, desc=f"Stage 1 - Epoch {epoch + 1}/{stage1_epochs}"
            ):
                optimizer.zero_grad()
                outputs = model(
                    input_ids=batch["input_ids"].to(device),
                    attention_mask=batch["attention_mask"].to(device),
                    features=batch["features"].to(device),
                    labels=batch["labels"].to(device),
                )
                loss = outputs.loss
                metrics["train_losses"].append(loss.item())
                metrics.update()
                if torch.isnan(loss) or torch.isinf(loss):
                    continue
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                del loss, outputs

        # Aggressive cleanup after Stage 1
        del optimizer, scheduler
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        gc.collect()

    # Stage 2: Fine-tuning on small dataset
    # If Stage 1 potentially unfroze BERT, and Stage 2 requires initial freezing,
    # re-freeze encoder before starting Stage 2 epochs.
    if stage2_frozen_bert_epochs > 0:
        for p in model.model.parameters():
            p.requires_grad = False

    print(
        f"Stage 2: Fine-tuning on small dataset ({len(train_df)} samples) "
        f"{'with' if val_df is not None else 'without'} early stopping (max {NUM_EPOCHS} epochs)"
    )
    sm_train_dataset = CustomDataset(train_df, tokenizer)
    sm_train_loader = DataLoader(
        sm_train_dataset, batch_size=BATCH_SIZE, shuffle=True, generator=g
    )

    optimizer = AdamW(
        [
            {"params": model.model.parameters(), "lr": lr_bert},
            {"params": model.feature_ff.parameters(), "lr": lr_custom},
            {"params": model.regressor.parameters(), "lr": lr_custom},
        ],
        weight_decay=weight_decay,
    )
    total_steps = len(sm_train_loader) * stage2_epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(total_steps * optimizer_warmup),
        num_training_steps=total_steps,
    )

    best_val_loss = float("inf")
    patience_counter = 0
    best_model_state = None

    model.train()
    for epoch in range(stage2_epochs):
        epoch_train_losses = []
        progress_bar = tqdm(
            sm_train_loader,
            desc=f"Stage 2 - Epoch {epoch + 1}/{stage2_epochs}",
        )

        # Unfreeze BERT after stage2_frozen_bert_epochs
        if epoch == stage2_frozen_bert_epochs:
            print(
                f"\n[Stage 2] Unfreezing {'LoRA adapters' if use_lora else 'BERT encoder'} at epoch {epoch + 1}"
            )
            if use_lora:
                for name, param in model.model.named_parameters():
                    if "lora" in name:
                        param.requires_grad = True
            else:
                for p in model.model.parameters():
                    p.requires_grad = True
            optimizer = AdamW(
                [
                    {"params": model.model.parameters(), "lr": lr_bert},
                    {"params": model.feature_ff.parameters(), "lr": lr_custom},
                    {"params": model.regressor.parameters(), "lr": lr_custom},
                ],
                weight_decay=weight_decay,
            )
            remaining_steps = len(sm_train_loader) * (stage2_epochs - epoch)
            scheduler = get_linear_schedule_with_warmup(
                optimizer,
                num_warmup_steps=int(remaining_steps * optimizer_warmup),
                num_training_steps=remaining_steps,
            )

        for step, batch in enumerate(progress_bar):
            optimizer.zero_grad()
            outputs = model(
                input_ids=batch["input_ids"].to(device),
                attention_mask=batch["attention_mask"].to(device),
                features=batch["features"].to(device),
                labels=batch["labels"].to(device),
            )
            loss = outputs.loss
            if torch.isnan(loss) or torch.isinf(loss):
                continue

            loss_value = loss.item()
            metrics["train_losses"].append(loss_value)
            metrics.update()

            loss.backward()

            # Gradient flow check at intervals
            if step == 0 or (step + 1) % 10 == 0:
                check_gradient_flow(model, step + 1, epoch + 1)

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad()
            scheduler.step()

            progress_bar.set_postfix({"loss": loss_value})
            del loss, outputs

        avg_train_loss = (
            float(np.mean(epoch_train_losses)) if epoch_train_losses else float("nan")
        )

        # Validation if val_df provided
        if val_loader is not None:
            model.eval()
            epoch_val_losses = []
            with torch.no_grad():
                for batch in tqdm(val_loader, desc="Validating", leave=False):
                    outputs = model(
                        input_ids=batch["input_ids"].to(device),
                        attention_mask=batch["attention_mask"].to(device),
                        features=batch["features"].to(device),
                        labels=batch["labels"].to(device),
                    )
                    epoch_val_losses.append(outputs.loss.item())
                    del outputs

            avg_val_loss = np.mean(epoch_val_losses)
            metrics["val_losses"].append(avg_val_loss)
            metrics.update()

            print(
                f"\nEpoch {epoch + 1}: Train Loss = {avg_train_loss:.4f}, Val Loss = {avg_val_loss:.4f}"
            )

            # Early stopping check
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                patience_counter = 0
                best_model_state = model.state_dict()
                print(f"✓ New best validation loss: {best_val_loss:.4f}")
            else:
                patience_counter += 1
                print(
                    f"✗ No improvement. Patience: {patience_counter}/{EARLY_STOPPING_PATIENCE}"
                )
                if patience_counter >= EARLY_STOPPING_PATIENCE:
                    print(f"Early stopping triggered at epoch {epoch + 1}")
                    break

            model.train()
        else:
            # No validation - just print training loss
            print(f"Epoch {epoch + 1} avg loss: {avg_train_loss:.4f}")

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        gc.collect()

    # Final evaluation
    model.eval()
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        print("Loaded best model state for final evaluation")
    eval_df = val_df if val_df is not None else train_df
    eval_dataset = CustomDataset(eval_df, tokenizer)
    eval_loader = DataLoader(eval_dataset, batch_size=BATCH_SIZE)

    y_true = []
    y_pred = []
    with torch.no_grad():
        for batch in tqdm(eval_loader, desc="Final evaluation"):
            outputs = model(
                input_ids=batch["input_ids"].to(device),
                attention_mask=batch["attention_mask"].to(device),
                features=batch["features"].to(device),
                labels=batch["labels"].to(device),
            )
            preds = outputs.logits.squeeze()
            y_true.extend(batch["labels"].cpu().numpy())
            y_pred.extend(preds.cpu().numpy())
            del outputs

    m = calculate_metrics(np.array(y_true), np.array(y_pred))
    metrics.update(**m)

    return {
        "model": model,
        "scaler": scaler,
        **metrics,
    }
