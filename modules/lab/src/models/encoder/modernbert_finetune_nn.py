import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from torch.utils.data import DataLoader
from transformers import ModernBertPreTrainedModel, ModernBertModel
from transformers.modeling_outputs import SequenceClassifierOutput
import torch.nn as nn
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
from tqdm.auto import tqdm
import torch
from sklearn.preprocessing import MinMaxScaler
from text2features import FeatureExtractorPipeline
from modules.lab.src.models.encoder.common import (
    run_study,
    device,
    run_cross_validation,
    CustomDataset,
)

BATCH_SIZE = 8


class ModernBertWithFeaturesTrainable(ModernBertPreTrainedModel):
    def __init__(
        self,
        config,
        feature_input_size,
        dropout_rate=0.25,
        feature_hidden_size=1024,
        norm_eps=1e-4,
    ):
        super().__init__(config)
        self.config = config
        self.model = ModernBertModel(config)

        self.feature_ff = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(feature_input_size, feature_hidden_size),
            nn.LayerNorm(feature_hidden_size, eps=norm_eps),
            nn.ReLU(),
        )

        bert_hidden_size = config.hidden_size

        self.regressor = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(bert_hidden_size + feature_hidden_size, 1),
        )

        self.loss_fct = nn.MSELoss()
        self.post_init()

    def _init_custom_weights(self, module):
        """Initializes the weights of the custom layers."""
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        features=None,
        labels=None,
        **kwargs,
    ):
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            **kwargs,
        )
        # Use the CLS token's representation (first token)
        cls_embedding = outputs.last_hidden_state[:, 0, :]

        feature_embedding = self.feature_ff(features)
        # if torch.isnan(feature_embedding).any():
        #     print("NaN found in feature_embedding")
        #     print("weight zero ratio:", (self.feature_ff[1].weight == 0).float().mean())
        concatenated_embedding = torch.cat((cls_embedding, feature_embedding), dim=1)

        logits = self.regressor(concatenated_embedding)
        loss = self.loss_fct(logits.squeeze(), labels)
        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


def objective(trial):
    # --- Hyperparameter Suggestions ---
    # num_epochs = trial.suggest_categorical("num_epochs", [3])
    # lr_bert = trial.suggest_float("lr_bert", 1e-6, 5e-5, log=True)
    # lr_custom = trial.suggest_float("lr_custom", 1e-7, 1e-5, log=True)
    # dropout_rate = trial.suggest_float("dropout_rate", 0.1, 0.5)
    # weight_decay = trial.suggest_float("weight_decay", 1e-3, 1e-1, log=True)
    # optimizer_warmup = trial.suggest_categorical("optimizer_warmup", [0.0, 0.1, 0.2])
    # feature_hidden_size = trial.suggest_categorical("feature_hidden_size", [512, 1024, 2048])

    # First try optuna with only two combinations overall of hyperparameters to test if it works
    num_epochs = trial.suggest_categorical("num_epochs", [3])
    lr_bert = trial.suggest_categorical("lr_bert", [5e-5, 1e-6])
    lr_custom = trial.suggest_categorical("lr_custom", [3e-6])
    dropout_rate = trial.suggest_categorical("dropout_rate", [0.25])
    weight_decay = trial.suggest_categorical("weight_decay", [1e-2])
    optimizer_warmup = trial.suggest_categorical("optimizer_warmup", [0.1])
    feature_hidden_size = trial.suggest_categorical("feature_hidden_size", [1024])

    def train_and_evaluate_fold(
        train_df: pd.DataFrame, val_df: pd.DataFrame, tokenizer
    ):
        scaler = MinMaxScaler()
        train_features_scaled = scaler.fit_transform(
            np.vstack(train_df["features"].values)
        )
        val_features_scaled = scaler.transform(np.vstack(val_df["features"].values))
        train_df["features"] = [f for f in np.nan_to_num(train_features_scaled)]
        val_df["features"] = [f for f in np.nan_to_num(val_features_scaled)]

        train_dataset = CustomDataset(train_df, tokenizer)
        val_dataset = CustomDataset(val_df, tokenizer)
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE)
        val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

        model = ModernBertWithFeaturesTrainable.from_pretrained(
            "answerdotai/ModernBERT-base",
            feature_input_size=FeatureExtractorPipeline.FEATURE_COUNT,
            dropout_rate=dropout_rate,
            feature_hidden_size=feature_hidden_size,
        )
        model.feature_ff.apply(model._init_custom_weights)
        model.regressor.apply(model._init_custom_weights)
        model.to(device)

        optimizer = AdamW(
            [
                {"params": model.model.parameters(), "lr": lr_bert},
                {"params": model.feature_ff.parameters(), "lr": lr_custom},
                {"params": model.regressor.parameters(), "lr": lr_custom},
            ],
            weight_decay=weight_decay,
        )

        total_steps = len(train_loader) * num_epochs
        warmup_steps = int(total_steps * optimizer_warmup)
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
        )

        # Train
        model.train()
        for epoch in range(num_epochs):
            progress_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}")
            for batch in progress_bar:
                optimizer.zero_grad()

                outputs = model(
                    input_ids=batch["input_ids"].to(device),
                    attention_mask=batch["attention_mask"].to(device),
                    features=batch["features"].to(device),
                    labels=batch["labels"].to(device),
                )

                loss = outputs.loss
                if torch.isnan(loss):
                    print("Loss is NaN, skipping backward pass.")
                    continue

                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                progress_bar.set_postfix({"loss": loss.item()})

                del loss, outputs
                if device.type == "cuda":
                    torch.cuda.empty_cache()

        # Validate
        model.eval()
        y_true_fold, y_pred_fold = [], []
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Evaluating"):
                outputs = model(
                    input_ids=batch["input_ids"].to(device),
                    attention_mask=batch["attention_mask"].to(device),
                    features=batch["features"].to(device),
                )
                predictions = outputs.logits.squeeze()

                y_true_fold.extend(batch["labels"].cpu().numpy())
                y_pred_fold.extend(predictions.cpu().numpy())

        mse = mean_squared_error(y_true_fold, y_pred_fold)
        return mse

    return run_cross_validation(
        trial=trial,
        train_and_eval_func=train_and_evaluate_fold,
        n_splits=3,
    )


def export_to_onnx(model, onnx_path):
    """Exports the model to ONNX format."""
    model.eval()
    model.to("cpu")

    # Create dummy inputs
    dummy_input_ids = torch.randint(
        0, model.config.vocab_size, (1, 160), dtype=torch.long
    )
    dummy_attention_mask = torch.ones(1, 160, dtype=torch.long)
    dummy_features = torch.randn(
        1, FeatureExtractorPipeline.FEATURE_COUNT, dtype=torch.float
    )

    # Define input and output names
    input_names = ["input_ids", "attention_mask", "features"]
    output_names = ["logits"]

    # Export the model
    torch.onnx.export(
        model,
        (dummy_input_ids, dummy_attention_mask, dummy_features),
        onnx_path,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes={
            "input_ids": {0: "batch_size"},
            "attention_mask": {0: "batch_size"},
            "features": {0: "batch_size"},
            "logits": {0: "batch_size"},
        },
        opset_version=14,
    )
    print(f"Model exported to {onnx_path}")


def train_final_model(best_params, tokenizer, train_df):
    num_epochs = best_params["num_epochs"]
    lr_bert = best_params["lr_bert"]
    lr_custom = best_params["lr_custom"]
    dropout_rate = best_params["dropout_rate"]
    weight_decay = best_params["weight_decay"]
    optimizer_warmup = best_params["optimizer_warmup"]
    feature_hidden_size = best_params["feature_hidden_size"]

    scaler = MinMaxScaler()
    train_features_scaled = scaler.fit_transform(np.vstack(train_df["features"].values))
    train_df["features"] = [f for f in np.nan_to_num(train_features_scaled)]

    train_dataset = CustomDataset(train_df, tokenizer)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE)

    model = ModernBertWithFeaturesTrainable.from_pretrained(
        "answerdotai/ModernBERT-base",
        feature_input_size=FeatureExtractorPipeline.FEATURE_COUNT,
        dropout_rate=dropout_rate,
        feature_hidden_size=feature_hidden_size,
    )
    model.feature_ff.apply(model._init_custom_weights)
    model.regressor.apply(model._init_custom_weights)
    model.to(device)

    optimizer = AdamW(
        [
            {"params": model.model.parameters(), "lr": lr_bert},
            {"params": model.feature_ff.parameters(), "lr": lr_custom},
            {"params": model.regressor.parameters(), "lr": lr_custom},
        ],
        weight_decay=weight_decay,
    )

    total_steps = len(train_loader) * num_epochs
    warmup_steps = int(total_steps * optimizer_warmup)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
    )

    # Train
    model.train()
    for epoch in range(num_epochs):
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}")
        for batch in progress_bar:
            optimizer.zero_grad()

            outputs = model(
                input_ids=batch["input_ids"].to(device),
                attention_mask=batch["attention_mask"].to(device),
                features=batch["features"].to(device),
                labels=batch["labels"].to(device),
            )

            loss = outputs.loss
            if torch.isnan(loss):
                print("Loss is NaN, skipping backward pass.")
                continue

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            progress_bar.set_postfix({"loss": loss.item()})

            del loss, outputs
            if device.type == "cuda":
                torch.cuda.empty_cache()
    return model


if __name__ == "__main__":
    search_space = {
        "num_epochs": [3],
        "lr_bert": [5e-5, 1e-6],
        "lr_custom": [3e-6],
        "dropout_rate": [0.1, 0.25],
        "weight_decay": [1e-2],
        "optimizer_warmup": [0.1],
        "feature_hidden_size": [512, 768, 1024],
    }
    run_study(
        objective_func=objective,
        study_name="modernbert_finetune_nn_regression",
        search_space=search_space,
        n_trials=None,
    )
