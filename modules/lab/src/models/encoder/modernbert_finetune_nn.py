from transformers import ModernBertPreTrainedModel, ModernBertModel
from transformers.modeling_outputs import SequenceClassifierOutput
import torch.nn as nn
import torch
import numpy as np


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

        self._init_custom_weights()
        self.post_init()

        for name, param in self.named_parameters():
            if "encoder" in name:
                param.requires_grad = True

        print(
            f"Trainable parameters: {sum(p.numel() for p in self.parameters() if p.requires_grad):,}"
        )
        print(f"Total parameters: {sum(p.numel() for p in self.parameters()):,}")

    def _init_custom_weights(self):
        for module in [self.feature_ff, self.regressor]:
            for m in module.modules():
                if isinstance(m, nn.Linear):
                    nn.init.normal_(m.weight, mean=0.0, std=0.1)
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        features=None,
        labels=None,
        **kwargs,
    ):
        print(
            f"Labels range: [{labels.min():.2f}, {labels.max():.2f}], mean: {labels.mean():.2f}"
        )
        print(f"Features range: [{features.min():.2f}, {features.max():.2f}]")
        print(f"Features std: {features.std():.2f}")

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
        print(
            f"Logits range: [{logits.min():.2f}, {logits.max():.2f}], mean: {logits.mean():.2f}"
        )

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
        "bert_embeddings": [],
        "bert_encoder": [],
        "bert_pooler": [],
        "feature_ff": [],
        "regressor": [],
    }

    for name, param in model.named_parameters():
        if param.grad is not None:
            grad_norm = param.grad.norm().item()
            if "embeddings" in name:
                grad_stats["bert_embeddings"].append(grad_norm)
            elif "encoder" in name:
                grad_stats["bert_encoder"].append(grad_norm)
            elif "pooler" in name:
                grad_stats["bert_pooler"].append(grad_norm)
            elif "feature_ff" in name:
                grad_stats["feature_ff"].append(grad_norm)
            elif "regressor" in name:
                grad_stats["regressor"].append(grad_norm)

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
