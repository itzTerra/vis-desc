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
        regressor_hidden_size=512,
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
            nn.Linear(bert_hidden_size + feature_hidden_size, regressor_hidden_size),
            nn.LayerNorm(regressor_hidden_size, eps=norm_eps),
            nn.GELU(),
            nn.Dropout(dropout_rate * 0.5),
            nn.Linear(regressor_hidden_size, 1),
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

        print(
            f"Trainable parameters: {sum(p.numel() for p in self.parameters() if p.requires_grad):,}"
        )
        print(f"Total parameters: {sum(p.numel() for p in self.parameters()):,}")

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
                        module.weight, gain=0.02
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
        print(
            f"Feature embeddings range: [{feature_embedding.min():.2f}, {feature_embedding.max():.2f}]"
        )
        concatenated_embedding = torch.cat((cls_embedding, feature_embedding), dim=1)

        concatenated_embedding = self.fusion_norm(concatenated_embedding)

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
