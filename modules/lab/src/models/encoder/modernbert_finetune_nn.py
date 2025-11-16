from transformers import ModernBertPreTrainedModel, ModernBertModel
from transformers.modeling_outputs import SequenceClassifierOutput
import torch.nn as nn
import torch


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
