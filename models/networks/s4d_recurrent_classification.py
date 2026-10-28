import torch
import torch.nn as nn

from models.backbones.s4d_recurrent_backbone import S4DRecurrentBackbone

class Model(nn.Module):

    def __init__(
        self,
        d_input,
        d_output=10,
        d_model=256,
        n_layers=4,
        dropout=0.2,
        prenorm=False,
        lr=0.001,
        kernel_type='recurrent',
        training=False,
        early_stop_strategy=None,
        debug = True
    ):
        super().__init__()

        self.d_output = d_output
        self.d_model = d_model
        self.debug = debug

        self.prenorm = prenorm

        # Backbone
        self.backbone = S4DRecurrentBackbone(
            d_input=d_input,
            d_model=d_model,
            n_layers=n_layers,
            dropout=dropout,
            prenorm=prenorm,
            lr=lr,
            kernel_type=kernel_type,
            training=training,
        )

        # Linear decoder
        self.decoder = nn.Linear(d_model, d_output)

        # Early stop strategy
        self.early_stop_strategy = early_stop_strategy # TODO

        self._currrent_step = 0
        self._total_steps = 0
        self._pool_sum = None # (B, d_model), sum of all features
        self._active_mask = None # (B,) Bool, True for mask samples for early stopped samples
        self._stop_step = None # (B,) Int, step at which each sample was stopped
        self._final_logits = None # (B, d_output), logits for each sample
        self._confidence_history = [] # List of confidence for each step

    def prepare_for_inference(self):
        """Prepare the model for inference by caching the discrete parameters of each S4D layer."""
        self.backbone.prepare_for_inference()

    def reset_state(self, batch_size, device, total_steps=0):
        self.backbone.reset_state(batch_size, device)
        # reset self.early_stop_strategy

        self._currrent_step = 0
        self._total_steps = total_steps
        self._pool_sum = torch.zeros(batch_size, self.d_model, device=device)
        self._active_mask = torch.zeros(batch_size, dtype=torch.bool, device=device)
        self._stop_step = torch.full((batch_size,), -1, dtype=torch.int, device=device)
        self._final_logits = torch.zeros(batch_size, self.d_output, device=device)
        self._confidence_history = []

    def forward_step(self, x_t):
        """
            input: x_t: (B, d_input)
            output:
                logits: (B, d_output)
                Dict(str, any) if Debug mode on
                    prediction: (B,)
                    confidence: (B,)
                    should_stop: (B,) Bool
                    all_stop: Bool
                    current_step: Int
                    info: others
        """
        B = x_t.shape[0]
        device = x_t.device
        info = None

        # Check reset state
        if self._active_mask is None or self._active_mask.shape[0] != B:
            self.reset_state(B, device)

        # Forward step and pooling
        torch.cuda.nvtx.range_push("pooling")
        self._pool_sum.add_(self.backbone.forward_step(x_t)) # (B, d_model) -> (B, d_model)
        torch.cuda.nvtx.range_pop()
        
        # decoder
        torch.cuda.nvtx.range_push("decoder")
        logits = self.decoder(self._pool_sum / (self._currrent_step + 1)) # (B, d_output)
        torch.cuda.nvtx.range_pop()

        # Early stop strategy
        if self.early_stop_strategy is not None:
            should_stop, info = self.early_stop_strategy.should_stop(
                logits, self._currrent_step, self._total_steps
            )

            # Update active mask
            newly_stopped = should_stop & ~self._active_mask
            self._final_logits[~self._active_mask] = logits[~self._active_mask]
            self._stop_step[newly_stopped] = self._currrent_step
            self._active_mask = self._active_mask | should_stop
        else:
            should_stop = torch.zeros(B, dtype=torch.bool, device=device)
            info = None


        self._currrent_step += 1

        all_stop = self._active_mask.all().item()

        
        # Update confidence history
        if self.debug:
            probs = torch.softmax(logits, dim=-1)
            confidence, prediction = probs.max(dim=-1)
            self._confidence_history.append(confidence.detach().cpu())
            return logits, all_stop, {
                'prediction': prediction,
                'confidence': confidence,
                'should_stop': should_stop,
                'current_step': self._currrent_step - 1,
                'info': info,
            }
        else:
            return logits, all_stop, None

    def forward(self, x, early_stop=True):
        """
            Full forward pass.
            input: x: (B, L, d_input)
            output: Dict(str, any)
        """
        B, L, D = x.shape
        device = x.device

        # Reset state
        self.reset_state(B, device, total_steps=L)

        # Forward step
        for x_t in x.unbind(dim=1):
            logits, all_stop, results = self.forward_step(x_t)

            if early_stop and all_stop:
                break

        # Caculate final logits
        if not self._active_mask.all():
            self._final_logits[~self._active_mask] = logits[~self._active_mask]
            self._stop_step[~self._active_mask] = self._currrent_step - 1

        # Final prediction
        if self.debug:
            probs = torch.softmax(self._final_logits, dim=-1)
            confidence, prediction = probs.max(dim=-1)

            return logits, {
                'prediction': prediction,
                'confidence': confidence,
                'stop_step': self._stop_step,
                'confidence_history': self._confidence_history
            }
        else:
            return logits, None

    def get_statistics(self):
        if self._stop_step is None:
            return {}

        stopped_mask = self._stop_step >= 0
        avg_stop_step = self._stop_step[stopped_mask].float().mean().item() if stopped_mask.any() else -1
        early_stop_ratio = stopped_mask.float().mean().item()

        return {
            'avg_stop_step': avg_stop_step,
            'early_stop_ratio': early_stop_ratio,
            'stop_step': self._stop_step,
            'confidence_history': self._confidence_history
        }