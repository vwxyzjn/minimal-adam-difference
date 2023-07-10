import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from torch import optim
import torch.nn.functional as F
torch.set_printoptions(precision=10)
import os, pickle
import matplotlib.pyplot as plt
from huggingface_hub import hf_hub_download

def left_padding_to_right_padding(query, pad_id):
    return torch.tensor([
        [pad_id]*(row==pad_id).sum() + [x for x in row if x != pad_id]
        for row in query
    ])


def run(optimizer_fn, params_and_grads_file, comment=""):
    print("==================================================")
    
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    pad_id = tokenizer.pad_token_id

    # print("left padding")
    pretrained_model = AutoModelForCausalLM.from_pretrained("gpt2")
    with open(hf_hub_download(repo_id="vwxyzjn/lm-human-preferences-debug", filename="queries.npy", repo_type="dataset"), 'rb') as f:
        query = np.load(f)[:64,:]
        query[query == 50259] = pad_id
    with open(hf_hub_download(repo_id="vwxyzjn/lm-human-preferences-debug", filename='responses.npy', repo_type="dataset"), 'rb') as f:
        response = np.load(f)[:64,:]
    with open(hf_hub_download(repo_id="vwxyzjn/lm-human-preferences-debug", filename='rewards.npy', repo_type="dataset"), 'rb') as f:
        rewards = np.load(f)[:64,:]
    query = torch.tensor(query)
    # query = query[:,53:]
    # print("query", query)
    query = left_padding_to_right_padding(query, pad_id)
    response = torch.tensor(response).long()
    rewards = torch.tensor(rewards)

    temperature = 1.0
    values = torch.zeros_like(rewards)

    optimizer = optimizer_fn(pretrained_model)
    print(f"working with {optimizer.__class__.__name__}:", comment)
    with torch.no_grad():
        context_length = query.shape[1]
        query_response = torch.cat((query, response), 1)
        attention_mask = query_response != pad_id
        # position_ids = ((attention_mask.cumsum(1) - 1)).clamp(min=0)
        position_ids = attention_mask.cumsum(1) - attention_mask.long()
        output = pretrained_model(
            input_ids=query_response,
            attention_mask=attention_mask,
            position_ids=position_ids,
            return_dict=True,
            output_hidden_states=True,
        )
        logits = output.logits[:,context_length-1:-1]
        logits /= temperature
        # print("logits", logits)
        all_logprobs = F.log_softmax(logits, dim=-1)
        logprobs = torch.gather(all_logprobs, 2, response.unsqueeze(-1)).squeeze(-1)
        # print("logprobs", logprobs)
    def whiten(values, shift_mean=True):
        mean, var = torch.mean(values), torch.var(values, unbiased=False)
        whitened = (values - mean) * torch.rsqrt(var + 1e-8)
        if not shift_mean:
            whitened += mean
        return whitened

    for epoch in range(4):
        rewards = whiten(rewards, shift_mean=False)
        # print("rewards", rewards)
        gamma = 1
        lam = 0.95
        cliprange = 0.2
        with torch.no_grad():
            lastgaelam = 0
            advantages_reversed = []
            gen_length = response.shape[1]
            for t in reversed(range(gen_length)):
                nextvalues = values[:, t + 1] if t < gen_length - 1 else 0.0
                delta = rewards[:, t] + gamma * nextvalues - values[:, t]
                lastgaelam = delta + gamma * lam * lastgaelam
                advantages_reversed.append(lastgaelam)
            advantages = torch.stack(advantages_reversed[::-1], axis=1)
            returns = advantages + values
            advantages = whiten(advantages)

        # print("advantages", advantages)
        # print("returns", returns)
        output = pretrained_model(
            input_ids=query_response,
            attention_mask=attention_mask,
            position_ids=position_ids,
            return_dict=True,
            output_hidden_states=True,
        )
        logits = output.logits[:,context_length-1:-1]
        logits /= temperature
        new_all_logprobs = F.log_softmax(logits, dim=-1)
        new_logprobs = torch.gather(new_all_logprobs, 2, response.unsqueeze(-1)).squeeze(-1)
        # print("new_logprobs", new_logprobs)
        logprobs_diff = new_logprobs - logprobs
        ratio = torch.exp(logprobs_diff)
        print(f"epoch={epoch}, logprobs_diff mean={logprobs_diff.detach().numpy().mean()}")
        print(f"epoch={epoch}, ratio mean={ratio.detach().numpy().mean()}")
        print(f"epoch={epoch}, ratio var={ratio.detach().numpy().var()}")
        print(f"epoch={epoch}, ratio max={ratio.detach().numpy().max()}")
        print(f"epoch={epoch}, ratio min={ratio.detach().numpy().min()}")
        pg_losses = -advantages * ratio
        pg_losses2 = -advantages * torch.clamp(ratio, 1.0 - cliprange, 1.0 + cliprange)
        # print("torch.max(pg_losses, pg_losses2)", torch.max(pg_losses, pg_losses2))
        pg_loss = torch.max(pg_losses, pg_losses2).mean()
        pg_clipfrac = (pg_losses2 > pg_losses).float().mean()
        loss = pg_loss.mean()
        optimizer.zero_grad()
        loss.backward()

        
        with open(params_and_grads_file, "rb") as f:
            params_and_gradss = pickle.load(f)

        named_params = list(pretrained_model.named_parameters())
        new_named_params = [named_params[1], named_params[0]] + named_params[2:]
        grad_diffs = {}
        param_diffs = {}
        i = 0
        if epoch < len(params_and_gradss):
            for oname, (name, param) in zip(params_and_gradss[epoch], new_named_params):
                oparam, ograd = params_and_gradss[epoch][oname]
                if param.requires_grad:
                    # print(f"param {name, oname}")
                    param_diff = np.abs(oparam - param.detach().numpy()).mean()
                    grad_diff = np.abs(ograd - param.grad.detach().numpy()).mean()

                    grad_diffs[f"{i} {name} ({ograd.shape})"] = grad_diff
                    param_diffs[f"{i} {name} ({oparam.shape})"] = param_diff
                    # print(f"param_diff={param_diff} grad_diff={grad_diff}")
                    i += 1

            # plot grad_diffs
            os.makedirs(f"diffs/{optimizer.__class__.__name__}", exist_ok=True)
            fig, ax = plt.subplots(figsize=(10, 20))
            plt.barh(list(grad_diffs.keys()), list(grad_diffs.values()))
            plt.title(f"Epoch: {epoch}, Optimizer: {optimizer.__class__.__name__}, Pytorch's gradient vs OAI's tensorflow's gradient")
            plt.tight_layout()
            fig.savefig(f"diffs/{optimizer.__class__.__name__}/grad_diffs_{epoch}.png")
            plt.close()
            # plot param_diffs
            fig, ax = plt.subplots(figsize=(10, 20))
            plt.barh(list(param_diffs.keys()), list(param_diffs.values()))
            plt.tight_layout()
            plt.title(f"Epoch: {epoch}, Optimizer: {optimizer.__class__.__name__}, Pytorch's param vs OAI's tensorflow's param")
            fig.savefig(f"diffs/{optimizer.__class__.__name__}/param_diffs_{epoch}.png")
            plt.close()

        optimizer.step()
        approxkl = .5 * ((logprobs_diff) ** 2).mean()
        print("approxkl", approxkl.item(), "pg_loss", pg_loss.item(), "pg_clipfrac", pg_clipfrac.item())


run(
    optimizer_fn=lambda model: optim.Adam(model.parameters(), lr=0.00001, eps=1e-5, betas=(0.9, 0.999)), 
    comment="Adam with eps=1e-5, betas=(0.9, 0.999)",
    params_and_grads_file=hf_hub_download(
        repo_id="vwxyzjn/lm-human-preferences-debug",
        filename="params_and_grads.pkl",
        repo_type="dataset"
    )
)
run(
    optimizer_fn=lambda model: optim.SGD(model.parameters(), lr=0.00001),
    params_and_grads_file=hf_hub_download(
        repo_id="vwxyzjn/lm-human-preferences-debug",
        filename="params_and_grads_sgd.pkl",
        repo_type="dataset"
    )
)
