import torch
import pandas as pd
import torch.nn.functional as F
from constants import BERT_CLS_TOKEN, BERT_SEP_TOKEN, BERT_PAD_TOKEN

# Modify the gradient modules extraction code.
def compute_grads(model, x_embeds, y_labels, grad_type='all_layers', layer_idxs=None, create_graph=False):
    num_layers = model.config.num_hidden_layers
    if layer_idxs is None:
        layer_idxs = list(range(num_layers))
    elif not all(0 <= idx < num_layers for idx in layer_idxs):
        raise ValueError("Layer indexes must be within the valid range of model layers.")
        
    outs = model(inputs_embeds=x_embeds, labels=y_labels)

    # Define a mapping from grad_type to corresponding parameters.
    param_selectors = {
        'all_layers': lambda name, layer_idxs: True,
        'encoder': lambda name, layer_idxs: 'encoder' in name,
        'layer_encoder': lambda name, layer_idxs: any(f'encoder.layer.{idx}.' in name for idx in layer_idxs),
        'attn_query': lambda name, layer_idxs: any(
            f'encoder.layer.{idx}.' in name for idx in layer_idxs) and 'attention.self.query.weight' in name,
        'attn_key': lambda name, layer_idxs: any(
            f'encoder.layer.{idx}.' in name for idx in layer_idxs) and 'attention.self.key.weight' in name,
        'attn_value': lambda name, layer_idxs: any(
            f'encoder.layer.{idx}.' in name for idx in layer_idxs) and 'attention.self.value.weight' in name,
        'attn_qkv': lambda name, layer_idxs: any(
            f'encoder.layer.{idx}.' in name for idx in layer_idxs) and (
                'attention.self.query.weight' in name or
                'attention.self.key.weight' in name or
                'attention.self.value.weight' in name
            ),
        'attn_output': lambda name, layer_idxs: any(
            f'encoder.layer.{idx}.' in name for idx in layer_idxs) and 'attention.output.dense.weight' in name,
        'ffn_fc': lambda name, layer_idxs: any(
            f'encoder.layer.{idx}.' in name for idx in layer_idxs) and 'intermediate.dense.weight' in name,
        'ffn_output': lambda name, layer_idxs: any(
            f'encoder.layer.{idx}.' in name for idx in layer_idxs) and 'output.dense.weight' in name,
        'word_emb': lambda name, layer_idxs: 'word_embeddings' in name or 'embedding' in name,
    }

    if grad_type not in param_selectors:
        valid_types = ', '.join(param_selectors.keys())
        raise ValueError(f"Invalid grad_type '{grad_type}'. Must be one of: {valid_types}")

    # Select parameters based on the grad_type.
    selector = param_selectors[grad_type]
    params_to_use = [param for name, param in model.named_parameters() if selector(name, layer_idxs)]

    # Compute gradients.
    grads = torch.autograd.grad(outs.loss, params_to_use, create_graph=create_graph, allow_unused=True)

    return grads if isinstance(grads, tuple) else (grads,)

def grad_dist(grads1, grads2, args):
    if not grads1 or not grads2:
        raise ValueError("grads1 and grads2 cannot be empty")

    device = grads1[0].device if grads1[0].is_cuda else 'cpu'
    ret = torch.tensor(0.0, device=device)
    
    n_g = 0
    for g1, g2 in zip(grads1, grads2):
        if (g1 is not None) and (g2 is not None):
            if args.loss == 'cos':
                ret += 1.0 - (g1 * g2).sum() / (g1.view(-1).norm(p=2) * g2.view(-1).norm(p=2))
            elif args.loss == 'dlg':
                ret += (g1 - g2).square().sum()
            elif args.loss == 'tag':
                ret += (g1 - g2).square().sum() + args.tag_factor * torch.abs(g1 - g2).sum()
            else:
                assert False, f"Unsupported loss type: {args.loss}"
            n_g += 1

    if n_g == 0:
        raise ValueError("No gradients were processed")

    if args.loss == 'cos':
        ret /= n_g
    return ret

def get_closest_tokens(inputs_embeds, unused_tokens, embeddings_weight, metric='cos'):
    embeddings_weight = embeddings_weight.repeat(inputs_embeds.shape[0], 1, 1)
    if metric == 'l2':
        d = torch.cdist(inputs_embeds, embeddings_weight, p=2)
    elif metric == 'cos':
        dp = torch.bmm(inputs_embeds, embeddings_weight.transpose(1, 2))
        norm1 = inputs_embeds.norm(p=2, dim=2).unsqueeze(2)
        norm2 = embeddings_weight.norm(p=2, dim=2).unsqueeze(1)
        d = -dp / (norm1 * norm2)
    else:
        assert False

    d[:, :, unused_tokens] = 1e9
    return d, d.min(dim=2)[1]


def get_reconstruction_loss(model, x_embeds, y_labels, true_grads, args, create_graph=False):
    grads = compute_grads(model, x_embeds, y_labels, grad_type=args.grad_type, layer_idxs=args.attack_layer, create_graph=create_graph)
    return grad_dist(true_grads, grads, args)


def get_perplexity(gpt2, x_embeds, bert_embeddings_weight, gpt2_embeddings_weight, c=0.1):
    gpt2_embeddings_weight = gpt2_embeddings_weight.repeat(x_embeds.shape[0], 1, 1)

    # Get alphas on BERT embeddings --> transfer to GPT-2
    alpha, _ = get_closest_tokens(x_embeds, bert_embeddings_weight)
    # alpha = torch.cdist(x_embeds[:, :-1, :], bert_embeddings_weight, p=2)
    alpha = F.softmax(-alpha/c, dim=2)
    gpt2_embeds = alpha.bmm(gpt2_embeddings_weight)

    # Pass through GPT-2 and get average perplexity
    out_gpt2 = gpt2(inputs_embeds=gpt2_embeds)
    log_probs = out_gpt2.logits.log_softmax(dim=2)
    fuzzy_perplexity = -(log_probs[:, :-1, :] * alpha[:, 1:, :]).sum(dim=2).mean(dim=1).sum()
    return fuzzy_perplexity


def fix_special_tokens(x_embeds, bert_embeddings_weight, pads):
    x_embeds.data[:, 0] = bert_embeddings_weight[BERT_CLS_TOKEN]
    if pads is not None:
        for sen_id in range(x_embeds.shape[0]):
            x_embeds.data[sen_id, pads[sen_id]:] = bert_embeddings_weight[BERT_PAD_TOKEN]
            x_embeds.data[sen_id, pads[sen_id]-1] = bert_embeddings_weight[BERT_SEP_TOKEN]
    elif x_embeds.shape[0] == 1:
        x_embeds.data[:, -1] = bert_embeddings_weight[BERT_SEP_TOKEN]
    return x_embeds


def remove_padding(tokenizer, ids):
    for i in range(ids.shape[0] - 1, -1, -1):
        if ids[i] == BERT_SEP_TOKEN:
            ids = ids[:i+1]
            break
    return tokenizer.decode(ids)

def transform_dataset(dataset):
    text_data = []
    label_data = []

    for sample, label in dataset:
        labels = label.tolist()[0]
        assert len(sample) == len(labels)
        text_data.extend(sample)
        label_data.extend(labels)

    train_df = pd.DataFrame({'text': text_data, 'label': label_data})

    from datasets import Dataset
    from torch.utils.data import DataLoader
    from transformers import AutoTokenizer, DataCollatorWithPadding

    train_dataset = Dataset.from_pandas(train_df)
    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased', use_fast=True)
    tokenizer.model_max_length = 128

    def tokenize_function(examples):
        return tokenizer(examples["text"], truncation=True)

    tokenized_datasets = train_dataset.map(tokenize_function, batched=True)
    tokenized_datasets = tokenized_datasets.remove_columns(["text"])
    tokenized_datasets = tokenized_datasets.rename_column("label", "labels")
    tokenized_datasets.set_format(type='torch')
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer, return_tensors="pt")

    train_loader = DataLoader(
        tokenized_datasets,
        shuffle=False,
        batch_size=8,
        collate_fn=data_collator
    )

    return train_loader
