import argparse
import json
import os

import numpy as np
import torch

import irec.utils
from irec.dataloader import BaseDataloader
from irec.dataset import BaseDataset
from irec.metric import BaseMetric, StatefullMetric
from irec.models import BaseModel
from irec.utils import DEVICE, fix_random_seed


def resolve_checkpoint_path(checkpoint):
    candidates = [
        checkpoint,
        f'{checkpoint}.pth',
        os.path.join('checkpoints', checkpoint),
        os.path.join('checkpoints', f'{checkpoint}.pth'),
    ]
    for path in candidates:
        if os.path.exists(path):
            return path
    raise FileNotFoundError(
        'Checkpoint not found. Tried: {}'.format(', '.join(candidates)),
    )


def get_inference_config(config, split):
    for callback_config in config['callback']['callbacks']:
        if callback_config['type'] == split:
            return callback_config
    raise ValueError('No `{}` callback found in config'.format(split))


def evaluate(model, dataloader, metric_configs, pred_prefix, labels_prefix, meta):
    metrics = {
        name: BaseMetric.create_from_config(metric_config, **meta)
        for name, metric_config in metric_configs.items()
    }
    running = {name: [] for name in metrics}

    model.eval()
    with torch.no_grad():
        for batch in dataloader:
            for key, value in batch.items():
                batch[key] = value.to(DEVICE)

            batch[pred_prefix] = model(batch)

            for key, value in batch.items():
                batch[key] = value.cpu()

            for name, metric in metrics.items():
                running[name].extend(
                    metric(
                        inputs=batch,
                        pred_prefix=pred_prefix,
                        labels_prefix=labels_prefix,
                    ),
                )

    result = {}
    for name, metric in metrics.items():
        values = running[name]
        if isinstance(metric, StatefullMetric):
            result[name] = metric.reduce(values)
        else:
            result[name] = float(np.mean(values))
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--params', required=True)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument(
        '--split',
        choices=['validation', 'eval', 'both'],
        default='eval',
    )
    args = parser.parse_args()

    fix_random_seed(42)
    with open(args.params) as f:
        config = json.load(f)

    dataset = BaseDataset.create_from_config(config['dataset'])
    _, validation_sampler, test_sampler = dataset.get_samplers()
    validation_dataloader = BaseDataloader.create_from_config(
        config['dataloader']['validation'],
        dataset=validation_sampler,
        **dataset.meta,
    )
    eval_dataloader = BaseDataloader.create_from_config(
        config['dataloader']['validation'],
        dataset=test_sampler,
        **dataset.meta,
    )

    model = BaseModel.create_from_config(config['model'], **dataset.meta).to(DEVICE)
    checkpoint_path = resolve_checkpoint_path(args.checkpoint)
    checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
    state_dict = checkpoint.get('model_state_dict', checkpoint)
    model.load_state_dict(state_dict)

    split_to_dataloader = {
        'validation': validation_dataloader,
        'eval': eval_dataloader,
    }
    splits = ['validation', 'eval'] if args.split == 'both' else [args.split]

    output = {}
    for split in splits:
        inference_config = get_inference_config(config, split)
        output[split] = evaluate(
            model=model,
            dataloader=split_to_dataloader[split],
            metric_configs=inference_config['metrics'],
            pred_prefix=inference_config['pred_prefix'],
            labels_prefix=inference_config['labels_prefix'],
            meta=dataset.meta,
        )

    print(json.dumps(output, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
