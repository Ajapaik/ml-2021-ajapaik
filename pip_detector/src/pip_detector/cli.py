from pathlib import Path

import click

from .model import Model


@click.group()
def main():
    pass


@main.command()
@click.option('-p', '--dataset_path', default=None, required=True, type=Path)
@click.option('-d', '--device', default='gpu', type=str)
@click.pass_context
def train(ctx, dataset_path, device):
    model = Model(dataset_path, device=device)
    model.train()


@main.command()
@click.option('-m', '--model_path', default=Path("output"), type=Path)
@click.option('-p', '--dataset_path', default=None, required=True, type=Path)
@click.option('-i', '--input_path', default=None, required=True, type=Path)
@click.option('-t', '--threshold', default=0.5, type=float)
@click.option('-d', '--device', default=None, type=str)
@click.pass_context
def detect(ctx, model_path, dataset_path, input_path, threshold, device):
    model = Model(dataset_path, device=device, output_path=model_path)
    model.detect(input_path, threshold=threshold)


if __name__ == "__main__":
    main()
