"""
    This script isn't a part of the CLI tool, but a helper script for image downloading from ajapaik.ee API.
"""
import json

import urllib.request
from pathlib import Path


def get_ajapaik_image_locations(
        location='https://ajapaik.ee/api/v1/album/state/?id=22197&limit=100',
        dimension='320') -> dict:
    with urllib.request.urlopen(location) as stream:
        response = json.loads(stream.read().decode('utf-8'))

    if response is None or 'photos' not in response.keys():
        raise ValueError('Response does not have necessary values')

    image_locations = {}
    for photo in response['photos']:
        location = photo['image'].replace('[DIM]', dimension)
        id = photo['id']
        image_locations[id] = {
            'location': location,
            'id': id
        }

    return image_locations


def download_locations(links: dict, destination_dir: Path = Path('../../../')):
    destination_dir.mkdir(parents=True, exist_ok=True)
    for id in links:
        output_path = destination_dir / (str(id) + '.jpg')
        urllib.request.urlretrieve(links[id]['location'], output_path)


if __name__ == '__main__':
    links = get_ajapaik_image_locations()
    download_locations(links, destination_dir=Path('../../data/set1'))
