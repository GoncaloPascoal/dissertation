
import json
import os
import random
from datetime import datetime
from time import sleep
from typing import Any, Final

import requests
from rich import print

IBMQ_ENDPOINT: Final = 'https://api.quantum-computing.ibm.com'


def fetch_properties(backend_name: str) -> dict[str, Any]:
    url = f'{IBMQ_ENDPOINT}/api/Backends/{backend_name}/properties'

    response = requests.get(url)

    if response.status_code == 200:
        return json.loads(response.text)
    else:
        raise RuntimeError(f'Request for backend {backend_name} failed with status code {response.status_code}')


def save_properties(base_dir: str, backend_name: str, data: dict[str, Any]):
    update_date = data['last_update_date']
    update_date = update_date if isinstance(update_date, datetime) else datetime.fromisoformat(update_date)

    file_name = update_date.isoformat(sep='T')

    path = os.path.join(base_dir, backend_name, f'{file_name}.json')
    os.makedirs(os.path.dirname(path), exist_ok=True)

    with open(path, 'w') as f:
        json.dump(data, f, default=str)


def main():
    base_dir = os.path.join('data', 'calibration')
    backends = ['ibmq_guadalupe', 'ibm_hanoi', 'ibm_cairo', 'ibmq_mumbai', 'ibmq_kolkata', 'ibm_algiers',
                'ibmq_manila', 'ibmq_belem', 'ibm_nairobi']

    for backend in backends:
        data = fetch_properties(backend)
        print(f'Fetched {backend} data, last updated at {datetime.fromisoformat(data["last_update_date"])}')

        save_properties(base_dir, backend, data)

        # Avoid getting rate limited
        sleep(random.uniform(0.5, 1.0))


if __name__ == '__main__':
    main()
