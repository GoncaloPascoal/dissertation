import random
from argparse import ArgumentParser
from datetime import datetime, timedelta
from time import sleep
from typing import cast

from qiskit.providers.ibmq import IBMQBackend
from qiskit_ibm_provider import IBMProvider
from tqdm.rich import tqdm

from scripts.ibmq.current_calibration import save_properties


def main():
    base_dir = 'data/calibration'

    parser = ArgumentParser('past_calibration', description='Obtain past IBMQ device calibration data')

    parser.add_argument('backend', help='backend name')
    parser.add_argument('--start-date', metavar='D', help='start date in ISO format', default='2023-08-01')
    parser.add_argument('--max-requests', metavar='N', type=int, default=365, help='max. number of requests')

    args = parser.parse_args()

    provider = IBMProvider()
    backend = cast(IBMQBackend, provider.get_backend(args.backend))
    config = backend.configuration()

    online_date = cast(datetime, config.online_date) + timedelta(hours=23)
    date = datetime.fromisoformat(args.start_date).replace(tzinfo=datetime.now().astimezone().tzinfo)
    one_day = timedelta(days=1)

    for _ in tqdm(range(args.max_requests)):  # type: ignore
        data = backend.properties(datetime=date)

        if data is not None:
            save_properties(base_dir, args.backend, data.to_dict())

        date -= one_day
        if date <= online_date:
            break

        # Avoid getting rate limited
        sleep(random.uniform(0.5, 1.0))


if __name__ == '__main__':
    main()
