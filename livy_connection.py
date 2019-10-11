"""
PySpark Decision Tree Classification Example.
"""
from __future__ import print_function

import os
import json
import time
import requests
import textwrap
from argparse import ArgumentParser


def _get_livy_host(livy_url):
    idx = livy_url.find('://')
    if idx >= 0:
        livy_url = livy_url[(idx + 3):]
    return 'http://' + livy_url


def polling_intervals(start, rest, max_duration=None):
    def _intervals():
        yield from start
        while True:
            yield rest

    cumulative = 0.0
    for interval in _intervals():
        cumulative += interval
        if max_duration is not None and cumulative > max_duration:
            break
        yield interval


def _wait_session_start(url, wait_for_state, headers):
    print(url)
    intervals = polling_intervals([0.1, 0.2, 0.3, 0.5], 1.0)
    while requests.get(url, headers=headers).json()['state'] != wait_for_state:
        time.sleep(next(intervals))


if __name__ == "__main__":
    # parser = ArgumentParser()
    # parser.add_argument("--max_depth", dest="max_depth", help="max_depth", default=2, type=int)
    # parser.add_argument("--max_bins", dest="max_bins", help="max_bins", default=32, type=int)
    # args = parser.parse_args()
    # current_file = os.path.basename(__file__)

    livy_server_url = 'http://172.16.80.22:8998'

    # livy_url = _get_livy_host(livy_server_url)
    kind = {'kind': 'pyspark'}
    headers = {'Content-Type': 'application/json'}

    file_path = './train.py'
    code = open(file_path, 'r').read()
    data = {'code': textwrap.dedent(code)}

    # request_session = requests.Session()
    # livy_connection = request_session.request('POST', livy_server_url).raise_for_status()
    livy_session = requests.post(livy_server_url + '/sessions', json=kind).json()
    print(livy_session)
    _wait_session_start('{}/sessions/{}'.format(livy_server_url, livy_session['id']), 'idle', headers)
    print(livy_session)
    # livy_session = request_session.request('POST', '/sessions', json=kind).raise_for_status()
    r = requests.post(livy_server_url + '/sessions/{}/statements'.format(livy_session['id']),
                      data=json.dumps(data),
                      headers=headers)
    _wait_session_start('{}/sessions/{}'.format(livy_server_url, livy_session['id']), 'available', headers)
