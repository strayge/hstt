import argparse
import asyncio
import curses
import platform
from collections import Counter
from dataclasses import dataclass
from functools import partial, wraps
from multiprocessing import Process, Queue
from queue import Empty
from statistics import mean, quantiles
from time import sleep, time
from typing import List, Optional

from aiohttp import ClientSession, ClientTimeout, TraceConfig

if platform.system() == 'Windows':
    # https://bugs.python.org/issue39232
    # https://github.com/aio-libs/aiohttp/issues/4324
    def silence_event_loop_closed(func):
        @wraps(func)
        def wrapper(self, *args, **kwargs):
            try:
                return func(self, *args, **kwargs)
            except RuntimeError as e:
                if str(e) != 'Event loop is closed':
                    raise

        return wrapper

    from asyncio.proactor_events import _ProactorBasePipeTransport  # noqa
    _ProactorBasePipeTransport.__del__ = silence_event_loop_closed(_ProactorBasePipeTransport.__del__)


def get_args():
    parser = argparse.ArgumentParser(description='HTTP stress testing tool (hstt)')
    parser.add_argument('-n', type=int, metavar='<num>', default=5, help='Total number of requests to perform')
    parser.add_argument('-c', type=int, metavar='<num>', default=1, help='Number of concurrent requests')
    parser.add_argument('-d', type=float, metavar='<sec>', default=30, help='Total duration limit')
    parser.add_argument('-t', type=float, metavar='<sec>', default=30, help='The timeout for each request.')
    parser.add_argument('-H', metavar='<header>', nargs='*', help='A request header to be sent')
    parser.add_argument('-b', metavar='<body>', help='A request body to be sent')
    parser.add_argument('-m', metavar='<method>', default='GET', help='An HTTP request method for each request')
    parser.add_argument('--debug', action='store_true', help='Run in debug mode')
    parser.add_argument('--insecure', action='store_true', help='Skip TLS verification')
    parser.add_argument('--chrome', action='store_true', help='Use Chrome User-Agent header')
    parser.add_argument('url', help='target URL')
    args = parser.parse_args()
    if not (args.url.startswith('http:') or args.url.startswith('https:')):
        print('Error: url should start with protocol')
        exit(1)
    return args


def worker_start(*args):
    try:
        loop = asyncio.new_event_loop()
        loop.run_until_complete(worker_loop(*args))
    except Exception:
        print(repr(e))


@dataclass
class Result:
    error: Optional[str] = None
    status: Optional[int] = None
    size: Optional[int] = None
    total_time: Optional[float] = None
    conn_time: Optional[float] = None
    ttfb_time: Optional[float] = None


async def worker_loop(args, tasks: Queue, results: Queue):
    try:
        async def trace_call(name, _, context, __):
            if name not in context.trace_request_ctx:
                context.trace_request_ctx[name] = time()
        trace = TraceConfig()
        trace.on_connection_create_start.append(partial(trace_call, 'conn_start'))  # noqa
        trace.on_connection_create_end.append(partial(trace_call, 'conn_end'))  # noqa
        trace.on_request_start.append(partial(trace_call, 'req_start'))  # noqa
        trace.on_request_end.append(partial(trace_call, 'req_end'))  # noqa
        trace.on_request_chunk_sent.append(partial(trace_call, 'chunk_sent'))  # noqa
        trace.on_response_chunk_received.append(partial(trace_call, 'chunk_received'))  # noqa
        async with ClientSession(
            timeout=ClientTimeout(total=args.t),
            skip_auto_headers=('User-Agent',),
            trace_configs=[trace],
        ) as session:
            while True:
                try:
                    tasks.get(block=False)
                except Empty:
                    return
                try:
                    context = {}
                    headers = {}
                    if args.H:
                        headers = dict(map(str.strip, h.split(':', 1)) for h in args.H)  # noqa
                    if args.chrome:
                        headers['User-Agent'] = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/92.0.4515.159 Safari/537.36'
                    response = await session.request(
                        method=args.m,
                        url=args.url,
                        data=args.b,
                        headers=headers,
                        allow_redirects=False,
                        verify_ssl=not args.insecure,
                        trace_request_ctx=context,
                    )
                    body = await response.read()
                except asyncio.TimeoutError:
                    result = Result(error='timeout')
                except Exception as e:
                    result = Result(error=repr(e))
                else:
                    result = Result(
                        status=response.status,
                        size=len(body),
                        total_time=context.get('req_end', 0) - context.get('req_start', 0),
                        conn_time=context.get('conn_end', 0) - context.get('conn_start', 0),
                        ttfb_time=context.get('chunk_received', 0) - context.get('chunk_sent', 0),
                    )
                results.put(result, block=False)
    except KeyboardInterrupt:
        pass
    except Exception:
        print(repr(e))


def init_screen():
    # screen = curses.initscr()
    # curses.noecho()
    # curses.cbreak()
    pass


def end_screen():
    # curses.nocbreak()
    # curses.echo()
    # curses.endwin()
    pass


def update_screen(results: List[Result]):
    pass


def main():
    args = get_args()
    tasks = Queue()
    results = Queue()
    all_results = []
    workers = [Process(target=worker_start, args=(args, tasks, results)) for _ in range(args.c)]

    for _ in range(args.n):
        tasks.put(args.url, block=False)

    for worker in workers:
        worker.start()

    init_screen()
    start_time = time()
    last_state_time = start_time
    try:
        while True:
            now = time()
            if now - start_time > args.d:
                stats(all_results, now - start_time)
                exit()
            if results.empty():
                if any([worker.is_alive() for worker in workers]):
                    update_screen(all_results)
                    sleep(0.1)
                    continue
                else:
                    stats(all_results, now - start_time)
                    exit()
            if now - last_state_time > 5:
                last_state_time = now
                print(f'Processed {len(all_results)} / {args.n} ({len(all_results) * 100 / args.n:.1f}%)')
            result: Result = results.get()
            all_results.append(result)
    except KeyboardInterrupt:
        pass
    except Exception as e:
        print(repr(e))
        raise
    finally:
        end_screen()


def stats(results: List[Result], total_time: float):
    def percentile(data: List, n: int):
        if len(data) == 1:
            return data[0]
        if len(data) == 0:
            return 0
        return quantiles(data, n=100)[n-1]
    completed = [r for r in results if r.status == 200]
    failed = [r for r in results if r.status != 200]
    print('')
    print(f'Time taken for tests:  {total_time:.2f} seconds')
    print(f'Complete requests:     {len(completed)}')
    print(f'Failed requests:       {len(failed)}')
    print(f'Requests per second:   {total_time / len(results):.2f} [#/sec] (mean)')
    if completed:
        print(f'Time per request:      {sum([r.total_time for r in completed]) * 1000 / len(completed):.3f} [ms] (mean)')
        print(f'Time per request:      {total_time * 1000 / len(completed):.3f} [ms] (mean, across all concurrent requests)')
    print('')
    for code, count in Counter([r.status for r in results if r.status]).items():
        print(f'{code}:  {count}')
    for error, count in Counter([r.error for r in results if r.error]).items():
        print(f'{error}:  {count}')
    print('')
    total_times = [r.total_time * 1000 for r in results if r.total_time]
    ttfb_times = [r.ttfb_time * 1000 for r in results if r.ttfb_time]
    conn_times = [r.conn_time * 1000 for r in results if r.conn_time]
    print(f'            Mean    Min    50%    80%    90%    95%    Max')
    print(
        f'Connect:  {mean(conn_times):6.0f} {min(conn_times):6.0f} {percentile(conn_times, 50):6.0f} '
        f'{percentile(conn_times, 80):6.0f} {percentile(conn_times, 90):6.0f} {percentile(conn_times, 95):6.0f} '
        f'{max(conn_times):6.0f}'
    )
    print(
        f'TTFB:     {mean(ttfb_times):6.0f} {min(ttfb_times):6.0f} {percentile(ttfb_times, 50):6.0f} '
        f'{percentile(ttfb_times, 80):6.0f} {percentile(ttfb_times, 90):6.0f} {percentile(ttfb_times, 95):6.0f} '
        f'{max(ttfb_times):6.0f}'
    )
    print(
        f'Total:    {mean(total_times):6.0f} {min(total_times):6.0f} {percentile(total_times, 50):6.0f} '
        f'{percentile(total_times, 80):6.0f} {percentile(total_times, 90):6.0f} {percentile(total_times, 95):6.0f} '
        f'{max(total_times):6.0f}'
    )


if __name__ == '__main__':
    main()
