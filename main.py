import argparse
import asyncio
import curses
import platform
from collections import Counter, defaultdict
from dataclasses import dataclass
from functools import partial, wraps
from multiprocessing import Process, Queue
from queue import Empty
from statistics import mean
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
    except Exception as e:
        print(repr(e))


@dataclass
class Result:
    timestamp: float
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
                        timestamp=time(),
                        status=response.status,
                        size=len(body),
                        total_time=context.get('req_end', 0) - context.get('req_start', 0),
                        conn_time=context.get('conn_end', 0) - context.get('conn_start', 0),
                        ttfb_time=context.get('chunk_received', 0) - context.get('chunk_sent', 0),
                    )
                results.put(result, block=False)
                sleep(0.5)
    except KeyboardInterrupt:
        pass
    except Exception:
        print(repr(e))


def init_screen():
    screen = curses.initscr()
    curses.noecho()
    curses.cbreak()
    screen.nodelay(True)
    return screen


def end_screen():
    curses.nocbreak()
    curses.echo()
    curses.endwin()


def detect_window_resize(screen):
    # required for getting resize signal
    keycode = None
    while keycode != -1:
        keycode = screen.getch()
    # detect different size changed by signal
    if curses.is_term_resized(*screen.getmaxyx()):
        curses.resize_term(0, 0)
        screen.erase()


def update_screen(screen, args, results: List[Result]):
    detect_window_resize(screen)
    screen.erase()

    timing_lines = timing_stats(results)
    codes_lines = codes_stats(results)
    max_total_time = max([int(r.total_time * 1000) for r in results if r.total_time] or [0])
    border_x = 2
    border_y = 1
    max_y, max_x = screen.getmaxyx()

    bottom_height = max(len(timing_lines), len(codes_lines))
    bottom_y = max_y - bottom_height - 2 - border_y
    timings_width = max([len(line) for line in timing_lines])
    bottom_right_x = max_x - timings_width - border_x

    top_left = border_x + 5
    top_width = max_x - border_x - top_left
    top_height = bottom_y - 2 * border_y

    # chart axis
    screen.addstr(border_y, border_x, f'{max_total_time:>4}')
    screen.addstr(border_y + top_height // 2, border_x, f'{max_total_time // 2:>4}')
    screen.addstr(border_y + top_height - 1, border_x, f'{0:>4}')

    # chart
    time_per_ts = defaultdict(list)
    for r in results:
        if r.total_time:
            ts = int(r.timestamp)
            time_per_ts[ts].append(r.total_time)
    max_time_per_ts = {ts: max(values) for ts, values in time_per_ts.items()}

    for i, ts in enumerate(list(sorted(max_time_per_ts.keys()))[-top_width:]):
        value_ms = max_time_per_ts[ts]
        value = int(value_ms * 1000 * top_height / max_total_time)
        try:
            screen.vline(border_y + top_height - value, top_left + i, curses.ACS_BLOCK, value)
        except curses.error as e:
            print(repr(e), border_y + top_height - value, top_left + i, curses.ACS_BLOCK, value)

    # splitter
    screen.hline(bottom_y - 1, 1, curses.ACS_HLINE, max_x)

    # codes
    for i, line in enumerate(codes_lines):
        screen.addstr(bottom_y + i, border_x, line)

    # vertical splitter
    screen.vline(bottom_y, bottom_right_x - 2, curses.ACS_VLINE, bottom_height)

    # timings
    for i, line in enumerate(timing_lines):
        screen.addstr(bottom_y + i, bottom_right_x, line)

    # splitter
    screen.hline(max_y - border_y - 2, 1, curses.ACS_HLINE, max_x)

    # progress bar
    last_line = max_y - 2
    completed_percent = len(results) / args.n
    screen.addstr(last_line, border_x, f'{len(results):>6} / {args.n:>6}')
    screen.addstr(last_line, max_x - border_x - 4, f'{len(results) * 100 / args.n:.0f}%')
    if completed_percent:
        screen.hline(last_line, 20, curses.ACS_BLOCK, int(completed_percent * (max_x - 28)) or 1)

    screen.border()
    screen.refresh()


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

    screen = init_screen()
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
                    update_screen(screen, args, all_results)
                    sleep(0.5)
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
        stats(all_results, time() - start_time)
    except Exception as e:
        print(repr(e))
        raise
    finally:
        end_screen()


def timing_stats(results: List[Result]):
    def percentile(data, percent: int):
        if not data:
            return '-'
        data_sorted = sorted(data)
        pos = max(int(round(percent / 100 * len(data) + 0.5)), 2)
        return data_sorted[pos - 2]

    def format_line(name, *values):
        line = f'{name:<10s}'
        for value in values:
            if isinstance(value, float):
                line += f' {value:6.0f}'
            else:
                line += f' {value:>6}'
        return line

    total_times = [r.total_time * 1000 for r in results if r.total_time]
    ttfb_times = [r.ttfb_time * 1000 for r in results if r.ttfb_time]
    conn_times = [r.conn_time * 1000 for r in results if r.conn_time]

    percentiles = (50, 80, 95, 99)
    lines = [
        format_line(
            '', 'Mean', 'Min', *(f'{p}%' for p in percentiles), 'Max',
        ),
        format_line(
            'Connect:', 
            mean(conn_times) if conn_times else '-',
            min(conn_times) if conn_times else '-',
            *(percentile(conn_times, p) for p in percentiles),
            max(conn_times) if conn_times else '-',
        ),
        format_line(
            'TTFB:',
            mean(ttfb_times) if ttfb_times else '-', 
            min(ttfb_times) if ttfb_times else '-',
            *(percentile(ttfb_times, p) for p in percentiles),
            max(ttfb_times) if ttfb_times else '-',
        ),
        format_line(
            'Total:', 
            mean(total_times) if total_times else '-',
            min(total_times) if total_times else '-',
            *(percentile(total_times, p) for p in percentiles),
            max(total_times) if total_times else '-',
        ),
    ]
    return lines


def codes_stats(results: List[Result]):
    lines = []
    with_codes = Counter([r.status for r in results if r.status])
    with_errors = Counter([r.error for r in results if r.error])
    for code, count in with_codes.items():
        lines.append(f'{code}: {count:>6} ({count * 100 / len(results):6.2f}%)')
    for error, count in with_errors.items():
        lines.append(f'{error}: {count:>6} ({count * 100 / len(results):6.2f}%)')
    return lines


def stats(results: List[Result], total_time: float):
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
    for line in codes_stats(results):
        print(line)
    print('')
    for line in timing_stats(results):
        print(line)


if __name__ == '__main__':
    main()
