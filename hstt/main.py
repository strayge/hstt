import argparse
import asyncio
import curses
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from functools import partial, wraps
from queue import Empty, Queue
from statistics import mean
from threading import Event, Thread
from time import sleep, time
from types import SimpleNamespace
from typing import Any, Callable, Dict, List, Optional, Union

from aiohttp import ClientSession, ClientTimeout, TraceConfig


@dataclass
class Result:
    timestamp: float
    error: Optional[str] = None
    status: Optional[int] = None
    size: Optional[int] = None
    total_time: Optional[float] = None
    conn_time: Optional[float] = None
    ttfb_time: Optional[float] = None


def worker_start(*args: Any) -> None:
    """Worker entry point."""
    try:
        loop = asyncio.new_event_loop()
        loop.run_until_complete(worker_loop(*args))
    except (KeyboardInterrupt, SystemExit):
        exit(1)
    except Exception as e:
        print(repr(e))


async def worker_loop(args: argparse.Namespace, tasks: Queue, results: Queue, start_event: Event) -> None:
    """Make actual requests in loop."""
    try:
        # trace for tracking time different times inside requests
        async def trace_call(name: str, session: ClientSession, context: SimpleNamespace, params: Any) -> None:
            if name not in context.trace_request_ctx:
                context.trace_request_ctx[name] = time()
        trace = TraceConfig()
        trace.on_connection_create_start.append(partial(trace_call, 'conn_start'))  # noqa
        trace.on_connection_create_end.append(partial(trace_call, 'conn_end'))  # noqa
        trace.on_request_start.append(partial(trace_call, 'req_start'))  # noqa
        trace.on_request_end.append(partial(trace_call, 'req_end'))  # noqa
        trace.on_request_chunk_sent.append(partial(trace_call, 'chunk_sent'))  # noqa
        trace.on_response_chunk_received.append(partial(trace_call, 'chunk_received'))  # noqa

        # wait until all threads will be initialized
        if not start_event.wait(timeout=10):
            return

        if not args.no_reuse:
            # common session for all requests
            session = ClientSession(
                timeout=ClientTimeout(total=args.t),
                skip_auto_headers=('User-Agent',),
                trace_configs=[trace],
            )
        while True:
            if args.no_reuse:
                # new session for each request
                session = ClientSession(
                    timeout=ClientTimeout(total=args.t),
                    skip_auto_headers=('User-Agent',),
                    trace_configs=[trace],
                )
            else:
                # do not reuse cookies with common session
                session.cookie_jar.clear()
            try:
                tasks.get(block=True, timeout=1)
            except Empty:
                # no more tasks, stop worker
                await session.close()
                return
            try:
                context: Dict[str, float] = {}
                headers: Dict[str, str] = {}
                if args.H:
                    headers = dict(map(str.strip, h.split(':', 1)) for h in args.H)  # noqa
                if args.chrome:
                    headers['User-Agent'] = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/92.0.4515.159 Safari/537.36'  # noqa: E501
                response = await session.request(
                    method=args.m,
                    url=args.url,
                    data=args.b,
                    headers=headers,
                    allow_redirects=False,
                    ssl=False if args.insecure else None,
                    trace_request_ctx=context,
                )
                body = await response.read()
            except asyncio.TimeoutError:
                result = Result(timestamp=time(), error='timeout')
            except Exception as e:
                result = Result(timestamp=time(), error=e.__class__.__name__)
            else:
                result = Result(
                    timestamp=time(),
                    status=response.status,
                    size=len(body),
                    total_time=context.get('req_end', 0) - context.get('req_start', 0),
                    conn_time=context.get('conn_end', 0) - context.get('conn_start', 0),
                    ttfb_time=context.get('chunk_received', 0) - context.get('chunk_sent', 0),
                )
            results.put(result, block=True)
            if args.no_reuse:
                await session.close()
    except KeyboardInterrupt:
        exit(1)
    except Exception as e:
        print(repr(e))


def main() -> None:
    """App entry point."""
    args: argparse.Namespace = get_args()
    tasks: Queue = Queue()
    results: Queue = Queue()
    start_event = Event()
    all_results: List[Result] = []
    workers = [
        Thread(
            target=worker_start,
            args=(args, tasks, results, start_event),
            daemon=True,
        )
        for _ in range(args.c)
    ]

    for _ in range(args.n):
        tasks.put(args.url, block=True)

    for worker in workers:
        worker.start()

    screen = init_screen()

    # tell worked to start requests
    start_event.set()
    start_time = time()
    try:
        while True:
            now = time()
            if now - start_time > args.d:
                # total duration exceeded
                stats(args, all_results, now - start_time)
                print(f'\nDuration exceeded ({args.d} seconds).')
                exit()
            if results.empty():
                # no new results
                if any([worker.is_alive() for worker in workers]):
                    # workers still alive, so wait for new results
                    update_screen(screen, args, all_results)
                    sleep(0.5)
                    continue
                else:
                    # no workers, assuming tasks completed
                    stats(args, all_results, now - start_time)
                    exit()
            result: Result = results.get()
            all_results.append(result)
    except KeyboardInterrupt:
        stats(args, all_results, time() - start_time)
        exit(1)
    except Exception:
        raise
    finally:
        end_screen()


class CustomArgsFormatter(argparse.HelpFormatter):
    """Show only actual default values."""
    def _get_help_string(self, action: Any) -> str:
        help: str = action.help
        if action.default not in (argparse.SUPPRESS, None, False):
            help += ' (default: %(default)s)'
        return help


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='HTTP stress testing tool (hstt)', formatter_class=CustomArgsFormatter)
    parser.add_argument('-n', type=int, metavar='<num>', default=10, help='Total number of requests to perform')
    parser.add_argument('-c', type=int, metavar='<num>', default=1, help='Number of concurrent requests')
    parser.add_argument('-d', type=float, metavar='<sec>', default=30, help='Total duration limit')
    parser.add_argument('-t', type=float, metavar='<sec>', default=30, help='The timeout for each request')
    parser.add_argument('-H', metavar='<header>', nargs='*', help='A request header to be sent')
    parser.add_argument('-b', metavar='<body>', help='A request body to be sent')
    parser.add_argument('-m', metavar='<method>', default='GET', help='An HTTP request method for each request')
    parser.add_argument('--debug', action='store_true', help='Run in debug mode')
    parser.add_argument('--insecure', action='store_true', help='Skip TLS verification')
    parser.add_argument('--chrome', action='store_true', help='Use Chrome User-Agent header')
    parser.add_argument('--no-reuse', action='store_true', help='New connection for each request')
    parser.add_argument('url', help='target URL')
    args = parser.parse_args()
    if not (args.url.startswith('http:') or args.url.startswith('https:')):
        print('Error: url should start with protocol')
        exit(1)
    return args


# Stats for TUI / console output

def timing_stats(results: List[Result]) -> List[str]:
    """Calculate and format lines with timings across completed results."""
    def percentile(data: List[float], percent: int) -> Union[float, str]:
        if not data:
            return '-'
        data_sorted = sorted(data)
        pos = max(int(round(percent / 100 * len(data) + 0.5)), 2)
        return data_sorted[pos - 2]

    def format_line(name: str, *values: Union[float, int, str]) -> str:
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


def codes_stats(results: List[Result]) -> List[str]:
    """Calculate and format lines return codes / errors across results."""
    lines = []
    with_codes = Counter([r.status for r in results if r.status])
    with_errors = Counter([r.error for r in results if r.error])
    for code, count in with_codes.items():
        lines.append(f'{code:<10}: {count:>6} ({count * 100 / len(results):6.2f}%)')
    for error, count in with_errors.items():
        lines.append(f'{error:<10}: {count:>6} ({count * 100 / len(results):6.2f}%)')
    return lines


def stats(args: argparse.Namespace, results: List[Result], total_time: float) -> None:
    """Print final stats."""
    completed = [r for r in results if r.status == 200]
    failed = [r for r in results if r.status != 200]
    print('')
    print('## HSTT results')
    print(f'Concurrency:           {args.c}{" (reusing connection per worker)" if not args.no_reuse else ""}')
    print(f'Time taken for tests:  {total_time:.2f} seconds')
    print(f'Complete requests:     {len(completed)}')
    print(f'Failed requests:       {len(failed)}')
    print(f'Requests per second:   {len(results) / total_time:.2f} [#/sec] (mean)')
    if completed:
        print(f'Time per request:      {sum([r.total_time for r in completed]) * 1000 / len(completed):.3f} [ms] (mean)')  # noqa: E501
        print(f'Time per request:      {total_time * 1000 / len(completed):.3f} [ms] (mean, across all concurrent requests)')  # noqa: E501
    print('')
    print('## Statuses')
    for line in codes_stats(results):
        print(line)
    print('')
    print('## Timings (ms)')
    for line in timing_stats(results):
        print(line)


# TUI (curses)

def init_screen() -> Any:
    """Initialize curses."""
    screen = curses.initscr()
    curses.noecho()
    curses.cbreak()
    screen.nodelay(True)
    return screen


def end_screen() -> None:
    """Deinitialize curses."""
    curses.nocbreak()
    curses.echo()
    curses.endwin()


def detect_window_resize(screen) -> None:
    """Detect terminal resize & ^C pressing."""
    # required for getting resize signal
    keycode = None
    while keycode != -1:
        keycode = screen.getch()
        if keycode == 3:
            # curses eat ^C, so we need raise it manually
            raise KeyboardInterrupt()
    # detect different size changed by signal
    if curses.is_term_resized(*screen.getmaxyx()):
        curses.resize_term(0, 0)
        screen.erase()


class SubArea:
    """Analog of subwindow from curses, but without actually without creating any new window.

    Mostly for simplify coordinates calculation.
    Do not have boundary checks at the moment.
    """
    def __init__(self, window, y: int, x: int, h: int, w: int) -> None:
        self.window = window
        self.top = y
        self.left = x
        self.height = h
        self.width = w
        self.border_top = False
        self.border_bottom = False
        self.border_left = False
        self.border_right = False

    def set_border(self, top: bool = False, bottom: bool = False, left: bool = False, right: bool = False) -> None:
        """Enabled borders will be skipped from calculation for draw other elements."""
        self.border_top = top
        self.border_bottom = bottom
        self.border_left = left
        self.border_right = right

    def draw_border(self) -> None:
        if self.border_top:
            self.window.hline(self.top, self.left, curses.ACS_HLINE, self.width)
        if self.border_bottom:
            self.window.hline(self.top + self.height, self.left, curses.ACS_HLINE, self.width)
        if self.border_left:
            self.window.vline(self.top, self.left, curses.ACS_VLINE, self.height)
        if self.border_right:
            self.window.vline(self.top, self.left + self.width, curses.ACS_VLINE, self.width)

    def draw_text(self, y: int, x: int, text: str) -> None:
        """Draw text at specific coords from SubArea start (excluding border)."""
        y_start = self.top + y
        if self.border_top:
            y_start += 1
        x_start = self.left + x
        if self.border_left:
            x_start += 1
        try:
            self.window.addstr(y_start, x_start, text)
        except curses.error as e:
            print(repr(e), y_start, x_start, text)

    def draw_hline(self, y: int, x: int, length: int) -> None:
        """Draw horizontal line at specific coords from SubArea start (excluding border)."""
        if not length:
            return
        y_start = self.top + y
        if self.border_top:
            y_start += 1
        x_start = self.left + x
        if self.border_left:
            x_start += 1
        try:
            self.window.hline(y_start, x_start, curses.ACS_BLOCK, length)
        except curses.error as e:
            print(repr(e), y_start, x_start, length)

    def draw_vline(self, y: int, x: int, length: int) -> None:
        """Draw vertical line at specific coords from SubArea start (excluding border)."""
        if not length:
            return
        y_start = self.top + y
        if self.border_top:
            y_start += 1
        x_start = self.left + x
        if self.border_left:
            x_start += 1
        try:
            self.window.vline(y_start, x_start, curses.ACS_BLOCK, length)
        except curses.error as e:
            print(repr(e), y_start, x_start, length)


def update_screen(screen, args: argparse.Namespace, results: List[Result]) -> None:
    """Redraw curses UI."""
    detect_window_resize(screen)
    screen.erase()

    max_y, max_x = screen.getmaxyx()
    timing_lines = timing_stats(results)
    codes_lines = codes_stats(results)
    timings_width = max([len(line) for line in timing_lines])
    bottom_height = max(len(timing_lines), len(codes_lines))

    # progress bar
    area_progress = SubArea(screen, max_y - 3, 1, 2, max_x - 2)
    area_progress.set_border(top=True)
    completed_percent = len(results) / args.n
    area_progress.draw_text(0, 0, f'{len(results):>6} / {args.n:>6}')
    area_progress.draw_text(0, area_progress.width - 4, f'{len(results) * 100 / args.n:.0f}%')
    if completed_percent:
        area_progress.draw_hline(0, 17, int(completed_percent * (area_progress.width - 22)) or 1)
    area_progress.draw_border()

    # timings
    area_timings = SubArea(
        screen, max_y - 1 - area_progress.height - bottom_height, max_x - timings_width - 4,
        bottom_height, timings_width + 2,
    )
    area_timings.set_border(left=True)
    for i, line in enumerate(timing_lines):
        area_timings.draw_text(i, 1, line)
    area_timings.draw_border()

    # codes
    area_codes = SubArea(screen, area_timings.top, 1, bottom_height, max_x - area_timings.width - 4)
    for i, line in enumerate(codes_lines):
        area_codes.draw_text(i, 1, line)

    # chart times
    chart_time = SubArea(screen, 1, 1, (area_timings.top - 2) // 2, max_x - 2)
    chart_time.set_border(bottom=True)
    # axis
    max_total_time = max([int(r.total_time * 1000) for r in results if r.total_time] or [0])
    chart_time.draw_text(0, 0, f'{max_total_time:>5}')
    chart_time.draw_text(chart_time.height // 2, 0, f'{max_total_time // 2:>5}')
    chart_time.draw_text(chart_time.height - 1, 0, f'{0:>5}')
    # data
    time_per_ts = defaultdict(list)
    for r in results:
        if r.total_time:
            ts = int(r.timestamp)
            time_per_ts[ts].append(r.total_time)
    max_time_per_ts = {ts: max(values) for ts, values in time_per_ts.items()}
    # plot
    for i, ts in enumerate(list(sorted(max_time_per_ts.keys()))[-(chart_time.width - 7):]):
        value_ms = max_time_per_ts[ts]
        value = int(value_ms * 1000 * chart_time.height / max_total_time)
        chart_time.draw_vline(chart_time.height - value, 6 + i, value or 1)
    chart_time.draw_border()

    # chart count
    chart_count = SubArea(
        screen,
        chart_time.top + chart_time.height + 1, chart_time.left,
        area_timings.top - chart_time.top - chart_time.height - 2, chart_time.width,
    )
    chart_count.set_border(bottom=True)
    # data
    resp_per_ts: Dict[int, int] = defaultdict(int)
    for r in results:
        if r.status:
            ts = int(r.timestamp)
            resp_per_ts[ts] += 1
    # axis
    max_rps = max(resp_per_ts.values() or [0])
    chart_count.draw_text(0, 0, f'{max_rps:>5}')
    chart_count.draw_text(chart_count.height // 2, 0, f'{max_rps // 2:>5}')
    chart_count.draw_text(chart_count.height - 1, 0, f'{0:>5}')
    # plot
    for i, ts in enumerate(list(sorted(resp_per_ts.keys()))[-(chart_count.width - 7):]):
        count = resp_per_ts[ts]
        value = int(count * chart_count.height / max_rps)
        chart_count.draw_vline(chart_count.height - value, 6 + i, value)
    chart_count.draw_border()

    screen.border()
    screen.refresh()


# Run app

if sys.platform == 'win32':
    # https://bugs.python.org/issue39232
    # https://github.com/aio-libs/aiohttp/issues/4324
    def silence_event_loop_closed(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            try:
                return func(*args, **kwargs)
            except RuntimeError as e:
                if str(e) != 'Event loop is closed':
                    raise

        return wrapper

    from asyncio.proactor_events import _ProactorBasePipeTransport  # noqa
    _ProactorBasePipeTransport.__del__ = silence_event_loop_closed(_ProactorBasePipeTransport.__del__)


if __name__ == '__main__':
    main()
