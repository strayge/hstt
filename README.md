## hstt - HTTP stress testing tool

[![PyPI version shields.io](https://img.shields.io/pypi/v/hstt.svg)](https://pypi.org/project/hstt/)
[![PyPI pyversions](https://img.shields.io/pypi/pyversions/hstt.svg)](https://pypi.org/project/hstt/)
[![PyPI download month](https://img.shields.io/pypi/dm/hstt.svg)](https://pypi.org/project/hstt/)

![screenshot](https://user-images.githubusercontent.com/2664578/130337933-40b131e0-2f27-4f77-a389-c3ecc0668b00.gif)

### Params

```
usage: hstt [-h] [-n <num>] [-c <num>] [-d <sec>] [-t <sec>] [-H [<header> ...]] [-b <body>]
            [-m <method>] [--debug] [--insecure] [--chrome] [--no-reuse]
            url

HTTP stress testing tool (hstt)

positional arguments:
  url                target URL

optional arguments:
  -h, --help         show this help message and exit
  -n <num>           Total number of requests to perform (default: 10)
  -c <num>           Number of concurrent requests (default: 1)
  -d <sec>           Total duration limit (default: 30)
  -t <sec>           The timeout for each request (default: 30)
  -H [<header> ...]  A request header to be sent
  -b <body>          A request body to be sent
  -m <method>        An HTTP request method for each request (default: GET)
  --debug            Run in debug mode
  --insecure         Skip TLS verification
  --chrome           Use Chrome User-Agent header
  --no-reuse         New connection for each request
```

### Output

```
$ hstt https://example.com -n 3000 -c 10 -t 0.3

## HSTT results
Concurrency:           10 (reusing connection per worker)
Time taken for tests:  17.22 seconds
Complete requests:     2997
Failed requests:       3
Requests per second:   174.21 [#/sec] (mean)
Time per request:      51.818 [ms] (mean)
Time per request:      5.746 [ms] (mean, across all concurrent requests)

## Statuses
200       :   2997 ( 99.90%)
timeout   :      3 (  0.10%)

## Timings (ms)
             Mean    Min    50%    80%    95%    99%    Max
Connect:      171    145    167    180    208    211    240
TTFB:          49     20     50     50     60     63     76
Total:         52     35     50     53     60    200    290
```
