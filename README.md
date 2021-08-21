## hstt - HTTP stress testing tool

### Params

```
positional arguments:
  url                   target URL

optional arguments:
  -h, --help            show this help message and exit
  -n <num>              Total number of requests to perform
  -c <num>              Number of concurrent requests
  -d <sec>              Total duration limit
  -t <sec>              The timeout for each request.
  -H [<header> [<header> ...]]
                        A request header to be sent
  -b <body>             A request body to be sent
  -m <method>           An HTTP request method for each request
  --debug               Run in debug mode
  --insecure            Skip TLS verification
  --chrome              Use Chrome User-Agent header
```

### Output
```
$ hstt https://example.com -c 30 -n 1000 -d 300
Processed 182 / 1000 (18.2%)
Processed 518 / 1000 (51.8%)
Processed 615 / 1000 (61.5%)
Processed 718 / 1000 (71.8%)
Processed 819 / 1000 (81.9%)
Processed 920 / 1000 (92.0%)

Time taken for tests:  34.56 seconds
Complete requests:     1000
Failed requests:       0
Requests per second:   0.03 [#/sec] (mean)
Time per request:      60.452 [ms] (mean)
Time per request:      34.556 [ms] (mean, across all concurrent requests)

200:  1000

            Mean    Min    50%    80%    90%    95%    Max
Connect:     322    147    248    488    685    764    895
TTFB:         48     31     47     53     54     61     70
Total:        60     31     48     53     58     63    949
```
