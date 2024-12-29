import multiprocessing

# Number of worker processes - adjust based on CPU cores
workers = multiprocessing.cpu_count() * 2 + 1

# Number of threads per worker
threads = 4

# Maximum number of pending connections
backlog = 2048

# Maximum number of requests a worker will process before restarting
max_requests = 10000
max_requests_jitter = 50

# Timeout for worker processes (5 minutes)
timeout = 300

# Keep-alive timeout
keepalive = 5

# Log settings
loglevel = "info"
accesslog = "-"
errorlog = "-"
access_log_format = '%(h)s %(l)s %(u)s %(t)s "%(r)s" %(s)s %(b)s "%(f)s" "%(a)s"'

# Bind address
bind = "0.0.0.0:8008"

# Worker class
worker_class = "gevent"

# Process name
proc_name = "selector_server"

# Preload app for faster worker spawning
preload_app = True

# Graceful timeout
graceful_timeout = 30

# Server mechanics
daemon = False
pidfile = None
umask = 0
user = None
group = None
tmp_upload_dir = None

# Logging
capture_output = True
enable_stdio_inheritance = False
