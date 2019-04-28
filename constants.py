FILE_ENCODING = [1, 0, 0]
PROCESS_ENCODING = [0, 1, 0]
SOCKET_ECODING = [0, 0, 1]

CMD_LINE_SIZE = 12
LOGIN_NAME_SIZE = 6
EUID_SIZE = 5
BINARY_FILE_SIZE = 4

EMPTY_CMD_LINE = [0] * CMD_LINE_SIZE
EMPTY_LOGIN_NAME = [0] * LOGIN_NAME_SIZE
EMPTY_EUID = [0] * EUID_SIZE
EMPTY_BINARY_FILE = [0] * BINARY_FILE_SIZE

CMD_LINE_CHOICES = range(0, CMD_LINE_SIZE)
LOGIN_NAME_CHOICES = range(0, LOGIN_NAME_SIZE)
EUID_CHOICES = range(0, EUID_SIZE)
BINARY_FILE_CHOICES = range(0, BINARY_FILE_SIZE)

CMD_LINE = [
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]
]
