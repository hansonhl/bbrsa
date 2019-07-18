import logging

def init_logger(no_format=False, print_level=logging.DEBUG, log_file=None,
    log_file_level=logging.DEBUG, log_mode='a'):
    """Initialize logger""" # modified from onmt/utils/logging.py
    default_level = logging.DEBUG
    if no_format:
        log_format = logging.Formatter("%(message)s")
    else:
        log_format = logging.Formatter("[%(asctime)s %(levelname)s] %(message)s")
    logger = logging.getLogger()
    logger.setLevel(default_level)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(print_level)
    console_handler.setFormatter(log_format)
    logger.handlers = [console_handler]

    if log_file and log_file != '':
        file_handler = logging.FileHandler(log_file, mode=log_mode)
        file_handler.setLevel(log_file_level)
        file_handler.setFormatter(log_format)
        logger.addHandler(file_handler)

    return logger

def display(names, preds):
    num_examples = len(preds[0])
    for i in range(num_examples):
        for j, s in enumerate(preds):
            if isinstance(s[i], list):
                s = s[i][0]
            elif isinstance(s[i], str):
                s = s[i]
            else:
                print('Error in pred type!')
            print(names[j] + ': ' + s.strip())
        print('')
