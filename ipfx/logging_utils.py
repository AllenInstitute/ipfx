import os
import logging


def configure_logger(cell_dir):

    logging.getLogger().setLevel(logging.INFO)
    logging.basicConfig(filename=os.path.join(cell_dir,"log.txt"))
    stderrLogger = logging.StreamHandler()
    stderrLogger.setFormatter(logging.Formatter(logging.BASIC_FORMAT))
    logging.getLogger().addHandler(stderrLogger)


def log_pretty_header(header, level=1, top_line_break=True, bottom_line_break=True):
    """
    Decorate logging message to make logging output more human readable

    Parameters
    ----------
    header: str
        header message
    level: int
        1 or 2 as in markdown
    top_line_break: bool (True)
        add a blank line at the top
    bottom_line_break: bool (True)
        add a blank line at the bottom
    """

    if top_line_break:
        logging.info("  ")

    header = "***** ***** ***** " + header + " ***** ***** *****"
    logging.info(header)

    if level ==1:
        logging.info("="*len(header))
    elif level == 2:
        logging.info("-"*len(header))

    if bottom_line_break:
        logging.info("  ")

