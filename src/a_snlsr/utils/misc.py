import atexit
import signal
import subprocess
from pathlib import Path

from a_snlsr.logging import get_logger

logger = get_logger()


def is_power_of_two(n: int) -> bool:
    if n <= 0:
        return False
    return (n & (n - 1)) == 0


def launch_tensorboard_process(model_folder: Path):
    logger.info(
        "Launching TensorBoard process on model folder: %s...", model_folder.as_posix()
    )

    tensorboard_command = [
        "tensorboard",
        "--logdir",
        model_folder.as_posix(),
    ]
    log_file = open(model_folder / "tensorboard.log", "w")
    error_file = open(model_folder / "tensorboard_error.log", "w")

    tensorboard_process = subprocess.Popen(
        tensorboard_command,
        stdout=log_file,
        stderr=error_file,
    )

    def signal_handler(sig, frame):
        logger.info("Received signal to terminate TensorBoard process.")
        tensorboard_process.terminate()
        if sig == signal.SIGINT:
            raise KeyboardInterrupt
        elif sig == signal.SIGTERM:
            raise SystemExit

    def cleanup():
        logger.info("Cleaning up TensorBoard process.")
        tensorboard_process.terminate()
        log_file.close()
        error_file.close()

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    atexit.register(cleanup)


def list_tile_names(folder_path: Path):
    file_names = []

    for subpath in folder_path.iterdir():
        if subpath.suffix == ".mat":
            file_names.append(subpath.stem)
    return file_names
