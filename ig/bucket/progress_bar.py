# type: ignore
"""Progress bar callback."""
import sys
import threading


class ProgressBar:
    """Progress bar for calculating and displaying download progress."""

    def __init__(self, client, bucket: str, filename: str) -> None:
        """Initialize with: file name, file size and lock. Set seen_so_far to 0. Set progress bar length.

        Args:
            filename: File to upload.
            bucket: Bucket to upload to.
            client: boto3 client.
        """
        self._filename = filename
        self._size = client.head_object(Bucket=bucket, Key=filename)["ContentLength"]
        self._seen_so_far = 0
        self._lock = threading.Lock()
        self.prog_bar_len = 80

    def __call__(self, bytes_amount: int):
        """Call When called.

        Increments seen_so_far by bytes_amount,
        calculates percentage of seen_so_far/total file size and prints progress bar.

        Args:
            bytes_amount: bytes amount.

        """
        with self._lock:
            self._seen_so_far += bytes_amount
            ratio = round(
                (float(self._seen_so_far) / float(self._size)) * (self.prog_bar_len - 6), 1
            )
            current_length = int(round(ratio))

            percentage = round(100 * ratio / (self.prog_bar_len - 6), 1)

            bars = "+" * current_length
            output = (
                bars
                + " " * (self.prog_bar_len - current_length - len(str(percentage)) - 1)
                + str(percentage)
                + "%"
            )

            if self._seen_so_far != self._size:
                sys.stdout.write(output + "\r")
            else:
                sys.stdout.write(output + "\n")
            sys.stdout.flush()
