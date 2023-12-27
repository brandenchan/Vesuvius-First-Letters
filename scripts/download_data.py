from vesuvius.downloader import download_segment_files
from tap import tap

class Arguments(tap):
    segment_id: str

args = Arguments().parse_args().as_dict()

download_segment_files(
    segment_id=args["segment_id"],
    save_dir="data"
)