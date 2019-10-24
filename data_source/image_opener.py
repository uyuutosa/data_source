import PIL.Image as I
import s3fs


class ImageOpener:
    def __init__(self, header='s3://'):
        self.fs = s3fs.S3FileSystem()
        self.header = header

    def open(self, path):
        if self.header in path:
            return I.open(self.fs.open(path))
        else:
            return I.open(path)

