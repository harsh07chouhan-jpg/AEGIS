class MediaTooLongError(Exception):
    def __init__(self, message="Media exceeds the maximum allowed duration"):
        super().__init__(message)
