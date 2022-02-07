from config.Config import Config

class Video_pair:
    def __init__(self, video_a, video_b):
        self.video_a = video_a
        self.video_b = video_b

    def is_parsed(self):
        """Checks whether both videos in the pair are parsed to the frames folder on the disc"""
        return self.video_a.is_parsed() and self.video_b.is_parsed()

    def parse(self):
        """Parses video pair to the frames folder on the disc"""
        self.video_a.parse()
        self.video_b.parse()

    def generate_frame_pairs_from_video(self):
        """Returns a generator of pairs of frames directly from the video"""
        return list(zip(self.video_a.generate_frames_from_video(), self.video_b.generate_frames_from_video()))

    def generate_frame_pairs_from_folder(self):
        """Returns a generator of pairs of frames directly from the video"""
        return list(zip(self.video_a.generate_frames_from_folder(), self.video_b.generate_frames_from_folder()))
        
    def generate_paths_to_frames(self):
        """Returns a generator of pairs of frames directly from the video"""
        return list(zip(self.video_a.generate_paths_to_frames(), self.video_b.generate_paths_to_frames()))