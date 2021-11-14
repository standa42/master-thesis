class video_metadata:
    def __init__(self, year, month, day, hour, minute, second, camera, file_extension):
        self.year = year
        self.month = month
        self.day = day
        self.hour = hour
        self.minute = minute
        self.second = second
        self.camera = camera
        self.file_extension = file_extension
    
    def __str__(self):
        return (f"{self.year}_{self.month}_{self.day}_{self.hour}_{self.minute}_{self.second}_{self.camera}.{self.file_extension}")

    def day_string(self):
        return (f"{self.year}_{self.month}_{self.day}")

    def time_string(self):
        return (f"{self.hour}_{self.minute}_{self.second}")

    def day_time_string(self):
        return (f"{self.year}_{self.month}_{self.day}_{self.hour}_{self.minute}_{self.second}")